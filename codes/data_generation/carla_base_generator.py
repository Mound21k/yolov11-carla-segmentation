#!/usr/bin/env python3
"""
Dynamic CARLA Dataset Generator - Base Class
===========================================

Enhanced base class for generating diverse YOLOv11 instance segmentation datasets
with dynamic ego vehicle positioning and comprehensive object coverage.

FIXES:
- Robust actor cleanup to prevent "failed to destroy actor" errors
- Safe division in statistics calculations
- Improved error handling

Author: MLDL Assistant  
Date: June 2025
Assignment: YOLOv11 Instance Segmentation in Adverse Conditions
"""

import os
import sys
import glob
import json
import time
import random
import logging
import numpy as np
import cv2
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import math
import threading
import queue

# CARLA imports
import carla

# Configure logging with logs directory
Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/carla_dynamic_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DynamicCARLADatasetGenerator:
    """
    Enhanced CARLA dataset generator with dynamic positioning and comprehensive object coverage.
    Ensures all 4 target classes are properly represented in diverse scenarios.
    """
    
    def __init__(self, host='localhost', port=2000, output_dir='carla_dynamic_dataset'):
        """Initialize the dynamic dataset generator."""
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        
        # Create organized directory structure
        self.setup_directories()
        
        # Target classes for YOLOv11 instance segmentation
        self.target_classes = {
            'vehicle.car': 'car',
            'vehicle.bus': 'bus', 
            'vehicle.truck': 'bus',  # Include trucks as buses
            'walker.pedestrian': 'pedestrian',
            'traffic.traffic_light': 'traffic_light'
        }
        
        # YOLO class mapping (0-indexed)
        self.yolo_class_mapping = {
            'car': 0,
            'bus': 1, 
            'pedestrian': 2,
            'traffic_light': 3
        }
        
        # Enhanced camera configuration
        self.camera_config = {
            'image_size_x': 1920,
            'image_size_y': 1080,
            'fov': 90,
            'sensor_tick': 0.0  # Fastest possible
        }
        
        # Traffic configuration for comprehensive coverage
        self.traffic_config = {
            'num_vehicles': 150,      # Increased for better coverage
            'num_walkers': 40,        # More pedestrians
            'safe_distance': 2.0,     # Minimum spawn distance
            'vehicle_spawn_attempts': 300,
            'walker_spawn_attempts': 80,
            'min_cars_required': 50,  # Minimum cars for valid scene
            'min_buses_required': 8,  # Minimum buses for valid scene
            'min_pedestrians_required': 15,  # Minimum pedestrians
            'min_traffic_lights_required': 5   # Minimum traffic lights
        }
        
        # Dynamic positioning configuration
        self.positioning_config = {
            'num_positions_per_session': 5,    # Different ego positions per weather
            'movement_radius': 100,             # Max distance for repositioning
            'frames_per_position': 60,          # Frames to collect per position
            'position_change_interval': 10,     # Seconds between repositions
            'enable_ego_movement': True         # Enable continuous movement
        }
        
        # Frame synchronization
        self.sync_config = {
            'max_frame_time_diff': 0.05,  # 50ms max difference
            'sync_timeout': 5.0,          # Timeout for frame sync
            'buffer_size': 10             # Frame buffer size
        }
        
        # Initialize CARLA connection variables
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sensors = {}
        self.actor_list = []
        
        # Frame synchronization
        self.frame_queue = queue.Queue(maxsize=self.sync_config['buffer_size'])
        self.sync_lock = threading.Lock()
        self.current_frames = {}
        
        # Data collection tracking
        self.collection_stats = defaultdict(lambda: defaultdict(int))
        self.position_counter = 0
        
        # Spawn point management
        self.vehicle_spawn_points = []
        self.ego_spawn_points = []
        self.traffic_lights = []
        
    def setup_directories(self):
        """Create organized directory structure for dataset storage."""
        directories = [
            'images/rgb',
            'images/segmentation', 
            'annotations',
            'metadata',
            'logs',
            'visualizations'
        ]
        
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset directories created at: {self.output_dir}")
    
    def connect_to_carla(self):
        """Establish robust connection to CARLA server."""
        max_retries = 5
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connecting to CARLA server (attempt {attempt + 1})")
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(30.0)
                
                # Test connection
                version = self.client.get_server_version()
                logger.info(f"Connected to CARLA server version: {version}")
                
                # Load Town01 for comprehensive object coverage
                logger.info("Loading Town01...")
                self.world = self.client.load_world('Town01')
                time.sleep(5.0)  # Allow complete map loading
                
                # Configure synchronous mode
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05  # 20 FPS
                settings.no_rendering_mode = False   # Keep rendering for data
                self.world.apply_settings(settings)
                
                logger.info("CARLA connection and synchronous mode configured")
                return True
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 1.5
        
        logger.error("Failed to connect to CARLA after all retries")
        return False
    
    def initialize_spawn_points(self):
        """Initialize and categorize all available spawn points."""
        # Get vehicle spawn points
        self.vehicle_spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(self.vehicle_spawn_points)
        
        # Select strategic ego spawn points (well-distributed)
        total_points = len(self.vehicle_spawn_points)
        ego_indices = [
            0, total_points//4, total_points//2, 
            3*total_points//4, total_points-1
        ]
        self.ego_spawn_points = [self.vehicle_spawn_points[i] for i in ego_indices]
        
        # Get traffic lights
        self.traffic_lights = list(self.world.get_actors().filter('traffic.traffic_light'))
        
        logger.info(f"Initialized spawn points:")
        logger.info(f"  Vehicle spawn points: {len(self.vehicle_spawn_points)}")
        logger.info(f"  Ego spawn points: {len(self.ego_spawn_points)}")
        logger.info(f"  Traffic lights: {len(self.traffic_lights)}")
    
    def spawn_ego_vehicle_at_position(self, position_index):
        """Spawn ego vehicle at specific position."""
        if self.ego_vehicle:
            self.safe_destroy_actor(self.ego_vehicle)
            self.ego_vehicle = None
        
        blueprint_library = self.world.get_blueprint_library()
        ego_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        ego_bp.set_attribute('color', '255,0,0')  # Red for visibility
        
        spawn_point = self.ego_spawn_points[position_index % len(self.ego_spawn_points)]
        
        # Try spawning with small variations if needed
        for attempt in range(5):
            try:
                if attempt > 0:
                    # Add small random offset for collision avoidance
                    offset_transform = carla.Transform(
                        carla.Location(
                            spawn_point.location.x + random.uniform(-2, 2),
                            spawn_point.location.y + random.uniform(-2, 2),
                            spawn_point.location.z + 0.5
                        ),
                        spawn_point.rotation
                    )
                    self.ego_vehicle = self.world.spawn_actor(ego_bp, offset_transform)
                else:
                    self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
                
                self.actor_list.append(self.ego_vehicle)
                logger.info(f"Ego vehicle spawned at position {position_index}: {spawn_point.location}")
                return True
                
            except Exception as e:
                logger.debug(f"Ego spawn attempt {attempt + 1} failed: {e}")
        
        logger.error(f"Failed to spawn ego vehicle at position {position_index}")
        return False
    
    def setup_synchronized_sensors(self):
        """Setup RGB and segmentation cameras with enhanced synchronization."""
        if not self.ego_vehicle:
            logger.error("Cannot setup sensors: ego vehicle not spawned")
            return False
        
        blueprint_library = self.world.get_blueprint_library()
        
        # Camera mounting position for optimal FOV
        camera_transform = carla.Transform(
            carla.Location(x=2.5, z=1.8),   # Slightly forward and elevated
            carla.Rotation(pitch=-10.0)      # Slight downward angle
        )
        
        # RGB Camera
        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        for key, value in self.camera_config.items():
            rgb_bp.set_attribute(key, str(value))
        
        self.sensors['rgb'] = self.world.spawn_actor(
            rgb_bp, camera_transform, attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.sensors['rgb'])
        
        # Instance Segmentation Camera
        seg_bp = blueprint_library.find('sensor.camera.instance_segmentation')
        for key, value in self.camera_config.items():
            seg_bp.set_attribute(key, str(value))
        
        self.sensors['segmentation'] = self.world.spawn_actor(
            seg_bp, camera_transform, attach_to=self.ego_vehicle
        )
        self.actor_list.append(self.sensors['segmentation'])
        
        # Setup enhanced synchronization callbacks
        self.setup_enhanced_sensor_callbacks()
        
        logger.info("Synchronized RGB and segmentation cameras attached")
        return True
    
    def setup_enhanced_sensor_callbacks(self):
        """Setup enhanced sensor callbacks with precise frame synchronization."""
        self.current_frames = {'rgb': None, 'segmentation': None}
        
        def rgb_callback(image):
            with self.sync_lock:
                self.current_frames['rgb'] = {
                    'image': image,
                    'timestamp': image.timestamp,
                    'frame_number': image.frame
                }
                self.check_synchronized_frame()
        
        def segmentation_callback(image):
            with self.sync_lock:
                self.current_frames['segmentation'] = {
                    'image': image,
                    'timestamp': image.timestamp,
                    'frame_number': image.frame
                }
                self.check_synchronized_frame()
        
        self.sensors['rgb'].listen(rgb_callback)
        self.sensors['segmentation'].listen(segmentation_callback)
    
    def check_synchronized_frame(self):
        """Check for properly synchronized frame pairs."""
        rgb_data = self.current_frames.get('rgb')
        seg_data = self.current_frames.get('segmentation')
        
        if rgb_data and seg_data:
            # Check frame number synchronization (preferred)
            if rgb_data['frame_number'] == seg_data['frame_number']:
                synchronized_frame = {
                    'rgb': rgb_data['image'],
                    'segmentation': seg_data['image'],
                    'timestamp': rgb_data['timestamp'],
                    'frame_number': rgb_data['frame_number']
                }
                
                try:
                    self.frame_queue.put_nowait(synchronized_frame)
                except queue.Full:
                    # Remove oldest frame if buffer full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(synchronized_frame)
                    except queue.Empty:
                        pass
                
                # Reset for next frame
                self.current_frames = {'rgb': None, 'segmentation': None}
            
            # Check timestamp synchronization as fallback
            elif abs(rgb_data['timestamp'] - seg_data['timestamp']) <= self.sync_config['max_frame_time_diff']:
                synchronized_frame = {
                    'rgb': rgb_data['image'],
                    'segmentation': seg_data['image'],
                    'timestamp': min(rgb_data['timestamp'], seg_data['timestamp']),
                    'frame_number': rgb_data['frame_number']
                }
                
                try:
                    self.frame_queue.put_nowait(synchronized_frame)
                except queue.Full:
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(synchronized_frame)
                    except queue.Empty:
                        pass
                
                self.current_frames = {'rgb': None, 'segmentation': None}
    
    def spawn_comprehensive_traffic(self):
        """Spawn traffic ensuring all 4 target classes are well represented."""
        logger.info("Spawning comprehensive traffic with guaranteed class coverage")
        
        blueprint_library = self.world.get_blueprint_library()
        
        # Categorize vehicle blueprints
        car_blueprints = []
        bus_blueprints = []
        
        for bp in blueprint_library.filter('vehicle.*'):
            if any(bus_type in bp.id.lower() for bus_type in ['bus', 'truck']):
                bus_blueprints.append(bp)
            else:
                car_blueprints.append(bp)
        
        logger.info(f"Available blueprints: {len(car_blueprints)} cars, {len(bus_blueprints)} buses")
        
        # Spawn vehicles with strategic distribution
        vehicles_spawned = self.spawn_strategic_vehicles(car_blueprints, bus_blueprints)
        
        # Spawn pedestrians with enhanced algorithms
        pedestrians_spawned = self.spawn_enhanced_pedestrians(blueprint_library)
        
        # Count traffic lights
        traffic_lights_count = len(self.traffic_lights)
        
        logger.info(f"Traffic spawning results:")
        logger.info(f"  Cars: {vehicles_spawned['cars']}")
        logger.info(f"  Buses: {vehicles_spawned['buses']}")
        logger.info(f"  Pedestrians: {pedestrians_spawned}")
        logger.info(f"  Traffic lights: {traffic_lights_count}")
        
        # Validate minimum requirements
        valid_scene = self.validate_scene_requirements(
            vehicles_spawned, pedestrians_spawned, traffic_lights_count
        )
        
        if valid_scene:
            # Allow traffic to settle and start moving
            for i in range(50):
                self.world.tick()
                time.sleep(0.02)
            logger.info("Traffic spawning completed successfully")
        
        return valid_scene
    
    def spawn_strategic_vehicles(self, car_blueprints, bus_blueprints):
        """Spawn vehicles with strategic placement for comprehensive coverage."""
        cars_spawned = 0
        buses_spawned = 0
        
        # Ensure minimum bus spawning first
        min_buses = self.traffic_config['min_buses_required']
        bus_spawn_points = self.vehicle_spawn_points[:min_buses * 3]  # Extra points for buses
        
        for i, spawn_point in enumerate(bus_spawn_points):
            if buses_spawned >= min_buses:
                break
                
            if bus_blueprints:
                try:
                    bp = random.choice(bus_blueprints)
                    if bp.has_attribute('color'):
                        color = random.choice(bp.get_attribute('color').recommended_values)
                        bp.set_attribute('color', color)
                    
                    vehicle = self.world.spawn_actor(bp, spawn_point)
                    vehicle.set_autopilot(True)
                    self.actor_list.append(vehicle)
                    buses_spawned += 1
                    
                except Exception as e:
                    logger.debug(f"Bus spawn failed at point {i}: {e}")
        
        # Spawn remaining vehicles (mix of cars and additional buses)
        remaining_points = self.vehicle_spawn_points[min_buses * 3:]
        target_remaining = self.traffic_config['num_vehicles'] - buses_spawned
        
        for i, spawn_point in enumerate(remaining_points[:target_remaining]):
            # 80% cars, 20% additional buses if available
            if bus_blueprints and random.random() < 0.2:
                blueprints = bus_blueprints
                is_bus = True
            else:
                blueprints = car_blueprints
                is_bus = False
            
            try:
                bp = random.choice(blueprints)
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)
                
                vehicle = self.world.spawn_actor(bp, spawn_point)
                vehicle.set_autopilot(True)
                self.actor_list.append(vehicle)
                
                if is_bus:
                    buses_spawned += 1
                else:
                    cars_spawned += 1
                    
            except Exception as e:
                logger.debug(f"Vehicle spawn failed at point {i}: {e}")
        
        return {'cars': cars_spawned, 'buses': buses_spawned}
    
    def spawn_enhanced_pedestrians(self, blueprint_library):
        """Enhanced pedestrian spawning with better success rate."""
        pedestrians_spawned = 0
        target_pedestrians = self.traffic_config['num_walkers']
        
        # Get pedestrian blueprints
        walker_bps = list(blueprint_library.filter('walker.pedestrian.*'))
        if not walker_bps:
            logger.error("No pedestrian blueprints available")
            return 0
        
        # Get walker controller
        try:
            walker_controller_bp = blueprint_library.find('controller.ai.walker')
            has_controller = True
        except:
            logger.warning("Walker controller not available")
            has_controller = False
        
        # Generate enhanced spawn points
        spawn_points = self.generate_enhanced_pedestrian_points()
        
        # Spawn pedestrians
        for i, spawn_point in enumerate(spawn_points[:target_pedestrians]):
            try:
                walker_bp = random.choice(walker_bps)
                pedestrian = self.world.spawn_actor(walker_bp, spawn_point)
                self.actor_list.append(pedestrian)
                
                # Add controller for realistic movement
                if has_controller and random.random() < 0.8:
                    try:
                        controller = self.world.spawn_actor(
                            walker_controller_bp, carla.Transform(), pedestrian
                        )
                        self.actor_list.append(controller)
                        controller.start()
                        controller.set_max_speed(random.uniform(1.0, 2.0))
                        
                        # Set random movement
                        if i < len(spawn_points) - 1:
                            target = spawn_points[i + 1].location
                            controller.go_to_location(target)
                        
                    except Exception as controller_error:
                        logger.debug(f"Controller spawn failed: {controller_error}")
                
                pedestrians_spawned += 1
                
            except Exception as e:
                logger.debug(f"Pedestrian spawn failed at point {i}: {e}")
        
        return pedestrians_spawned
    
    def generate_enhanced_pedestrian_points(self):
        """Generate enhanced pedestrian spawn points around ego vehicle."""
        spawn_points = []
        
        if not self.ego_vehicle:
            return spawn_points
        
        ego_location = self.ego_vehicle.get_location()
        
        # Multiple spawning strategies for better coverage
        
        # Strategy 1: Sidewalk grid pattern
        grid_range = 60
        grid_step = 6
        
        for x_offset in range(-grid_range, grid_range + 1, grid_step):
            for y_offset in range(-grid_range, grid_range + 1, grid_step):
                # Skip central area (road)
                if abs(x_offset) < 15 and abs(y_offset) < 15:
                    continue
                
                location = carla.Location(
                    x=ego_location.x + x_offset,
                    y=ego_location.y + y_offset,
                    z=ego_location.z + 0.5
                )
                spawn_points.append(carla.Transform(location))
        
        # Strategy 2: Circular patterns (crosswalk simulation)
        for radius in [20, 35, 50]:
            for angle_deg in range(0, 360, 30):
                angle_rad = math.radians(angle_deg)
                x = ego_location.x + radius * math.cos(angle_rad)
                y = ego_location.y + radius * math.sin(angle_rad)
                
                location = carla.Location(x=x, y=y, z=ego_location.z + 0.5)
                spawn_points.append(carla.Transform(location))
        
        # Strategy 3: Traffic light vicinity
        for traffic_light in self.traffic_lights[:10]:  # Limit to prevent overcrowding
            tl_location = traffic_light.get_location()
            
            # Add pedestrian points near traffic lights
            for offset in [(-5, -5), (-5, 5), (5, -5), (5, 5)]:
                location = carla.Location(
                    x=tl_location.x + offset[0],
                    y=tl_location.y + offset[1],
                    z=tl_location.z + 0.5
                )
                spawn_points.append(carla.Transform(location))
        
        random.shuffle(spawn_points)
        return spawn_points
    
    def validate_scene_requirements(self, vehicles_spawned, pedestrians_spawned, traffic_lights_count):
        """Validate that scene meets minimum requirements for all 4 classes."""
        cars = vehicles_spawned['cars']
        buses = vehicles_spawned['buses']
        
        requirements = {
            'cars': (cars, self.traffic_config['min_cars_required']),
            'buses': (buses, self.traffic_config['min_buses_required']),
            'pedestrians': (pedestrians_spawned, self.traffic_config['min_pedestrians_required']),
            'traffic_lights': (traffic_lights_count, self.traffic_config['min_traffic_lights_required'])
        }
        
        all_valid = True
        for class_name, (actual, required) in requirements.items():
            if actual < required:
                logger.warning(f"Insufficient {class_name}: {actual}/{required}")
                all_valid = False
            else:
                logger.info(f"[OK] {class_name}: {actual}/{required}")
        
        return all_valid
    
    def safe_destroy_actor(self, actor):
        """Safely destroy an actor with error handling."""
        if actor is None:
            return True
            
        try:
            # Check if actor still exists in world
            if self.world and hasattr(actor, 'id'):
                current_actors = self.world.get_actors()
                if current_actors.find(actor.id):
                    actor.destroy()
                    return True
            return True
        except Exception as e:
            logger.debug(f"Actor destruction failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up CARLA actors and restore settings with improved error handling."""
        logger.info("Cleaning up CARLA environment")
        
        try:
            # Stop sensors first
            for sensor in self.sensors.values():
                if sensor and hasattr(sensor, 'stop'):
                    try:
                        sensor.stop()
                    except:
                        pass
            
            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Enhanced actor cleanup
            if self.world:
                try:
                    # Get fresh actor list from world
                    current_world_actors = self.world.get_actors()
                    actors_destroyed = 0
                    
                    # Build list of actors that still exist
                    actors_to_destroy = []
                    for actor in self.actor_list:
                        try:
                            if hasattr(actor, 'id') and current_world_actors.find(actor.id):
                                actors_to_destroy.append(actor)
                        except:
                            continue
                    
                    # Destroy actors that still exist
                    for actor in actors_to_destroy:
                        if self.safe_destroy_actor(actor):
                            actors_destroyed += 1
                    
                    logger.info(f"Successfully destroyed {actors_destroyed} actors")
                    
                except Exception as e:
                    logger.debug(f"Error during actor cleanup: {e}")
            
            # Clear actor list
            self.actor_list.clear()
            self.sensors.clear()
            self.ego_vehicle = None
            
            # Reset world settings
            if self.world:
                try:
                    settings = self.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    self.world.apply_settings(settings)
                except Exception as e:
                    logger.debug(f"Could not reset world settings: {e}")
            
            logger.info("CARLA cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class WeatherConfigurationMixin:
    """Mixin class for weather-specific configurations."""
    
    def get_weather_parameters(self, condition_name):
        """Get weather parameters for specific condition."""
        weather_configs = {
            'daytime': {
                'cloudiness': 20.0,
                'precipitation': 0.0,
                'sun_altitude_angle': 70.0,
                'fog_density': 0.0,
                'wetness': 0.0,
                'wind_intensity': 10.0
            },
            'nighttime': {
                'cloudiness': 30.0,
                'precipitation': 0.0,
                'sun_altitude_angle': -30.0,
                'fog_density': 0.0,
                'wetness': 0.0,
                'wind_intensity': 20.0
            },
            'rain': {
                'cloudiness': 80.0,
                'precipitation': 60.0,
                'sun_altitude_angle': 45.0,
                'fog_density': 10.0,
                'wetness': 80.0,
                'wind_intensity': 50.0
            },
            'fog': {
                'cloudiness': 90.0,
                'precipitation': 0.0,
                'sun_altitude_angle': 50.0,
                'fog_density': 70.0,
                'wetness': 20.0,
                'wind_intensity': 30.0
            }
        }
        
        return weather_configs.get(condition_name, weather_configs['daytime'])
    
    def apply_weather_condition(self, condition_name):
        """Apply specific weather condition to the world."""
        weather_params = self.get_weather_parameters(condition_name)
        weather = carla.WeatherParameters(**weather_params)
        self.world.set_weather(weather)
        
        logger.info(f"Applied weather condition: {condition_name}")
        logger.info(f"Parameters: {weather_params}")
        
        # Allow weather to stabilize
        for _ in range(20):
            self.world.tick()
            time.sleep(0.05)


# Combined base class
class EnhancedCARLAGenerator(DynamicCARLADatasetGenerator, WeatherConfigurationMixin):
    """Enhanced CARLA generator combining dynamic positioning and weather control."""
    pass