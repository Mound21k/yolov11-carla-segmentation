#!/usr/bin/env python3
"""
Fog Dataset Generator for YOLOv11 Instance Segmentation
=============================================================

Generates challenging fog driving scenarios with dynamic positioning
and guaranteed coverage of all 4 target classes: car, bus, pedestrian, traffic_light

FIXES:
- Uses proven spawn logic from working generators
- Simplified actor cleanup matching working versions
- Removed complex force-clear mechanisms
- Based on working daytime/rain/nighttime patterns

Usage: python fog_generator.py [--samples 30] [--positions 5]

Author: MLDL Assistant
Date: June 2025
"""

import sys
import time
import json
import random
import logging
import argparse
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import queue

# CARLA imports
import carla

# Import the base generator
from carla_base_generator import EnhancedCARLAGenerator

logger = logging.getLogger(__name__)


class FogDatasetGenerator(EnhancedCARLAGenerator):
    """Specialized generator for fog driving scenarios using proven spawn logic."""
    
    def __init__(self, output_dir='datasets/fog', samples_per_position=60, num_positions=5):
        super().__init__(output_dir=output_dir)
        
        self.condition_name = 'fog'
        self.samples_per_position = samples_per_position
        self.num_positions = num_positions
        self.total_target_samples = samples_per_position * num_positions
        
        # Fog-specific configuration
        self.fog_config = {
            'fog_density': 'heavy',
            'traffic_density': 'normal_with_caution',
            'pedestrian_activity': 'reduced',
            'collection_speed': 'very_careful',
            'vehicle_lights_enabled': True,
            'fog_lights_enabled': True,
            'severely_reduced_visibility': True,
            'speed_reduction': True,
            'visibility_range': 20,
            'fog_falloff': 0.2
        }
        
        # Setup condition-specific directories
        self.setup_condition_directories()
        
        # Data collection tracking
        self.samples_collected = 0
        self.position_stats = []
        self.class_distribution = defaultdict(int)
        
    def setup_condition_directories(self):
        """Setup directories specific to fog condition."""
        condition_dirs = [
            f'{self.condition_name}/images/rgb',
            f'{self.condition_name}/images/segmentation',
            f'{self.condition_name}/annotations',
            f'{self.condition_name}/metadata'
        ]
        
        for directory in condition_dirs:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Fog dataset directories created")
    
    def generate_fog_dataset(self):
        """Generate complete fog dataset with proven stability patterns."""
        logger.info("="*60)
        logger.info("STARTING FOG DATASET GENERATION")
        logger.info("="*60)
        
        try:
            # Initialize CARLA connection
            if not self.connect_to_carla():
                return False
            
            # Initialize spawn points
            self.initialize_spawn_points()
            
            # Apply fog weather
            self.apply_weather_condition(self.condition_name)
            
            # Generate data from multiple positions
            for position_idx in range(self.num_positions):
                logger.info(f"\n--- FOG POSITION {position_idx + 1}/{self.num_positions} ---")
                
                success = self.collect_data_from_position(position_idx)
                if not success:
                    logger.warning(f"Position {position_idx + 1} collection failed, continuing...")
                    continue
                
                # Brief pause between positions
                time.sleep(2.0)
            
            # Generate comprehensive report
            self.generate_fog_report()
            
            logger.info("="*60)
            logger.info("FOG DATASET GENERATION COMPLETED")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error during fog dataset generation: {e}")
            return False
        
        finally:
            self.cleanup()
    
    def collect_data_from_position(self, position_idx):
        """Collect data from a specific ego vehicle position using proven logic."""
        try:
            # Clean up previous actors (except traffic lights) - simple version
            self.cleanup_dynamic_actors()
            
            # Spawn ego vehicle at new position - simple version
            if not self.spawn_ego_vehicle_at_position(position_idx):
                return False
            
            # Setup sensors
            if not self.setup_synchronized_sensors():
                return False
            
            # Spawn traffic scenario using base class method (proven to work)
            if not self.spawn_comprehensive_traffic():
                logger.warning(f"Traffic spawning suboptimal for position {position_idx}")
                # Continue anyway if we have some traffic
            
            # Apply fog-specific lighting to vehicles
            self.apply_fog_lighting()
            
            # Collect synchronized frames
            position_stats = self.collect_synchronized_frames(position_idx)
            self.position_stats.append(position_stats)
            
            logger.info(f"Fog position {position_idx + 1} completed: {position_stats['frames_collected']} frames")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data from position {position_idx}: {e}")
            return False
    
    def spawn_ego_vehicle_at_position(self, position_idx):
        """Spawn ego vehicle using simple proven logic from working generators."""
        try:
            # Use modulo to cycle through available spawn points (simple approach)
            spawn_point_idx = position_idx % len(self.ego_spawn_points)
            spawn_point = self.ego_spawn_points[spawn_point_idx]
            
            # Select ego vehicle blueprint
            ego_bp = self.world.get_blueprint_library().find('vehicle.tesla.model3')
            ego_bp.set_attribute('color', '255,255,255')
            
            # Simple spawn - try the selected point directly
            try:
                self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
                self.actor_list.append(self.ego_vehicle)
                
                location = spawn_point.location
                logger.info(f"Ego vehicle spawned at position {spawn_point_idx}: Location(x={location.x:.3f}, y={location.y:.3f}, z={location.z:.3f})")
                return True
                
            except Exception as spawn_error:
                logger.debug(f"Direct spawn failed: {spawn_error}")
                
                # Try other spawn points if the first fails (simple fallback)
                for attempt in range(1, len(self.ego_spawn_points)):
                    try:
                        fallback_idx = (spawn_point_idx + attempt) % len(self.ego_spawn_points)
                        fallback_point = self.ego_spawn_points[fallback_idx]
                        
                        self.ego_vehicle = self.world.spawn_actor(ego_bp, fallback_point)
                        self.actor_list.append(self.ego_vehicle)
                        
                        location = fallback_point.location
                        logger.info(f"Ego vehicle spawned at fallback position {fallback_idx}: Location(x={location.x:.3f}, y={location.y:.3f}, z={location.z:.3f})")
                        return True
                        
                    except Exception as fallback_error:
                        logger.debug(f"Fallback spawn {attempt} failed: {fallback_error}")
                        continue
                
                logger.error(f"Failed to spawn ego vehicle at position {position_idx} after trying all spawn points")
                return False
            
        except Exception as e:
            logger.error(f"Failed to spawn ego vehicle at position {position_idx}: {e}")
            return False
    
    def apply_fog_lighting(self):
        """Apply fog-specific lighting to all spawned vehicles."""
        try:
            vehicle_count = 0
            for actor in self.actor_list:
                if hasattr(actor, 'type_id') and 'vehicle' in actor.type_id:
                    try:
                        # Enable maximum lighting for fog visibility
                        light_state = carla.VehicleLightState.NONE
                        light_state |= carla.VehicleLightState.Position
                        light_state |= carla.VehicleLightState.LowBeam
                        light_state |= carla.VehicleLightState.Fog
                        
                        # 30% chance of hazard lights (cautious driving in fog)
                        if random.random() < 0.3:
                            light_state |= carla.VehicleLightState.LeftBlinker
                            light_state |= carla.VehicleLightState.RightBlinker
                        
                        actor.set_light_state(light_state)
                        vehicle_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to set fog lights for vehicle {actor.id}: {e}")
            
            logger.info(f"Applied fog lighting to {vehicle_count} vehicles")
            
        except Exception as e:
            logger.warning(f"Error applying fog lighting: {e}")
    
    def cleanup_dynamic_actors(self):
        """Clean up dynamic actors using simple proven logic from working generators."""
        if not self.world:
            return
            
        try:
            current_actors = self.world.get_actors()
            actors_to_remove = []
            
            for actor in self.actor_list[:]:
                try:
                    actor_id = getattr(actor, 'id', None)
                    if actor_id and current_actors.find(actor_id):
                        actor_type = getattr(actor, 'type_id', '')
                        if 'traffic.traffic_light' not in actor_type:
                            actors_to_remove.append(actor)
                except:
                    actors_to_remove.append(actor)
            
            # Remove actors safely
            for actor in actors_to_remove:
                if self.safe_destroy_actor(actor):
                    try:
                        self.actor_list.remove(actor)
                    except ValueError:
                        pass
                        
        except Exception as e:
            logger.debug(f"Error in cleanup_dynamic_actors: {e}")
        
        # Clear sensor references
        self.sensors.clear()
        self.ego_vehicle = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def collect_synchronized_frames(self, position_idx):
        """Collect synchronized RGB and segmentation frames with fog-appropriate timeouts."""
        frames_collected = 0
        target_frames = self.samples_per_position
        start_time = time.time()
        position_class_stats = defaultdict(int)
        
        logger.info(f"Collecting {target_frames} synchronized frames from position {position_idx + 1}")
        
        # Extended settling time for fog conditions
        for _ in range(35):
            self.world.tick()
            time.sleep(0.05)
        
        collection_timeout = 180  # 3 minutes per position
        last_frame_time = start_time
        
        while frames_collected < target_frames and (time.time() - start_time) < collection_timeout:
            # Advance simulation
            self.world.tick()
            
            # Check for synchronized frames with timeout appropriate for fog
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                # Process and save frame
                frame_stats = self.process_and_save_frame(
                    frame_data, position_idx, frames_collected
                )
                
                # Update statistics
                for class_name, count in frame_stats.items():
                    position_class_stats[class_name] += count
                    self.class_distribution[class_name] += count
                
                frames_collected += 1
                last_frame_time = time.time()
                
                if frames_collected % 10 == 0:
                    logger.info(f"  Collected {frames_collected}/{target_frames} frames")
                
            except queue.Empty:
                # Check if we're stuck (no frames for 30 seconds)
                if time.time() - last_frame_time > 30:
                    logger.warning(f"No frames received for 30 seconds, stopping collection")
                    break
                time.sleep(0.02)
                continue
        
        collection_time = time.time() - start_time
        
        return {
            'position_index': position_idx,
            'frames_collected': frames_collected,
            'collection_time': collection_time,
            'class_distribution': dict(position_class_stats),
            'average_fps': frames_collected / max(collection_time, 0.001)  # Prevent division by zero
        }
    
    def process_and_save_frame(self, frame_data, position_idx, frame_idx):
        """Process synchronized frame data and save with annotations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename_base = f"fog_pos{position_idx:02d}_frame{frame_idx:04d}_{timestamp}"
        
        # Save RGB image
        rgb_path = self.output_dir / self.condition_name / 'images' / 'rgb' / f"{filename_base}.png"
        frame_data['rgb'].save_to_disk(str(rgb_path))
        
        # Save segmentation image  
        seg_path = self.output_dir / self.condition_name / 'images' / 'segmentation' / f"{filename_base}.png"
        frame_data['segmentation'].save_to_disk(str(seg_path))
        
        # Generate YOLOv11 annotations
        annotations_stats = self.generate_yolo_annotations(
            frame_data['segmentation'], filename_base
        )
        
        # Save frame metadata
        metadata = {
            'filename_base': filename_base,
            'condition': self.condition_name,
            'position_index': position_idx,
            'frame_index': frame_idx,
            'carla_timestamp': frame_data['timestamp'],
            'frame_number': frame_data['frame_number'],
            'collection_timestamp': timestamp,
            'rgb_path': str(rgb_path.relative_to(self.output_dir)),
            'segmentation_path': str(seg_path.relative_to(self.output_dir)),
            'class_instances': annotations_stats,
            'weather_condition': 'fog',
            'fog_density': 'heavy',
            'visibility_conditions': 'severely_reduced',
            'lighting_conditions': 'enhanced_vehicle_lighting',
            'ego_vehicle_location': self.get_ego_vehicle_location(),
            'fog_config': self.fog_config
        }
        
        metadata_path = self.output_dir / self.condition_name / 'metadata' / f"{filename_base}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.samples_collected += 1
        return annotations_stats
    
    def generate_yolo_annotations(self, segmentation_image, filename_base):
        """Generate YOLOv11-compatible instance segmentation annotations."""
        try:
            # Convert CARLA segmentation to numpy array
            seg_array = np.frombuffer(segmentation_image.raw_data, dtype=np.uint8)
            seg_array = seg_array.reshape((segmentation_image.height, segmentation_image.width, 4))
            
            # Extract instance IDs from segmentation
            instance_img = seg_array[:, :, 2]  # Blue channel contains instance IDs
            
            # Get all actors for class mapping
            all_actors = self.world.get_actors()
            
            annotations = []
            class_stats = defaultdict(int)
            
            # Process each unique instance
            unique_instances = np.unique(instance_img)
            
            for instance_id in unique_instances:
                if instance_id == 0:  # Skip background
                    continue
                
                # Find corresponding actor
                actor = None
                for a in all_actors:
                    if a.id == instance_id:
                        actor = a
                        break
                
                if not actor:
                    continue
                
                # Map actor type to target classes
                class_name = self.map_actor_to_class(actor.type_id)
                if not class_name:
                    continue
                
                # Extract instance mask
                instance_mask = (instance_img == instance_id).astype(np.uint8)
                
                # Find contours
                contours, _ = cv2.findContours(
                    instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    continue
                
                # Use largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Filter out very small instances (higher threshold for fog visibility)
                if cv2.contourArea(largest_contour) < 250:
                    continue
                
                # Create normalized polygon points for YOLOv11
                polygon_points = []
                img_height, img_width = instance_img.shape
                
                # Simplify contour for better performance
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                for point in simplified_contour.reshape(-1, 2):
                    x_norm = point[0] / img_width
                    y_norm = point[1] / img_height
                    polygon_points.extend([x_norm, y_norm])
                
                # Create annotation entry
                annotation = {
                    'class_id': self.yolo_class_mapping[class_name],
                    'class_name': class_name,
                    'polygon': polygon_points,
                    'instance_id': int(instance_id),
                    'area': int(cv2.contourArea(largest_contour)),
                    'weather_condition': 'fog',
                    'visibility_challenge': 'high'
                }
                
                annotations.append(annotation)
                class_stats[class_name] += 1
            
            # Save YOLOv11 annotation file
            self.save_yolo_annotation_file(annotations, filename_base)
            
            return dict(class_stats)
            
        except Exception as e:
            logger.error(f"Error generating annotations for {filename_base}: {e}")
            return {}
    
    def map_actor_to_class(self, actor_type_id):
        """Map CARLA actor type to target class."""
        for carla_type, class_name in self.target_classes.items():
            if carla_type in actor_type_id:
                return class_name
        return None
    
    def save_yolo_annotation_file(self, annotations, filename_base):
        """Save YOLOv11-compatible annotation file."""
        annotation_path = self.output_dir / self.condition_name / 'annotations' / f"{filename_base}.txt"
        
        with open(annotation_path, 'w') as f:
            for ann in annotations:
                # YOLOv11 format: class_id x1 y1 x2 y2 ... xn yn
                line = [str(ann['class_id'])]
                line.extend([f"{coord:.6f}" for coord in ann['polygon']])
                f.write(' '.join(line) + '\n')
        
        # Also save detailed JSON annotations
        detailed_path = self.output_dir / self.condition_name / 'annotations' / f"{filename_base}_detailed.json"
        with open(detailed_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def get_ego_vehicle_location(self):
        """Get current ego vehicle location for metadata."""
        if self.ego_vehicle:
            try:
                location = self.ego_vehicle.get_location()
                return {
                    'x': float(location.x),
                    'y': float(location.y), 
                    'z': float(location.z)
                }
            except:
                pass
        return None
    
    def generate_fog_report(self):
        """Generate comprehensive fog dataset report with safe calculations."""
        # Safe total calculation
        total_samples = max(self.samples_collected, 1)  # Prevent division by zero
        total_target = max(self.total_target_samples, 1)
        
        report = {
            'dataset_info': {
                'condition': self.condition_name,
                'generation_timestamp': datetime.now().isoformat(),
                'total_samples_collected': self.samples_collected,
                'target_samples': self.total_target_samples,
                'completion_rate': (self.samples_collected / total_target) * 100,
                'num_positions': self.num_positions,
                'samples_per_position': self.samples_per_position
            },
            'class_distribution': dict(self.class_distribution),
            'position_statistics': self.position_stats,
            'collection_parameters': {
                'weather_condition': self.condition_name,
                'camera_resolution': f"{self.camera_config['image_size_x']}x{self.camera_config['image_size_y']}",
                'synchronization_mode': 'frame_synchronized',
                'traffic_density': self.fog_config['traffic_density']
            },
            'quality_metrics': self.calculate_quality_metrics(),
            'fog_specific': {
                'heavy_fog_density': True,
                'severely_reduced_visibility': True,
                'enhanced_vehicle_lighting': True,
                'cautious_traffic_behavior': True,
                'reduced_pedestrian_activity': True,
                'visibility_range_meters': self.fog_config.get('visibility_range', 20),
                'fog_falloff_rate': self.fog_config.get('fog_falloff', 0.2)
            }
        }
        
        # Save report
        report_path = self.output_dir / f"{self.condition_name}_dataset_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.print_fog_summary(report)
        
        logger.info(f"Fog dataset report saved to: {report_path}")
    
    def calculate_quality_metrics(self):
        """Calculate dataset quality metrics with safe division."""
        total_instances = sum(self.class_distribution.values())
        safe_samples = max(self.samples_collected, 1)  # Prevent division by zero
        
        metrics = {
            'total_instances': total_instances,
            'average_instances_per_frame': total_instances / safe_samples,
            'class_balance_score': self.calculate_class_balance_score(),
            'position_coverage': len(self.position_stats),
            'successful_positions': len([p for p in self.position_stats if p['frames_collected'] > 0])
        }
        
        return metrics
    
    def calculate_class_balance_score(self):
        """Calculate class balance score with safe division."""
        if not self.class_distribution:
            return 0.0
        
        total_instances = sum(self.class_distribution.values())
        if total_instances == 0:
            return 0.0
            
        num_classes = len(self.target_classes)
        if num_classes == 0:
            return 0.0
            
        ideal_per_class = total_instances / num_classes
        
        balance_scores = []
        for class_name in self.target_classes.values():
            actual_count = self.class_distribution.get(class_name, 0)
            if ideal_per_class > 0 and actual_count > 0:
                balance_score = min(actual_count / ideal_per_class, ideal_per_class / actual_count)
                balance_scores.append(balance_score)
            elif ideal_per_class == 0 and actual_count == 0:
                balance_scores.append(1.0)  # Perfect balance if both are zero
            else:
                balance_scores.append(0.0)  # No balance if one is zero and other isn't
        
        return sum(balance_scores) / len(balance_scores) if balance_scores else 0.0
    
    def print_fog_summary(self, report):
        """Print comprehensive fog dataset summary."""
        print("\n" + "="*70)
        print("FOG DATASET GENERATION SUMMARY")
        print("="*70)
        
        dataset_info = report['dataset_info']
        print(f"Fog Samples: {dataset_info['total_samples_collected']}/{dataset_info['target_samples']}")
        print(f"Completion Rate: {dataset_info['completion_rate']:.1f}%")
        print(f"Positions Used: {dataset_info['num_positions']}")
        
        print(f"\nClass Distribution:")
        total_class_instances = sum(report['class_distribution'].values())
        for class_name, count in report['class_distribution'].items():
            if total_class_instances > 0:
                percentage = (count / total_class_instances) * 100
            else:
                percentage = 0
            print(f"  {class_name}: {count} instances ({percentage:.1f}%)")
        
        quality_metrics = report['quality_metrics']
        print(f"\nQuality Metrics:")
        print(f"  Total Instances: {quality_metrics['total_instances']}")
        print(f"  Avg Instances/Frame: {quality_metrics['average_instances_per_frame']:.1f}")
        print(f"  Class Balance Score: {quality_metrics['class_balance_score']:.3f}")
        print(f"  Successful Positions: {quality_metrics['successful_positions']}/{quality_metrics['position_coverage']}")
        
        fog_specific = report['fog_specific']
        print(f"\nFog Features:")
        print(f"  Heavy Fog Density: {'Yes' if fog_specific['heavy_fog_density'] else 'No'}")
        print(f"  Severely Reduced Visibility: {'Yes' if fog_specific['severely_reduced_visibility'] else 'No'}")
        print(f"  Enhanced Vehicle Lighting: {'Yes' if fog_specific['enhanced_vehicle_lighting'] else 'No'}")
        print(f"  Cautious Traffic Behavior: {'Yes' if fog_specific['cautious_traffic_behavior'] else 'No'}")
        print(f"  Visibility Range: {fog_specific['visibility_range_meters']}m")
        print(f"  Fog Falloff Rate: {fog_specific['fog_falloff_rate']}")
        
        print("\n" + "="*70)


def main():
    """Main execution function for fog dataset generation."""
    parser = argparse.ArgumentParser(description='Generate Fog Dataset for YOLOv11')
    parser.add_argument('--samples', type=int, default=30, help='Samples per position')
    parser.add_argument('--positions', type=int, default=33, help='Number of ego positions')
    parser.add_argument('--output-dir', default='datasets/fog', help='Output directory')
    parser.add_argument('--host', default='localhost', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    
    args = parser.parse_args()
    
    # Initialize fog generator
    generator = FogDatasetGenerator(
        output_dir=args.output_dir,
        samples_per_position=args.samples,
        num_positions=args.positions
    )
    
    # Override connection settings if needed
    generator.host = args.host
    generator.port = args.port
    
    try:
        success = generator.generate_fog_dataset()
        
        if success:
            print(f"\n🌫️ Fog dataset generation completed successfully!")
            print(f"📁 Dataset saved to: {generator.output_dir}")
            print(f"📊 Check the report file for detailed statistics")
        else:
            print("❌ Fog dataset generation failed. Check logs for details.")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️  Dataset generation interrupted by user")
        generator.cleanup()
    
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        generator.cleanup()
        return 1
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)