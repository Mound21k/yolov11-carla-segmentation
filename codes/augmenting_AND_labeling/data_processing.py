import os
os.environ['SAM_ENCODER_VERSION'] = 'vit_b'

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm
import json
from collections import defaultdict
import albumentations as A
import random
import re
from dataclasses import dataclass


import autodistill_grounded_sam.helpers as helpers
helpers.SAM_ENCODER_VERSION = 'vit_b'
helpers.SAM_CHECKPOINT_PATH = '/home/mound21k/.autodistill/sam_vit_b_01ec64.pth'
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

# # Debug prints
# print(f"SAM_ENCODER_VERSION: {os.environ.get('SAM_ENCODER_VERSION', 'not set')}")
# print(f"Library using: {helpers.SAM_ENCODER_VERSION}")


@dataclass
class ImageMetadata:
    """
    Simple container for image information needed for position-based splitting.
    
    This helps us keep track of which position each image came from so we can
    ensure that validation data comes from completely different positions than training data.
    """
    file_path: Path
    condition: str  # daytime, nighttime, rain, fog
    position_id: str  # extracted from filename (e.g., "pos01", "pos02")
    frame_number: int  # frame number within the position sequence

class SimpleDataAugmentation:
    """
    Clean data augmentation for instance segmentation that maintains perfect
    correspondence between image transformations and polygon coordinates.
    
    This implementation uses Albumentations with keypoint support to ensure
    that when we rotate or flip an image, the polygon coordinates are
    transformed in exactly the same way.
    """
    
    def __init__(self, image_size: int = 640):
        self.image_size = image_size
        
        # Geometric transforms that affect both image and coordinates
        self.geometric_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),  # Flip image left-right
            A.Rotate(limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT),  # Fixed: removed 'value' parameter
            A.Affine(
                scale=(0.9, 1.1),      # Scale image slightly larger/smaller
                translate_percent=(-0.05, 0.05),  # Move image slightly
                rotate=(-5, 5),        # Small rotation
                p=0.3
            )
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=True))
        
        # Visual transforms that only affect image appearance, not coordinates
        self.visual_transforms = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.4),
            A.GaussNoise(var_limit=(5, 25), mean=0, p=0.2),  # Fixed: changed var_limit to tuple of ints
            A.Blur(blur_limit=(3, 5), p=0.1),  # Fixed: proper range for blur_limit
            # Weather effects to make model more robust
            A.RandomRain(p=0.05),
            A.RandomFog(p=0.05),
        ])
    
    def augment_image_and_polygons(self, image: np.ndarray, 
                                 polygons: List[List[float]], 
                                 class_ids: List[int]) -> Tuple[np.ndarray, List[List[float]], List[int]]:
        """
        Apply augmentations to both image and polygon coordinates simultaneously.
        
        The key insight here is that we convert polygon coordinates to keypoints,
        apply the geometric transformation to both image and keypoints together,
        then convert back to polygons. This ensures perfect correspondence.
        """
        # Convert polygon coordinates to keypoints for transformation
        keypoints = []
        polygon_info = []  # Keep track of which keypoints belong to which polygon
        
        for poly_idx, polygon in enumerate(polygons):
            if len(polygon) < 6:  # Need at least 3 points (6 coordinates)
                continue
                
            start_idx = len(keypoints)
            # Convert normalized coordinates to pixel coordinates for transformation
            for i in range(0, len(polygon), 2):
                x = polygon[i] * self.image_size
                y = polygon[i+1] * self.image_size
                keypoints.append((x, y))
            
            end_idx = len(keypoints)
            polygon_info.append((poly_idx, start_idx, end_idx))
        
        # Apply geometric transformations to both image and keypoints
        try:
            transformed = self.geometric_transforms(image=image, keypoints=keypoints)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
        except:
            # If transformation fails, return original data
            transformed_image = image
            transformed_keypoints = keypoints
        
        # Convert transformed keypoints back to polygons
        new_polygons = []
        new_class_ids = []
        
        for original_poly_idx, start_idx, end_idx in polygon_info:
            # Extract keypoints for this polygon
            poly_keypoints = transformed_keypoints[start_idx:end_idx]
            
            # Convert back to normalized coordinates
            normalized_coords = []
            valid_polygon = True
            
            for x, y in poly_keypoints:
                x_norm = x / self.image_size
                y_norm = y / self.image_size
                
                # Check if point is still within image bounds
                if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                    normalized_coords.extend([x_norm, y_norm])
                else:
                    valid_polygon = False
                    break
            
            # Only keep polygons that remain valid after transformation
            if valid_polygon and len(normalized_coords) >= 6:
                new_polygons.append(normalized_coords)
                new_class_ids.append(class_ids[original_poly_idx])
        
        # Apply visual transformations (image only)
        try:
            final_image = self.visual_transforms(image=transformed_image)['image']
        except:
            final_image = transformed_image
        
        return final_image, new_polygons, new_class_ids

class CarlaToYOLOConverter:
    """
    Main converter class that handles the complete pipeline from CARLA images
    to YOLO-compatible dataset with position-based splitting and augmentation.
    
    This implementation uses the simple approach: resize images first, then
    run GroundedSAM on the resized images to get coordinates in the right space.
    """
    
    def __init__(self, 
                 dataset_root: str,
                 output_root: str,
                 ontology_mapping: Dict[str, str],
                 confidence_threshold: float = 0.3,
                 polygon_epsilon: float = 0.002,
                 target_image_size: int = 640,
                 augmentation_factor: int = 2):
        """
        Initialize the converter with all necessary parameters.
        
        The key insight is that we'll resize images to target_image_size BEFORE
        running GroundedSAM, so all coordinates come out in the right format already.
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.ontology_mapping = ontology_mapping
        self.confidence_threshold = confidence_threshold
        self.polygon_epsilon = polygon_epsilon
        self.target_image_size = target_image_size
        self.augmentation_factor = augmentation_factor
        
        # Create mapping from class names to numerical indices for YOLO
        unique_classes = list(set(ontology_mapping.values()))
        self.class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
        
        # Initialize GroundedSAM with your semantic ontology
        self.grounded_sam = GroundedSAM(ontology=CaptionOntology(ontology_mapping))
        
        # Initialize augmentation system
        self.augmenter = SimpleDataAugmentation(target_image_size)
        
        # Statistics tracking for understanding processing quality
        self.processing_stats = defaultdict(int)
        
        print(f"Converter initialized:")
        print(f"  Target size: {target_image_size}x{target_image_size}")
        print(f"  Augmentation: {augmentation_factor}x per image")
        print(f"  Classes: {list(self.class_to_index.keys())}")

    def extract_position_info(self, image_path: Path) -> ImageMetadata:
        """
        Extract position information from CARLA image filenames.
        
        This function parses your filename structure to identify which position
        each image came from. Adjust the regex patterns based on your actual
        filename format.
        """
        filename = image_path.stem
        condition = image_path.parts[-4]  # Extract from directory structure
        
        # Parse position and frame from filename
        # Example: "daytime_pos01_frame0009_20250607_005333_521.png"
        position_match = re.search(r'pos(\d+)', filename)
        frame_match = re.search(r'frame(\d+)', filename)
        
        position_id = position_match.group(1) if position_match else "unknown"
        frame_number = int(frame_match.group(1)) if frame_match else 0
        
        return ImageMetadata(
            file_path=image_path,
            condition=condition,
            position_id=position_id,
            frame_number=frame_number
        )

    def create_position_based_splits(self, image_metadata: List[ImageMetadata], 
                                   train_ratio: float = 0.7, 
                                   val_ratio: float = 0.2) -> Dict[str, List[ImageMetadata]]:
        """
        Split dataset by positions rather than individual images.
        
        This is the key innovation for spatial generalization: instead of randomly
        splitting images, we split positions. This means validation data comes from
        completely different spatial locations than training data.
        """
        # Group images by weather condition and position
        condition_position_groups = defaultdict(lambda: defaultdict(list))
        
        for metadata in image_metadata:
            condition_position_groups[metadata.condition][metadata.position_id].append(metadata)
        
        splits = {'train': [], 'val': [], 'test': []}
        
        print("Creating position-based splits:")
        
        # For each weather condition, split positions (not individual images)
        for condition, position_groups in condition_position_groups.items():
            positions = list(position_groups.keys())
            n_positions = len(positions)
            
            print(f"  {condition}: {n_positions} positions")
            
            # Randomly shuffle positions for fair assignment
            random.shuffle(positions)
            
            # Calculate split boundaries for positions
            train_end = int(n_positions * train_ratio)
            val_end = int(n_positions * (train_ratio + val_ratio))
            
            # Assign entire positions to splits
            train_positions = positions[:train_end]
            val_positions = positions[train_end:val_end]
            test_positions = positions[val_end:]
            
            # Add all images from assigned positions to respective splits
            for pos in train_positions:
                splits['train'].extend(position_groups[pos])
            for pos in val_positions:
                splits['val'].extend(position_groups[pos])
            for pos in test_positions:
                splits['test'].extend(position_groups[pos])
        
        return splits

    def simple_resize_image(self, image_path: Path) -> np.ndarray:
        """
        Simple image resizing from 1920x1080 to target size.
        
        This is much simpler than the complex cropping approach. We just
        resize the image and accept that there will be some aspect ratio
        distortion, which modern models handle well.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for consistency
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Simple resize to target dimensions
        resized = cv2.resize(image, (self.target_image_size, self.target_image_size))
        
        return resized

    def mask_to_polygon(self, mask: np.ndarray) -> List[float]:
        """
        Convert a boolean mask to normalized polygon coordinates.
        
        This function extracts the boundary of objects in the mask and
        converts them to a list of normalized coordinate pairs suitable for YOLO.
        """
        # Convert boolean mask to uint8 for OpenCV
        mask_uint8 = (mask.astype(np.uint8) * 255)
        
        # Find contours (object boundaries)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Use the largest contour (main object, ignoring small noise)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Simplify polygon to reduce complexity while preserving shape
        epsilon = self.polygon_epsilon * cv2.arcLength(largest_contour, True)
        simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Extract points and flatten to coordinate list
        points = simplified_contour.reshape(-1, 2)
        
        if len(points) < 3:  # Need at least 3 points for a valid polygon
            return []
        
        # Normalize coordinates to [0, 1] range
        normalized_coords = []
        for x, y in points:
            norm_x = max(0.0, min(1.0, float(x) / self.target_image_size))
            norm_y = max(0.0, min(1.0, float(y) / self.target_image_size))
            normalized_coords.extend([norm_x, norm_y])
        
        return normalized_coords

    def validate_polygon(self, polygon: List[float]) -> bool:
        """
        Check if a polygon meets YOLO format requirements.
        
        This ensures we only include valid polygons in our training data,
        which prevents training issues later.
        """
        # Check minimum requirements
        if len(polygon) < 6:  # At least 3 points
            return False
        if len(polygon) % 2 != 0:  # Even number (x,y pairs)
            return False
        
        # Check coordinate bounds
        for coord in polygon:
            if not (0.0 <= coord <= 1.0):
                return False
        
        # Check polygon area (shoelace formula)
        points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
        area = 0.0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2.0
        
        return area > 1e-6  # Must have non-zero area

    def process_single_image(self, metadata: ImageMetadata) -> List[Tuple[np.ndarray, List[str]]]:
        """
        Process one image through the complete pipeline with augmentation.
        
        This is where the magic happens: we resize the image, run GroundedSAM,
        extract polygons, and generate augmented versions.
        """
        results = []
        
        try:
            # Step 1: Resize the image to target size
            resized_image = self.simple_resize_image(metadata.file_path)
            
            # Step 2: Run GroundedSAM on the resized image
            # This gives us detections in the correct coordinate space already!
            detections = self.grounded_sam.predict(resized_image)
            
            # Step 3: Extract polygons and class information
            polygons = []
            class_ids = []
            
            if len(detections.mask) > 0:
                for i in range(len(detections.mask)):
                    # Filter by confidence
                    if detections.confidence[i] < self.confidence_threshold:
                        continue
                    
                    # Convert mask to polygon
                    polygon = self.mask_to_polygon(detections.mask[i])
                    
                    if not polygon or not self.validate_polygon(polygon):
                        continue
                    
                    # Map class ID to our index system
                    grounded_class_id = detections.class_id[i]
                    if grounded_class_id < len(list(self.ontology_mapping.values())):
                        class_name = list(self.ontology_mapping.values())[grounded_class_id]
                        if class_name in self.class_to_index:
                            polygons.append(polygon)
                            class_ids.append(self.class_to_index[class_name])
            
            # Step 4: Create original version
            if polygons:
                annotations = []
                for poly, class_id in zip(polygons, class_ids):
                    coord_str = ' '.join(f'{coord:.6f}' for coord in poly)
                    annotations.append(f"{class_id} {coord_str}")
                
                results.append((resized_image, annotations))
                self.processing_stats['successful_conversions'] += 1
            
            # Step 5: Generate augmented versions
            for aug_idx in range(self.augmentation_factor):
                try:
                    aug_image, aug_polygons, aug_class_ids = self.augmenter.augment_image_and_polygons(
                        resized_image.copy(), polygons.copy(), class_ids.copy()
                    )
                    
                    if aug_polygons:
                        aug_annotations = []
                        for poly, class_id in zip(aug_polygons, aug_class_ids):
                            coord_str = ' '.join(f'{coord:.6f}' for coord in poly)
                            aug_annotations.append(f"{class_id} {coord_str}")
                        
                        results.append((aug_image, aug_annotations))
                        self.processing_stats['successful_augmentations'] += 1
                
                except Exception as e:
                    self.processing_stats['failed_augmentations'] += 1
            
            self.processing_stats['processed_images'] += 1
            
        except Exception as e:
            print(f"Error processing {metadata.file_path}: {e}")
            self.processing_stats['processing_errors'] += 1
        
        return results

    def create_yolo_structure(self) -> Dict[str, Path]:
        """Create the directory structure that YOLO expects."""
        structure = {}
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'val', 'test']:
            for data_type in ['images', 'labels']:
                dir_path = self.output_root / split / data_type
                dir_path.mkdir(parents=True, exist_ok=True)
                structure[f'{split}_{data_type}'] = dir_path
        
        return structure

    def create_dataset_yaml(self) -> None:
        """Create the YAML configuration file that YOLO needs for training."""
        config = {
            'path': str(self.output_root.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.index_to_class
        }
        
        yaml_path = self.output_root / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created YOLO config: {yaml_path}")

    def process_complete_dataset(self, train_ratio: float = 0.6, val_ratio: float = 0.25):
        """
        Main function that processes the entire CARLA dataset.
        
        This orchestrates the complete pipeline: discovery, position-based splitting,
        processing with augmentation, and YOLO dataset creation.
        """
        print("🚀 Starting CARLA to YOLO conversion with simple resizing approach...")
        
        # Create output structure
        dir_structure = self.create_yolo_structure()
        
        # Discover all images
        all_metadata = []
        conditions = ['daytime', 'nighttime', 'rain', 'fog']
        
        for condition in conditions:
            rgb_dir = self.dataset_root / condition / 'images' / 'rgb'
            if rgb_dir.exists():
                images = list(rgb_dir.glob('*.png')) + list(rgb_dir.glob('*.jpg'))
                for img_path in images:
                    metadata = self.extract_position_info(img_path)
                    all_metadata.append(metadata)
                print(f"Found {len(images)} images in {condition} condition")
        
        print(f"Total images: {len(all_metadata)}")
        
        # Create position-based splits
        splits = self.create_position_based_splits(all_metadata, train_ratio, val_ratio)
        
        # Process each split
        for split_name, metadata_list in splits.items():
            print(f"\n📊 Processing {split_name} split ({len(metadata_list)} images)...")
            
            images_dir = dir_structure[f'{split_name}_images']
            labels_dir = dir_structure[f'{split_name}_labels']
            
            sample_count = 0
            
            for metadata in tqdm(metadata_list, desc=f"Converting {split_name}"):
                generated_samples = self.process_single_image(metadata)
                
                # Save all generated samples
                for sample_idx, (image, annotations) in enumerate(generated_samples):
                    base_name = metadata.file_path.stem
                    if sample_idx == 0:
                        sample_name = f"{base_name}_orig"
                    else:
                        sample_name = f"{base_name}_aug{sample_idx}"
                    
                    # Save image
                    image_path = images_dir / f"{sample_name}.jpg"
                    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(image_path), image_bgr)
                    
                    # Save annotations
                    label_path = labels_dir / f"{sample_name}.txt"
                    with open(label_path, 'w') as f:
                        for annotation in annotations:
                            f.write(annotation + '\n')
                    
                    sample_count += 1
            
            print(f"Generated {sample_count} samples for {split_name}")
        
        # Create YOLO configuration
        self.create_dataset_yaml()
        
        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print processing statistics and next steps."""
        print("\n" + "="*60)
        print("🎯 CONVERSION COMPLETE")
        print("="*60)
        
        total_processed = self.processing_stats['processed_images']
        successful = self.processing_stats['successful_conversions']
        augmentations = self.processing_stats['successful_augmentations']
        
        print(f"Images processed: {total_processed}")
        print(f"Base conversions: {successful}")
        print(f"Augmentations: {augmentations}")
        print(f"Total samples: {successful + augmentations}")
        print(f"Multiplier achieved: {(successful + augmentations) / max(successful, 1):.1f}x")
        
        print(f"\n✅ Dataset ready at: {self.output_root}")
        print(f"Start training: yolo segment train data={self.output_root}/dataset.yaml model=yolo11n-seg.pt epochs=100")

# Main execution with debugging
if __name__ == "__main__":
    # Set random seeds for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # 🔍 STEP 1: Find your dataset location
    print("🔍 DEBUGGING: Looking for your dataset...")
    
    # Common locations in Kaggle
    possible_paths = [
        "/kaggle/input",  # Most common location for uploaded datasets
        "/kaggle/working",
        "/content",  # If you're in Colab
        "/home/mound21k/Desktop/forAida/carla_proj/New_proj/finaL_dataset",  # Original path
        ".",  # Current directory
    ]
    
    dataset_found = False
    actual_dataset_path = None
    
    for path in possible_paths:
        print(f"  Checking: {path}")
        if os.path.exists(path):
            # List what's in this directory
            contents = os.listdir(path)
            print(f"    Contents: {contents}")
            
            # Look for CARLA dataset structure
            for item in contents:
                item_path = Path(path) / item
                if item_path.is_dir():
                    # Check if this looks like our dataset
                    subdirs = [f.name for f in item_path.iterdir() if f.is_dir()]
                    if any(weather in subdirs for weather in ['daytime', 'nighttime', 'rain', 'fog']):
                        print(f"    🎯 Found dataset at: {item_path}")
                        actual_dataset_path = str(item_path)
                        dataset_found = True
                        break
            
            if dataset_found:
                break
    
    if not dataset_found:
        print("❌ Could not find your CARLA dataset!")
        print("Please manually set the correct path below:")
        print("Look for a folder that contains: daytime/, nighttime/, rain/, fog/")
        
        # You can manually set your dataset path here:
        actual_dataset_path = "/home/mound21k/Desktop/forAida/carla_proj/New_proj/finaL_dataset"  # 👈 UPDATE THIS PATH
        
        print(f"Using manual path: {actual_dataset_path}")
    
    # 🔍 STEP 2: Verify the dataset structure
    print(f"\n🔍 VERIFYING: Dataset structure at {actual_dataset_path}")
    
    if os.path.exists(actual_dataset_path):
        dataset_root = Path(actual_dataset_path)
        conditions = ['daytime', 'nighttime', 'rain', 'fog']
        
        for condition in conditions:
            condition_path = dataset_root / condition
            if condition_path.exists():
                rgb_path = condition_path / 'images' / 'rgb'
                if rgb_path.exists():
                    images = list(rgb_path.glob('*.png')) + list(rgb_path.glob('*.jpg'))
                    print(f"  ✅ {condition}: {len(images)} images found in {rgb_path}")
                else:
                    print(f"  ❌ {condition}: RGB folder not found at {rgb_path}")
            else:
                print(f"  ❌ {condition}: Condition folder not found")
    else:
        print(f"❌ Dataset path does not exist: {actual_dataset_path}")
        print("\n🛠️  MANUAL FIX NEEDED:")
        print("1. Look at your Kaggle dataset browser on the right side")
        print("2. Find your CARLA dataset folder")
        print("3. Copy the full path and update 'actual_dataset_path' above")
        exit()
    
    # Define your class mapping
    ontology = {
        "car": "vehicle",
        "bus": "vehicle", 
        "truck": "vehicle",
        "traffic light": "traffic_light",
        "person": "pedestrian",
        "bicycle": "bicycle"
    }
    
    # Initialize converter with the found dataset path
    converter = CarlaToYOLOConverter(
        dataset_root=actual_dataset_path,  # Using the found/manual path
        output_root="/home/mound21k/Desktop/forAida/carla_proj/New_proj/instance_seg_proj",  # Output in working directory
        ontology_mapping=ontology,
        confidence_threshold=0.25,
        polygon_epsilon=0.001,
        target_image_size=640,
        augmentation_factor=1  # 3x total data (1 original + 2 augmented)
    )
    
    # Run the conversion
    converter.process_complete_dataset(
        train_ratio=0.6,   # 60% of positions for training
        val_ratio=0.25     # 25% for validation, 15% for testing
    )