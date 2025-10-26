"""
VapeWatch Data Preprocessing Module
Compatible with SageMaker for vape/cigarette detection
"""

import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VapeDataPreprocessor:
    """
    Data preprocessor for VapeWatch dataset compatible with SageMaker
    """
    
    def __init__(self, input_data_path: str, output_data_path: str, config: Dict[str, Any] = None):
        """
        Initialize the data preprocessor
        
        Args:
            input_data_path: Path to the input dataset
            output_data_path: Path to save processed data
            config: Configuration dictionary
        """
        self.input_data_path = Path(input_data_path)
        self.output_data_path = Path(output_data_path)
        self.config = config or self._get_default_config()
        
        # Create output directories
        self.output_data_path.mkdir(parents=True, exist_ok=True)
        (self.output_data_path / "train").mkdir(exist_ok=True)
        (self.output_data_path / "validation").mkdir(exist_ok=True)
        (self.output_data_path / "test").mkdir(exist_ok=True)
        
        # Class mapping
        self.class_mapping = {
            "cigarette": 0,
            "vape": 1
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "image_size": (640, 640),
            "augmentation": {
                "horizontal_flip": 0.5,
                "rotation": 15,
                "brightness_contrast": 0.2,
                "hue_saturation": 0.1
            },
            "validation_split": 0.2,
            "test_split": 0.1
        }
    
    def load_coco_annotations(self, json_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load COCO format annotations
        
        Args:
            json_path: Path to COCO JSON file
            
        Returns:
            Tuple of (images, annotations)
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
        
        logger.info(f"Loaded {len(images)} images and {len(annotations)} annotations")
        logger.info(f"Categories: {[cat['name'] for cat in categories]}")
        
        return images, annotations
    
    def create_yolo_annotations(self, images: List[Dict], annotations: List[Dict]) -> Dict[str, List[str]]:
        """
        Convert COCO annotations to YOLO format
        
        Args:
            images: List of image metadata
            annotations: List of annotation data
            
        Returns:
            Dictionary mapping image filenames to YOLO annotation strings
        """
        yolo_annotations = {}
        
        # Create image ID to filename mapping
        image_id_to_filename = {img['id']: img['file_name'] for img in images}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in annotations:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Convert to YOLO format
        for image_id, image_anns in annotations_by_image.items():
            filename = image_id_to_filename[image_id]
            yolo_lines = []
            
            for ann in image_anns:
                # Get category name
                category_id = ann['category_id']
                category_name = None
                for cat in self.class_mapping.keys():
                    if category_id == self.class_mapping[cat]:
                        category_name = cat
                        break
                
                if category_name is None:
                    continue
                
                # Convert bbox from COCO to YOLO format
                x, y, w, h = ann['bbox']
                img_width = next(img['width'] for img in images if img['id'] == image_id)
                img_height = next(img['height'] for img in images if img['id'] == image_id)
                
                # Normalize coordinates
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width = w / img_width
                height = h / img_height
                
                # YOLO format: class_id x_center y_center width height
                yolo_line = f"{self.class_mapping[category_name]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)
            
            yolo_annotations[filename] = yolo_lines
        
        return yolo_annotations
    
    def get_augmentation_pipeline(self, is_training: bool = True) -> A.Compose:
        """
        Get augmentation pipeline for training/validation
        
        Args:
            is_training: Whether this is for training (more augmentations)
            
        Returns:
            Albumentations compose object
        """
        if is_training:
            transforms = A.Compose([
                A.Resize(height=self.config['image_size'][0], width=self.config['image_size'][1]),
                A.HorizontalFlip(p=self.config['augmentation']['horizontal_flip']),
                A.Rotate(limit=self.config['augmentation']['rotation'], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['augmentation']['brightness_contrast'],
                    contrast_limit=self.config['augmentation']['brightness_contrast'],
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=int(self.config['augmentation']['hue_saturation'] * 180),
                    sat_shift_limit=int(self.config['augmentation']['hue_saturation'] * 255),
                    val_shift_limit=int(self.config['augmentation']['hue_saturation'] * 255),
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            transforms = A.Compose([
                A.Resize(height=self.config['image_size'][0], width=self.config['image_size'][1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transforms
    
    def process_split(self, split_name: str, images: List[Dict], yolo_annotations: Dict[str, List[str]]) -> None:
        """
        Process a data split (train/validation/test)
        
        Args:
            split_name: Name of the split
            images: List of image metadata
            yolo_annotations: YOLO format annotations
        """
        split_dir = self.output_data_path / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        
        # Create parent directory first
        split_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        logger.info(f"Processing {split_name} split...")
        
        for img_data in images:
            filename = img_data['file_name']
            
            # Copy image
            src_image_path = self.input_data_path / split_name / filename
            dst_image_path = images_dir / filename
            
            if src_image_path.exists():
                shutil.copy2(src_image_path, dst_image_path)
                
                # Save YOLO annotation
                if filename in yolo_annotations:
                    label_filename = filename.replace('.jpg', '.txt')
                    label_path = labels_dir / label_filename
                    
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(yolo_annotations[filename]))
        
        logger.info(f"Processed {len(images)} images for {split_name}")
    
    def create_dataset_yaml(self) -> None:
        """Create dataset.yaml file for YOLO training"""
        yaml_content = {
            'path': str(self.output_data_path),
            'train': 'train/images',
            'val': 'validation/images',
            'test': 'test/images',
            'nc': len(self.class_mapping),
            'names': list(self.class_mapping.keys())
        }
        
        yaml_path = self.output_data_path / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        logger.info(f"Created dataset.yaml at {yaml_path}")
    
    def process_dataset(self) -> None:
        """Process the entire dataset"""
        logger.info("Starting dataset processing...")
        
        # Process each split
        for split in ['train', 'valid', 'test']:
            split_path = self.input_data_path / split
            if not split_path.exists():
                logger.warning(f"Split {split} not found, skipping...")
                continue
            
            # Load COCO annotations
            json_path = split_path / '_annotations.coco.json'
            if not json_path.exists():
                logger.warning(f"Annotations not found for {split}, skipping...")
                continue
            
            images, annotations = self.load_coco_annotations(str(json_path))
            
            # Convert to YOLO format
            yolo_annotations = self.create_yolo_annotations(images, annotations)
            
            # Process the split
            self.process_split(split, images, yolo_annotations)
        
        # Create dataset.yaml
        self.create_dataset_yaml()
        
        logger.info("Dataset processing completed!")


def main():
    """Main function for SageMaker entry point"""
    parser = argparse.ArgumentParser(description='VapeWatch Data Preprocessing')
    parser.add_argument('--input-data-path', type=str, required=True,
                       help='Path to input dataset')
    parser.add_argument('--output-data-path', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    
    args = parser.parse_args()
    
    # Load config if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Initialize preprocessor
    preprocessor = VapeDataPreprocessor(
        input_data_path=args.input_data_path,
        output_data_path=args.output_data_path,
        config=config
    )
    
    # Process dataset
    preprocessor.process_dataset()


if __name__ == "__main__":
    main()
