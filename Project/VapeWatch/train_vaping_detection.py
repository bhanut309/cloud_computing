#!/usr/bin/env python3
"""
VapeWatch - Vaping Device Detection Training Script
Optimized for YOLOv8 training on processed dataset
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class VapingDetectionTrainer:
    """
    Trainer class for vaping device detection using YOLOv8
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trainer
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.results = None
        
        # Multi-threading configuration
        self.num_workers = min(mp.cpu_count(), config.get('workers', 8))
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
        
        logger.info(f"🚀 Multi-threading enabled with {self.num_workers} workers")
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'mAP50': [],
            'mAP50-95': []
        }
        
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model checkpoints directory
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Results directory
        self.results_dir = self.output_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_size: str = 'n') -> None:
        """
        Load YOLOv8 model
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'x')
        """
        model_name = f'yolov8{model_size}.pt'
        logger.info(f"Loading {model_name}...")
        
        # Download pretrained model if not exists
        if not os.path.exists(model_name):
            logger.info(f"Downloading {model_name}...")
            self.model = YOLO(model_name)
        else:
            self.model = YOLO(model_name)
            
        logger.info(f"Model loaded successfully: {model_name}")
    
    def validate_dataset(self, dataset_path: str) -> bool:
        """
        Validate the dataset structure and annotations
        
        Args:
            dataset_path: Path to dataset.yaml
            
        Returns:
            bool: True if dataset is valid
        """
        logger.info("Validating dataset...")
        
        try:
            # Load dataset config
            with open(dataset_path, 'r') as f:
                dataset_config = yaml.safe_load(f)
            
            # Check required keys
            required_keys = ['path', 'train', 'val', 'nc', 'names']
            for key in required_keys:
                if key not in dataset_config:
                    logger.error(f"Missing required key in dataset.yaml: {key}")
                    return False
            
            # Check if directories exist
            base_path = Path(dataset_config['path'])
            for split in ['train', 'val']:
                split_path = base_path / dataset_config[split]
                if not split_path.exists():
                    logger.error(f"Split directory not found: {split_path}")
                    return False
                
                # Check for images and labels
                images_dir = split_path / 'images' if (split_path / 'images').exists() else split_path
                labels_dir = split_path / 'labels' if (split_path / 'labels').exists() else split_path
                
                if not images_dir.exists() or not labels_dir.exists():
                    logger.error(f"Images or labels directory not found for {split}")
                    return False
            
            logger.info("Dataset validation passed!")
            logger.info(f"Classes: {dataset_config['names']}")
            logger.info(f"Number of classes: {dataset_config['nc']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False
    
    def setup_training_args(self) -> Dict[str, Any]:
        """
        Setup training arguments based on configuration
        
        Returns:
            Dictionary of training arguments
        """
        args = {
            'data': self.config['dataset_path'],
            'epochs': self.config['epochs'],
            'imgsz': self.config['image_size'],
            'batch': self.config['batch_size'],
            'device': self.device,
            'project': str(self.output_dir),
            'name': self.config['experiment_name'],
            'save': True,
            'save_period': self.config.get('save_period', 10),
            'cache': self.config.get('cache', False),
            'workers': self.config.get('workers', 8),
            'patience': self.config.get('patience', 50),
            'lr0': self.config.get('learning_rate', 0.01),
            'lrf': self.config.get('lr_final', 0.01),
            'momentum': self.config.get('momentum', 0.937),
            'weight_decay': self.config.get('weight_decay', 0.0005),
            'warmup_epochs': self.config.get('warmup_epochs', 3),
            'warmup_momentum': self.config.get('warmup_momentum', 0.8),
            'warmup_bias_lr': self.config.get('warmup_bias_lr', 0.1),
            'box': self.config.get('box_loss_gain', 0.05),
            'cls': self.config.get('cls_loss_gain', 0.5),
            'dfl': self.config.get('dfl_loss_gain', 1.5),
            'pose': self.config.get('pose_loss_gain', 12.0),
            'kobj': self.config.get('kobj_loss_gain', 2.0),
            'label_smoothing': self.config.get('label_smoothing', 0.0),
            'nbs': self.config.get('nominal_batch_size', 64),
            'overlap_mask': self.config.get('overlap_mask', True),
            'mask_ratio': self.config.get('mask_ratio', 4),
            'dropout': self.config.get('dropout', 0.0),
            'val': True,
            'plots': True,
            'verbose': True,
        }
        
        # Add augmentation parameters
        if 'augmentation' in self.config:
            aug_config = self.config['augmentation']
            args.update({
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'hsv_v': aug_config.get('hsv_v', 0.4),
                'degrees': aug_config.get('degrees', 0.0),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.5),
                'shear': aug_config.get('shear', 0.0),
                'perspective': aug_config.get('perspective', 0.0),
                'flipud': aug_config.get('flipud', 0.0),
                'fliplr': aug_config.get('fliplr', 0.5),
                'mosaic': aug_config.get('mosaic', 1.0),
                'mixup': aug_config.get('mixup', 0.0),
                'copy_paste': aug_config.get('copy_paste', 0.0),
            })
        
        # Add multi-threading and performance optimizations
        # Auto-detect device: use CUDA if available, otherwise CPU
        device = self.config.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        args.update({
            'workers': self.num_workers,  # Number of data loading workers
            'device': device,  # Use detected device
            'cache': True,  # Cache images for faster training
            'amp': torch.cuda.is_available(),  # Only use AMP if CUDA is available
            'close_mosaic': 10,  # Close mosaic augmentation in last 10 epochs
            'patience': self.config.get('patience', 50),  # Early stopping patience
            'save_period': 5,  # Save checkpoint every 5 epochs
            'val': True,  # Enable validation during training
            'plots': True,  # Generate training plots
            'verbose': True,  # Verbose output
        })
        
        return args
    
    def train(self) -> None:
        """
        Train the model
        """
        logger.info("Starting training...")
        start_time = time.time()
        
        try:
            # Setup training arguments
            train_args = self.setup_training_args()
            
            # Start training
            self.results = self.model.train(**train_args)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Save training results
            self.save_training_results()
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            # Clean up thread pool
            self.thread_pool.shutdown(wait=True)
    
    def save_training_results(self) -> None:
        """
        Save training results and metrics
        """
        logger.info("Saving training results...")
        
        try:
            # Save model
            best_model_path = self.results_dir / 'best.pt'
            last_model_path = self.results_dir / 'last.pt'
            
            if hasattr(self.results, 'best'):
                torch.save(self.results.best, best_model_path)
                logger.info(f"Best model saved to: {best_model_path}")
            
            if hasattr(self.results, 'last'):
                torch.save(self.results.last, last_model_path)
                logger.info(f"Last model saved to: {last_model_path}")
            
            # Save training configuration
            config_path = self.results_dir / 'training_config.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            # Save training metrics if available
            if hasattr(self.results, 'results_dict'):
                metrics_path = self.results_dir / 'training_metrics.json'
                with open(metrics_path, 'w') as f:
                    json.dump(self.results.results_dict, f, indent=2)
            
            logger.info("Training results saved successfully!")
            
        except Exception as e:
            logger.error(f"Failed to save training results: {e}")
    
    def validate_model(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """
        Validate the trained model
        
        Args:
            model_path: Path to model weights (if None, uses best model)
            
        Returns:
            Dictionary of validation metrics
        """
        logger.info("Validating model...")
        
        try:
            if model_path is None:
                model_path = self.results_dir / 'best.pt'
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found: {model_path}")
                return {}
            
            # Load model
            model = YOLO(str(model_path))
            
            # Run validation
            val_results = model.val(
                data=self.config['dataset_path'],
                imgsz=self.config['image_size'],
                batch=self.config['batch_size'],
                device=self.device,
                plots=True,
                save_json=True,
                project=str(self.results_dir),
                name='validation'
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(val_results.box.map50),
                'mAP50-95': float(val_results.box.map),
                'precision': float(val_results.box.mp),
                'recall': float(val_results.box.mr),
                'f1': float(val_results.box.f1)
            }
            
            logger.info("Validation completed!")
            logger.info(f"mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1-Score: {metrics['f1']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}
    
    def test_inference(self, model_path: Optional[str] = None, test_image_path: Optional[str] = None) -> None:
        """
        Test model inference on sample images
        
        Args:
            model_path: Path to model weights
            test_image_path: Path to test image
        """
        logger.info("Testing model inference...")
        
        try:
            if model_path is None:
                model_path = self.results_dir / 'best.pt'
            
            if not os.path.exists(model_path):
                logger.error(f"Model not found: {model_path}")
                return
            
            # Load model
            model = YOLO(str(model_path))
            
            # Test on sample images
            if test_image_path and os.path.exists(test_image_path):
                # Test on specific image
                results = model(test_image_path, save=True, project=str(self.results_dir), name='inference')
                logger.info(f"Inference completed on: {test_image_path}")
            else:
                # Test on validation images
                val_images_dir = Path(self.config['dataset_path']).parent / 'valid' / 'images'
                if val_images_dir.exists():
                    sample_images = list(val_images_dir.glob('*.jpg'))[:5]
                    for img_path in sample_images:
                        results = model(str(img_path), save=True, project=str(self.results_dir), name='inference')
                        logger.info(f"Inference completed on: {img_path.name}")
            
            logger.info("Inference testing completed!")
            
        except Exception as e:
            logger.error(f"Inference testing failed: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration for vaping detection training
    
    Returns:
        Default configuration dictionary
    """
    return {
        'dataset_path': 'processed_data/dataset.yaml',
        'output_dir': 'vaping_detection_results',
        'experiment_name': 'vaping_detection_v1',
        'model_size': 'n',  # n, s, m, l, x
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
        'learning_rate': 0.01,
        'lr_final': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 50,
        'save_period': 10,
        'cache': False,
        'workers': 8,
        'augmentation': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0,
        },
        'box_loss_gain': 0.05,
        'cls_loss_gain': 0.5,
        'dfl_loss_gain': 1.5,
        'label_smoothing': 0.0,
        'nominal_batch_size': 64,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
    }


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='VapeWatch Training Script')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration YAML file')
    parser.add_argument('--dataset', type=str, default='processed_data/dataset.yaml',
                       help='Path to dataset YAML file')
    parser.add_argument('--output-dir', type=str, default='vaping_detection_results',
                       help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--model-size', type=str, default='n',
                       choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate the dataset without training')
    parser.add_argument('--test-inference', action='store_true',
                       help='Test inference on sample images')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    else:
        config = create_default_config()
        logger.info("Using default configuration")
    
    # Override with command line arguments
    config['dataset_path'] = args.dataset
    config['output_dir'] = args.output_dir
    config['epochs'] = args.epochs
    config['batch_size'] = args.batch_size
    config['model_size'] = args.model_size
    
    # Initialize trainer
    trainer = VapingDetectionTrainer(config)
    
    # Validate dataset
    if not trainer.validate_dataset(config['dataset_path']):
        logger.error("Dataset validation failed. Exiting.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Dataset validation completed successfully!")
        return
    
    # Load model
    trainer.load_model(config['model_size'])
    
    if args.test_inference:
        # Test inference only
        trainer.test_inference()
        return
    
    # Train model
    trainer.train()
    
    # Validate trained model
    metrics = trainer.validate_model()
    
    # Test inference
    trainer.test_inference()
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
