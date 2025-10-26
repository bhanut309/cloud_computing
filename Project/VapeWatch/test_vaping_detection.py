#!/usr/bin/env python3
"""
VapeWatch Single Image Testing Script
Tests the trained model on a single image
"""

import os
import sys
import argparse
from pathlib import Path
import logging

from ultralytics import YOLO
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_single.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def find_trained_model(model_dir='vaping_detection_results'):
    """
    Find the best trained model
    
    Args:
        model_dir: Directory containing training results
        
    Returns:
        Path to best.pt model file
    """
    model_dir = Path(model_dir)
    
    # Search for best.pt in all subdirectories
    best_models = list(model_dir.rglob('best.pt'))
    
    if not best_models:
        logger.error(f"No trained model found in {model_dir}")
        logger.info("Please train a model first using: python train_vaping_detection.py --epochs 10 --batch-size 8")
        return None
    
    # Return the most recent best.pt
    best_model = max(best_models, key=lambda x: x.stat().st_mtime)
    logger.info(f"Found trained model: {best_model}")
    
    return best_model


def test_single_image(model_path, image_path, output_dir='test_results/single_test', conf=0.25):
    """
    Test model on a single image
    
    Args:
        model_path: Path to trained model
        image_path: Path to test image
        output_dir: Directory to save results
        conf: Confidence threshold
    """
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    logger.info(f"Testing image: {image_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run inference
    results = model(
        str(image_path),
        save=True,
        conf=conf,
        project=str(output_dir.parent),
        name=output_dir.name,
        save_txt=True
    )
    
    # Get detection results
    result = results[0]
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
        logger.info(f"✅ Detections found: {len(boxes)}")
        
        # Count by class
        class_counts = {}
        for box in boxes:
            cls = int(box.cls.item())
            class_name = model.names[cls]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            conf_val = box.conf.item()
            logger.info(f"  - {class_name}: confidence={conf_val:.3f}")
        
        logger.info(f"\n📊 Detection Summary:")
        for class_name, count in class_counts.items():
            logger.info(f"  - {class_name}: {count}")
    else:
        logger.info("❌ No detections found")
    
    logger.info(f"📁 Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test trained VapeWatch model on a single image')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to test image')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (auto-detects if not specified)')
    parser.add_argument('--output-dir', type=str, default='test_results/single_test',
                       help='Directory to save results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    
    args = parser.parse_args()
    
    # Find model if not specified
    if args.model is None:
        model_path = find_trained_model()
        if model_path is None:
            sys.exit(1)
    else:
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            sys.exit(1)
    
    # Check if image exists
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    # Run test
    logger.info("=" * 50)
    logger.info("🚀 VapeWatch Single Image Testing")
    logger.info("=" * 50)
    
    test_single_image(model_path, image_path, args.output_dir, args.conf)
    
    logger.info("✅ Testing completed successfully!")


if __name__ == "__main__":
    main()

