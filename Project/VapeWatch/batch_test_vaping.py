#!/usr/bin/env python3
"""
VapeWatch Batch Testing Script
Tests the trained model on multiple images (test set)
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from collections import Counter

from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_test.log'),
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


def batch_test(model_path, test_dir, output_dir='test_results/batch_test', conf=0.25):
    """
    Test model on multiple images
    
    Args:
        model_path: Path to trained model
        test_dir: Directory containing test images
        output_dir: Directory to save results
        conf: Confidence threshold
    """
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    
    logger.info(f"Testing directory: {test_dir}")
    
    # Count total images
    test_path = Path(test_dir)
    total_images = len(list(test_path.glob('*.jpg'))) + len(list(test_path.glob('*.png')))
    
    if total_images == 0:
        logger.error(f"No images found in {test_dir}")
        return
    
    logger.info(f"Found {total_images} images to test")
    
    # Run batch inference
    logger.info("Starting batch inference...")
    results = model(
        str(test_dir),
        save=True,
        save_txt=True,
        conf=conf,
        project=str(Path(output_dir).parent),
        name=Path(output_dir).name,
        verbose=True
    )
    
    # Count results
    output_path = Path(output_dir)
    labels_path = output_path / 'labels'
    
    # Count images with detections
    images_with_detections = 0
    total_detections = 0
    class_counts = Counter()
    confidence_scores = []
    
    if labels_path.exists():
        for label_file in labels_path.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    images_with_detections += 1
                    for line in lines:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) >= 6:
                                class_id = int(parts[0])
                                confidence = float(parts[5])
                                class_counts[class_id] += 1
                                confidence_scores.append(confidence)
                                total_detections += 1
    
    # Get class names
    class_names = model.names
    
    # Print summary
    logger.info("=" * 50)
    logger.info("📊 BATCH TESTING RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total images: {total_images}")
    logger.info(f"Images with detections: {images_with_detections}")
    logger.info(f"Detection rate: {images_with_detections/total_images*100:.1f}%")
    logger.info(f"Total detections: {total_detections}")
    
    logger.info(f"\n📈 Detection breakdown by class:")
    for class_id, count in class_counts.most_common():
        class_name = class_names[class_id]
        logger.info(f"  - {class_name}: {count}")
    
    if confidence_scores:
        avg_conf = sum(confidence_scores) / len(confidence_scores)
        logger.info(f"\n🎯 Average confidence: {avg_conf:.3f}")
    
    logger.info(f"\n📁 Results saved to: {output_dir}")
    logger.info(f"  - Annotated images: {output_dir}/*.jpg")
    logger.info(f"  - Annotations: {output_dir}/labels/*.txt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Batch test trained VapeWatch model')
    parser.add_argument('--test-dir', type=str, default='processed_data/test/images',
                       help='Directory containing test images')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model (auto-detects if not specified)')
    parser.add_argument('--output-dir', type=str, default='test_results/batch_test',
                       help='Directory to save results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to test (None for all)')
    
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
    
    # Check if test directory exists
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        logger.info("Please make sure processed_data/test/images exists")
        sys.exit(1)
    
    # Run batch test
    logger.info("=" * 50)
    logger.info("🚀 VapeWatch Batch Testing")
    logger.info("=" * 50)
    
    batch_test(model_path, test_dir, args.output_dir, args.conf)
    
    logger.info("✅ Batch testing completed successfully!")


if __name__ == "__main__":
    main()

