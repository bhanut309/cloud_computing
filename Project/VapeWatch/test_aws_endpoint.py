#!/usr/bin/env python3
"""
VapeWatch AWS SageMaker Endpoint Testing Script
Tests the deployed endpoint directly without local model
"""

import os
import sys
import json
import base64
import cv2
import numpy as np
import boto3
from botocore.config import Config
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def test_aws_endpoint(image_path, endpoint_name, output_dir="aws_test_results"):
    """
    Test the SageMaker endpoint directly
    """
    try:
        print("🚀 VapeWatch AWS Endpoint Testing")
        print("=" * 50)
        print(f"📸 Image: {image_path}")
        print(f"🔗 Endpoint: {endpoint_name}")
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found: {image_path}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load and encode image
        print("🖼️ Loading and encoding image...")
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
            image_b64 = base64.b64encode(image_bytes).decode()
        
        # Get original image for display
        original_image = cv2.imread(image_path)
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Create SageMaker client with longer timeout
        print("🔗 Connecting to SageMaker endpoint...")
        sagemaker_runtime = boto3.client(
            'sagemaker-runtime',
            config=Config(
                read_timeout=300,  # 5 minutes
                connect_timeout=60  # 1 minute
            )
        )
        
        # Prepare input data
        input_data = {
            'image': image_b64,
            'confidence_threshold': 0.5
        }
        
        # Make prediction
        print("🔍 Running vape detection on AWS...")
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(input_data),
            CustomAttributes='accept_eula=true'
        )
        
        # Parse response
        result = json.loads(response['Body'].read())
        detections = result.get('detections', [])
        
        print(f"✅ Found {len(detections)} detections")
        
        # Print detection details
        for i, detection in enumerate(detections):
            print(f"  Detection {i+1}: {detection.get('class_name', 'unknown')} "
                  f"(confidence: {detection.get('confidence', 0):.2f})")
        
        # Apply face blurring locally
        print("🔒 Applying privacy protection (face blurring)...")
        privacy_image = apply_face_blur(original_image_rgb.copy())
        
        # Draw detections on original image
        detection_image = draw_detections(original_image_rgb.copy(), detections)
        
        # Create comparison visualization
        print("📊 Creating validation visualization...")
        create_validation_plot(
            original_image_rgb,
            detection_image,
            privacy_image,
            detections,
            output_dir
        )
        
        # Save individual results
        save_results(
            original_image_rgb,
            detection_image,
            privacy_image,
            detections,
            output_dir
        )
        
        print(f"\n✅ AWS endpoint testing successful!")
        print(f"📁 Results saved to: {output_dir}/")
        print(f"🖼️ Check the generated images to see vape detection and face blurring")
        
        return True
        
    except Exception as e:
        print(f"❌ AWS endpoint testing failed: {e}")
        return False

def apply_face_blur(image):
    """Apply face blurring for privacy protection"""
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces with multiple scales and parameters
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"🔍 Detected {len(faces)} faces")
        
        # If no faces detected, try alternative approach
        if len(faces) == 0:
            print("🔍 No faces detected with standard method, trying alternative...")
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05, 
                minNeighbors=2, 
                minSize=(20, 20)
            )
            print(f"🔍 Alternative method found {len(faces)} faces")
        
        # Blur detected faces
        for (x, y, w, h) in faces:
            # Expand the blur area slightly
            x1 = max(0, x - 10)
            y1 = max(0, y - 10)
            x2 = min(image.shape[1], x + w + 10)
            y2 = min(image.shape[0], y + h + 10)
            
            face_region = image[y1:y2, x1:x2]
            if face_region.size > 0:
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                image[y1:y2, x1:x2] = blurred_face
        
        print(f"🔒 Blurred {len(faces)} faces for privacy protection")
        return image
    except Exception as e:
        print(f"⚠️ Face blurring failed: {e}")
        return image

def draw_detections(image, detections):
    """Draw detection boxes on image"""
    for detection in detections:
        bbox = detection.get('bbox', {})
        class_name = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0)
        
        # Get bounding box coordinates
        x1 = int(bbox.get('x1', 0))
        y1 = int(bbox.get('y1', 0))
        x2 = int(bbox.get('x2', 0))
        y2 = int(bbox.get('y2', 0))
        
        # Draw bounding box
        color = (255, 0, 0) if 'vape' in class_name.lower() else (0, 255, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def create_validation_plot(original, detection, privacy, detections, output_dir):
    """Create a comparison plot showing all results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Detection image
    axes[1].imshow(detection)
    axes[1].set_title(f'Vape Detection ({len(detections)} found)')
    axes[1].axis('off')
    
    # Privacy image
    axes[2].imshow(privacy)
    axes[2].set_title('Privacy Protected')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aws_validation_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📊 Created AWS validation comparison plot")

def save_results(original, detection, privacy, detections, output_dir):
    """Save individual result images"""
    # Save original
    Image.fromarray(original).save(f"{output_dir}/01_original.jpg")
    
    # Save detection result
    Image.fromarray(detection).save(f"{output_dir}/02_vape_detection.jpg")
    
    # Save privacy protected
    Image.fromarray(privacy).save(f"{output_dir}/03_privacy_protected.jpg")
    
    # Save detection data
    with open(f"{output_dir}/aws_detections.json", 'w') as f:
        json.dump(detections, f, indent=2)
    
    print("💾 Saved individual result images and detection data")

def list_endpoints():
    """List available SageMaker endpoints"""
    try:
        sagemaker = boto3.client('sagemaker')
        response = sagemaker.list_endpoints()
        
        print("🔗 Available SageMaker Endpoints:")
        print("-" * 40)
        
        for endpoint in response['Endpoints']:
            endpoint_name = endpoint['EndpointName']
            status = endpoint['EndpointStatus']
            print(f"  {endpoint_name} - {status}")
        
        return response['Endpoints']
    except Exception as e:
        print(f"❌ Failed to list endpoints: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Test VapeWatch AWS SageMaker endpoint')
    parser.add_argument('--image-path', type=str, required=False,
                       help='Path to test image')
    parser.add_argument('--endpoint-name', type=str, default=None,
                       help='SageMaker endpoint name (if not provided, will list available)')
    parser.add_argument('--output-dir', type=str, default='aws_test_results',
                       help='Directory to save test results')
    parser.add_argument('--list-endpoints', action='store_true',
                       help='List available endpoints and exit')
    
    args = parser.parse_args()
    
    if args.list_endpoints:
        list_endpoints()
        return
    
    if not args.image_path:
        print("❌ Please provide --image-path for testing")
        return
    
    if not args.endpoint_name:
        print("❌ Please provide endpoint name or use --list-endpoints to see available endpoints")
        return
    
    success = test_aws_endpoint(
        args.image_path, 
        args.endpoint_name, 
        args.output_dir
    )
    
    if success:
        print("\n🎉 AWS endpoint testing successful!")
        print("📁 Check the aws_test_results/ folder for:")
        print("  - aws_validation_comparison.png (side-by-side comparison)")
        print("  - 01_original.jpg (original image)")
        print("  - 02_vape_detection.jpg (with detection boxes)")
        print("  - 03_privacy_protected.jpg (with face blurring)")
        print("  - aws_detections.json (detection data)")
    else:
        print("\n❌ AWS endpoint testing failed")
        print("🔧 Check your endpoint status and try again")

if __name__ == "__main__":
    main()
