#!/usr/bin/env python3
"""
VapeWatch - AWS Deployment Script
Deploy trained YOLO model to SageMaker endpoint
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import json
import base64
from pathlib import Path
import sys
import os

def deploy_to_sagemaker(use_trained_model=True, model_s3_path=None):
    """
    Deploy YOLO model to SageMaker endpoint
    
    Args:
        use_trained_model: If True, uses trained model (best.pt), else uses pretrained (yolov8n.pt)
        model_s3_path: S3 path to model file. If None, will use default paths
    """
    
    print("🚀 VapeWatch AWS Deployment")
    print("=" * 50)
    
    # Initialize SageMaker
    session = sagemaker.Session()
    role = 'arn:aws:iam::156999051350:role/SageMakerExecutionRole'
    
    # Configuration
    endpoint_name = 'vapewatch-endpoint'
    model_name = 'vapewatch-model'
    
    print(f"📦 Model: {model_name}")
    print(f"🌐 Endpoint: {endpoint_name}")
    print(f"🔑 Role: {role}")
    
    # Determine model S3 path
    if model_s3_path is None:
        if use_trained_model:
            # Default path for trained model
            model_s3_path = 's3://sagemaker-us-east-1-156999051350/vapewatch-trained/best.pt'
            print(f"📁 Using trained model: {model_s3_path}")
        else:
            # Default path for pretrained model
            model_s3_path = 's3://sagemaker-us-east-1-156999051350/vapewatch-pretrained/yolov8n.pt'
            print(f"📁 Using pretrained model: {model_s3_path}")
    
    # Create PyTorch model
    print("\n📡 Creating SageMaker model...")
    model = PyTorchModel(
        model_data=model_s3_path,
        role=role,
        entry_point='inference.py',  # Updated to use inference.py
        source_dir='.',
        framework_version='2.0.0',
        py_version='py310'
    )
    
    # Deploy endpoint
    print("🚀 Deploying endpoint...")
    try:
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type='ml.t2.medium',
            endpoint_name=endpoint_name,
            wait=True
        )
        
        print("✅ Deployment successful!")
        print(f"🌐 Endpoint: {endpoint_name}")
        print(f"🔗 ARN: {predictor.endpoint_name}")
        
        return predictor
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        return None

def test_endpoint():
    """Test the deployed endpoint"""
    
    print("\n🧪 Testing endpoint...")
    
    # Initialize SageMaker runtime client
    sagemaker_runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
    
    # Test image path
    test_image = "test_images/sample.jpg"
    
    if not Path(test_image).exists():
        print(f"⚠️  Test image not found: {test_image}")
        print("📝 Please add a test image to test_images/sample.jpg")
        return False
    
    # Read and encode image
    with open(test_image, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Prepare input
    input_data = {
        "image": image_data,
        "confidence_threshold": 0.5
    }
    
    # Make prediction
    try:
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName='vapewatch-endpoint',
            ContentType='application/json',
            Body=json.dumps(input_data)
        )
        
        result = json.loads(response['Body'].read().decode())
        
        print("✅ Test successful!")
        print(f"📊 Detections: {result.get('total_detections', 0)}")
        print(f"🚭 Vape detected: {result.get('vape_detected', False)}")
        print(f"🚬 Cigarette detected: {result.get('cigarette_detected', False)}")
        print(f"👤 Faces blurred: {result.get('faces_blurred', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def list_endpoints():
    """List all SageMaker endpoints"""
    
    print("\n📋 Listing endpoints...")
    
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    
    try:
        response = sagemaker_client.list_endpoints()
        endpoints = response.get('Endpoints', [])
        
        if endpoints:
            print("Available endpoints:")
            for endpoint in endpoints:
                status = endpoint['EndpointStatus']
                name = endpoint['EndpointName']
                print(f"  - {name} ({status})")
        else:
            print("No endpoints found")
            
    except Exception as e:
        print(f"❌ Failed to list endpoints: {e}")

def delete_endpoint():
    """Delete the endpoint"""
    
    print("\n🗑️  Deleting endpoint...")
    
    sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')
    
    try:
        sagemaker_client.delete_endpoint(EndpointName='vapewatch-endpoint')
        print("✅ Endpoint deleted successfully")
    except Exception as e:
        print(f"❌ Failed to delete endpoint: {e}")

def upload_model_to_s3(local_model_path, s3_bucket, s3_key_prefix='vapewatch-trained'):
    """
    Upload trained model to S3 for SageMaker deployment
    
    Args:
        local_model_path: Local path to best.pt model
        s3_bucket: S3 bucket name
        s3_key_prefix: S3 key prefix for the model file
    """
    print(f"📤 Uploading model to S3...")
    print(f"   Local: {local_model_path}")
    
    model_path = Path(local_model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {local_model_path}")
    
    s3_client = boto3.client('s3')
    
    # Construct S3 path
    s3_key = f"{s3_key_prefix}/{model_path.name}"
    s3_path = f"s3://{s3_bucket}/{s3_key}"
    
    print(f"   S3: {s3_path}")
    
    try:
        s3_client.upload_file(str(model_path), s3_bucket, s3_key)
        print(f"✅ Model uploaded successfully!")
        return s3_path
    except Exception as e:
        print(f"❌ Failed to upload model: {e}")
        raise

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python deploy.py deploy [--pretrained] [--model-s3-path S3_PATH]")
        print("                          - Deploy to SageMaker")
        print("  python deploy.py upload [--model-path LOCAL_PATH]")
        print("                          - Upload trained model to S3")
        print("  python deploy.py test   - Test endpoint")
        print("  python deploy.py list   - List endpoints")
        print("  python deploy.py delete  - Delete endpoint")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == 'deploy':
        # Check for flags
        use_trained = '--pretrained' not in sys.argv
        model_s3_path = None
        
        if '--model-s3-path' in sys.argv:
            idx = sys.argv.index('--model-s3-path')
            if idx + 1 < len(sys.argv):
                model_s3_path = sys.argv[idx + 1]
        
        deploy_to_sagemaker(use_trained_model=use_trained, model_s3_path=model_s3_path)
    elif action == 'upload':
        # Find best.pt model
        model_path = 'vaping_detection_results/vaping_detection_v1/weights/best.pt'
        
        if '--model-path' in sys.argv:
            idx = sys.argv.index('--model-path')
            if idx + 1 < len(sys.argv):
                model_path = sys.argv[idx + 1]
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            sys.exit(1)
        
        # Upload to S3
        bucket = 'sagemaker-us-east-1-156999051350'
        s3_path = upload_model_to_s3(model_path, bucket)
        print(f"\n✅ Upload complete! Use this S3 path for deployment:")
        print(f"   {s3_path}")
    elif action == 'test':
        test_endpoint()
    elif action == 'list':
        list_endpoints()
    elif action == 'delete':
        delete_endpoint()
    else:
        print(f"❌ Unknown action: {action}")
        sys.exit(1)

if __name__ == "__main__":
    main()
