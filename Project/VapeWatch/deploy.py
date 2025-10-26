#!/usr/bin/env python3
"""
VapeWatch - AWS Deployment Script
Deploy pre-trained YOLO model to SageMaker endpoint
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import json
import base64
from pathlib import Path
import sys

def deploy_to_sagemaker():
    """Deploy pre-trained YOLO model to SageMaker"""
    
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
    
    # Create PyTorch model
    print("\n📡 Creating SageMaker model...")
    model = PyTorchModel(
        model_data='s3://sagemaker-us-east-1-156999051350/vapewatch-pretrained/yolov8n.pt',
        role=role,
        entry_point='inference_pretrained.py',
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

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python deploy.py deploy    - Deploy to SageMaker")
        print("  python deploy.py test      - Test endpoint")
        print("  python deploy.py list      - List endpoints")
        print("  python deploy.py delete    - Delete endpoint")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == 'deploy':
        deploy_to_sagemaker()
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
