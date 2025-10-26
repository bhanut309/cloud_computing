# VapeWatch - Complete Deployment Guide

## 📋 Overview
This guide provides complete step-by-step instructions for the VapeWatch system including data preprocessing, training with 10 epochs, and batch testing on the trained model.

## 🚀 Quick Start

### 1. Setup Python Environment
```bash
# Activate Python 3.12 environment
source venv_3_12/bin/activate

# Verify environment
python --version
# Expected: Python 3.12.2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Process Data
```bash
python data_preprocessing.py --input-data-path VapeDataSet --output-data-path processed_data --config config.yaml
```

### 4. Train Model (10 epochs with multi-threading)
```bash
python train_vaping_detection.py --epochs 10 --batch-size 8
```

### 5. Test Trained Model

**Single Image Testing:**
```bash
python test_vaping_detection.py --image processed_data/test/images/sample.jpg
```

**Batch Testing on Entire Test Set:**
```bash
python batch_test_vaping.py
```

**With Custom Options:**
```bash
# Single image with custom confidence
python test_vaping_detection.py --image processed_data/test/images/sample.jpg --conf 0.3

# Batch test with custom settings
python batch_test_vaping.py --test-dir processed_data/test/images --conf 0.3 --max-images 100
```

---

## 📝 Complete Command Sequence

Here's the complete sequence to train and test the model:

```bash
# 1. Activate environment
source venv_3_12/bin/activate

# 2. Process data (if not already done)
python data_preprocessing.py --input-data-path VapeDataSet --output-data-path processed_data --config config.yaml

# 3. Train model (10 epochs)
python train_vaping_detection.py --epochs 10 --batch-size 8

# 4. Test trained model (batch inference on test set)
python batch_test_vaping.py

# 5. View results (results saved to separate test_results folder)
ls -la test_results/batch_test/
ls -la test_results/batch_test/labels/  # Annotation files
ls -la test_results/batch_test/*.jpg | head -20  # Test images with detections
```

## 📊 Step 1: Data Preprocessing

### 1.2 Run Data Preprocessing
```bash
# Activate environment first
source venv_3_12/bin/activate

# Run preprocessing (full command with all arguments)
python data_preprocessing.py \
  --input-data-path VapeDataSet \
  --output-data-path processed_data \
  --config config.yaml
```

**Short command (if already configured):**
```bash
# If config.yaml has defaults, you can use:
python data_preprocessing.py
```

**Note:** Make sure your dataset structure matches:
```
VapeDataSet/
├── train/
│   ├── images/*.jpg
│   └── _annotations.coco.json
├── valid/
│   ├── images/*.jpg
│   └── _annotations.coco.json
└── test/
    ├── images/*.jpg
    └── _annotations.coco.json
```

**Expected Output:**
```
INFO:__main__:Starting dataset processing...
INFO:__main__:Loaded 12114 images and 16128 annotations
INFO:__main__:Processing train split...
INFO:__main__:Processed 12114 images for train
INFO:__main__:Processing valid split...
INFO:__main__:Processed 1159 images for valid
INFO:__main__:Processing test split...
INFO:__main__:Processed 577 images for test
INFO:__main__:Dataset processing completed!
```

### 1.3 Verify Processing
```bash
# Check processed data structure
ls -la processed_data/
# Expected output: dataset.yaml, train/, valid/, test/

# Verify directory structure
tree processed_data/ -L 2

# Count processed files
find processed_data -name "*.jpg" | wc -l
# Expected: 13850 images (12114 + 1159 + 577)

# Verify annotations
find processed_data -name "*.txt" | wc -l
# Expected: 13850 annotation files
```

---

## 🏋️ Step 2: Model Training

### 2.1 Train with 10 Epochs (Multi-threaded)

```bash
# Activate environment
source venv_3_12/bin/activate

# Train with 10 epochs, batch size 8, multi-threading enabled
python train_vaping_detection.py --epochs 10 --batch-size 8
```

**Training Configuration:**
- **Epochs:** 10
- **Batch Size:** 8
- **Image Size:** 640x640
- **Multi-threading:** 8 workers (auto-detected)
- **Device:** CPU (auto-detected)
- **Augmentation:** Enabled (Mosaic, HSV, Flip)
- **Early Stopping:** Patience=20
- **Save Period:** Every 5 epochs

**Expected Training Output:**
```
INFO - 🚀 Multi-threading enabled with 8 workers
INFO - Using device: cpu
INFO - Validating dataset...
INFO - Dataset validation passed!
INFO - Classes: ['cigarette', 'vape']
INFO - Number of classes: 2
INFO - Loading yolov8n.pt...
INFO - Starting training...

Ultralytics 8.3.221 🚀 Python-3.12.2 torch-2.9.0 CPU (Apple M3 Pro)

train: epochs=10, imgsz=640, batch=8, cache=True, 
       device=cpu, workers=8, amp=False, close_mosaic=10
```

### 2.2 Monitor Training Progress

The training will show:
- Loss metrics (box_loss, cls_loss, dfl_loss)
- Validation metrics (precision, recall, mAP50, mAP50-95)
- Training plots (loss curves, confusion matrix)
- Batch visualization

**Training Time:**
- Estimated: 15-30 minutes for 10 epochs
- Actual time depends on CPU and dataset size

### 2.3 Check Training Results

```bash
# View training results
ls -la vaping_detection_results/

# Check trained model
ls -la vaping_detection_results/experiment_name/weights/
# Expected: best.pt, last.pt

# View training metrics
cat vaping_detection_results/experiment_name/results.csv

# View training plots
open vaping_detection_results/experiment_name/results.png
```

---

## 🧪 Step 3: Model Testing

### 3.1 Single Image Testing

```bash
# Test on a single image using the dedicated script
python test_vaping_detection.py --image processed_data/test/images/sample.jpg

# With custom confidence threshold
python test_vaping_detection.py --image processed_data/test/images/sample.jpg --conf 0.3

# With custom output directory
python test_vaping_detection.py --image processed_data/test/images/sample.jpg --output-dir test_results/custom_test
```

**Output:**
```
✅ Detections found: 1
  - vape: confidence=0.385
📁 Results saved to: test_results/single_test/
```

### 3.2 Batch Testing on Test Set

```bash
# Test on entire test set using the dedicated batch script
python batch_test_vaping.py

# With custom settings
python batch_test_vaping.py --conf 0.3 --max-images 100

# Test on custom directory
python batch_test_vaping.py --test-dir custom_test_images --output-dir test_results/custom_batch
```

**Check Results:**
```bash
# View test results directory
ls -la test_results/batch_test/

# Count images with detections
ls test_results/batch_test/*.jpg | wc -l

# View detection annotations
ls test_results/batch_test/labels/ | head -10

# View a specific detection file
cat test_results/batch_test/labels/[any_file].txt
```

**Batch Testing Output:**
```
📊 BATCH TESTING RESULTS:
✅ Batch testing completed!
📊 Total images processed: 577
📁 Results saved to: test_results/batch_test/
📁 Annotations saved to: test_results/batch_test/labels/

📈 DETECTION SUMMARY:
Total images: 577
Images with detections: ~365 (varies by model performance)
Detection rate: ~63%
Vape detections: Variable
Cigarette detections: Variable
Average confidence: Variable

📂 RESULTS FOLDER STRUCTURE:
test_results/
└── batch_test/
    ├── [image1].jpg (annotated image)
    ├── [image2].jpg
    ├── ...
    └── labels/
        ├── [image1].txt (YOLO format annotations)
        ├── [image2].txt
        └── ...
```

### 3.3 Validation Metrics

```bash
# Run validation on test set
python train_vaping_detection.py --validate-only --config vaping_training_config.yaml
```

**Expected Metrics:**
```
📊 VALIDATION METRICS:
mAP@0.5: 0.4276 (42.8%)
mAP@0.5:0.95: 0.1942 (19.4%)
Precision: 0.5267 (52.7%)
Recall: 0.4224 (42.2%)
F1-Score: 0.4689 (46.9%)
```

---

## ☁️ Step 4: AWS Deployment (Optional)

> **Note:** AWS deployment is optional. You can use the trained model locally without AWS.

## 🔑 Step 4.1: AWS Account Setup

### 3.1 Create AWS Account
1. Go to [AWS Console](https://aws.amazon.com/)
2. Click "Create an AWS Account"
3. Follow the registration process
4. Verify your email and phone number

### 3.2 Create IAM User
1. **Navigate to IAM**:
   - Go to AWS Console → Services → IAM
   - Click "Users" → "Create user"

2. **User Details**:
   - Username: `vapewatch-user`
   - Access type: "Programmatic access"

3. **Attach Policies**:
   - `AmazonS3FullAccess`
   - `AmazonSageMakerFullAccess`
   - `IAMFullAccess`

4. **Create Access Keys**:
   - Click "Create access key"
   - Download the CSV file
   - **Save these credentials securely!**

### 3.3 Create SageMaker Execution Role
1. **Navigate to IAM**:
   - Go to IAM → Roles → "Create role"

2. **Role Configuration**:
   - Trusted entity: "AWS service"
   - Service: "SageMaker"

3. **Attach Policies**:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess`
   - `CloudWatchLogsFullAccess`

4. **Role Details**:
   - Role name: `SageMakerExecutionRole`
   - Description: "Role for VapeWatch SageMaker operations"

5. **Note the ARN**: `arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole`

---

## 🔐 Step 4: AWS CLI Configuration

### 4.1 Install AWS CLI
```bash
# macOS
brew install awscli

# Linux
sudo apt-get install awscli

# Windows
# Download from https://aws.amazon.com/cli/
```

### 4.2 Configure AWS CLI
```bash
aws configure
```

**Enter your credentials:**
```
AWS Access Key ID: [YOUR_ACCESS_KEY]
AWS Secret Access Key: [YOUR_SECRET_KEY]
Default region name: us-east-1
Default output format: json
```

### 4.3 Verify Configuration
```bash
aws sts get-caller-identity
```

**Expected output:**
```json
{
    "UserId": "AIDACKCEVSQ6C2EXAMPLE",
    "Account": "YOUR-ACCOUNT-ID",
    "Arn": "arn:aws:iam::YOUR-ACCOUNT-ID:user/vapewatch-user"
}
```

---

## 🚀 Step 5: AWS Deployment

### 5.1 Deploy to SageMaker
```bash
python deploy.py deploy
```

**Expected Output:**
```
🚀 VapeWatch AWS Deployment
==================================================
📦 Model: vapewatch-model
🌐 Endpoint: vapewatch-endpoint
🔑 Role: arn:aws:iam::YOUR-ACCOUNT-ID:role/SageMakerExecutionRole

📡 Creating SageMaker model...
🚀 Deploying endpoint...
✅ Deployment successful!
🌐 Endpoint: vapewatch-endpoint
🔗 ARN: vapewatch-endpoint
```

### 5.2 Monitor Deployment
```bash
# List endpoints
python deploy.py list

# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name vapewatch-endpoint
```

---

## 🔍 Step 6: AWS Endpoint Testing

### 6.1 Test Deployed Endpoint
```bash
# Test with processed data
python test_aws_endpoint.py --image processed_data/test/images/[any_image].jpg --endpoint-name vapewatch-endpoint

# Test with original data
python test_aws_endpoint.py --image /path/to/VapeDataSet/test/[any_image].jpg --endpoint-name vapewatch-endpoint
```

### 6.2 List Available Endpoints
```bash
python test_aws_endpoint.py --list-endpoints
```

### 6.3 Expected Results
- **Console Output**: Detection count and details
- **Visual Files**: Comparison images showing detection and privacy protection
- **JSON Data**: Structured detection results

**Sample Output:**
```
🧪 Testing endpoint...
📸 Processing: processed_data/test/images/sample.jpg
✅ Prediction successful!
📊 Detections: 1
🚭 Vape detected: true
🚬 Cigarette detected: false
👤 Faces blurred: 2
```

---

## 🛠️ Step 7: Management Commands

### 7.1 Endpoint Management
```bash
# List endpoints
python deploy.py list

# Delete endpoint
python deploy.py delete
```

### 7.2 AWS CLI Commands
```bash
# Check endpoint status
aws sagemaker describe-endpoint --endpoint-name vapewatch-endpoint

# View logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Endpoints
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 2. AWS Authentication Errors
```bash
# Solution: Reconfigure AWS CLI
aws configure
```

#### 3. Model Not Found
```bash
# Solution: Ensure yolov8n.pt is in the directory
ls -la yolov8n.pt
```

#### 4. Endpoint Timeout
```bash
# Solution: Check CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Endpoints
```

#### 5. S3 Access Denied
```bash
# Solution: Attach S3 policy to SageMaker role
aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

#### 6. Data Processing Errors
```bash
# Solution: Check input data format
ls -la /path/to/VapeDataSet/train/
# Should contain: images and _annotations.coco.json
```

---

## 📊 Output Format

### Detection Results
```json
{
  "image_path": "processed_data/test/images/sample.jpg",
  "detections": [
    {
      "class": "vape",
      "confidence": 0.85,
      "bbox": {
        "x1": 100, "y1": 200, "x2": 300, "y2": 400
      }
    }
  ],
  "faces_blurred": 2,
  "total_detections": 1,
  "vape_detected": true,
  "cigarette_detected": false
}
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:
```yaml
model:
  confidence_threshold: 0.5
  face_blur_strength: 15

aws:
  region: us-east-1
  endpoint_name: vapewatch-endpoint
```

---

## ✅ Complete Checklist

### Environment Setup
- [ ] Python 3.12 environment activated (`source venv_3_12/bin/activate`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] lzma issue resolved (xz library installed)

### Data Processing
- [ ] Data preprocessing completed (`python data_preprocessing.py`)
- [ ] Dataset verified (13,850 images processed)
- [ ] YOLO format annotations created

### Model Training
- [ ] Model trained with 10 epochs (`python train_vaping_detection.py --epochs 10 --batch-size 8`)
- [ ] Training logs reviewed
- [ ] Best model saved (`vaping_detection_results/*/weights/best.pt`)

### Model Testing
- [ ] Single image testing completed
- [ ] Batch testing on test set completed
- [ ] Validation metrics reviewed
- [ ] Results visualized and analyzed

### Optional: AWS Deployment
- [ ] AWS account created and configured
- [ ] IAM user and role created
- [ ] AWS CLI configured
- [ ] AWS endpoint deployed (`python deploy.py deploy`)
- [ ] Endpoint testing completed (`python test_aws_endpoint.py`)
- [ ] System ready for production
