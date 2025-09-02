# Step 1: Install YOLOv8
!pip install ultralytics

# Step 2: Import and GPU Check
from ultralytics import YOLO
import torch
import os
import yaml
import matplotlib.pyplot as plt
from IPython.display import Image, display
import shutil

print("✅ GPU Available:", torch.cuda.is_available())
print(f"🔧 PyTorch Version: {torch._version_}")
print(f"🔧 CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'Not Available'}")

# Step 3: Set dataset path
dataset_path = '/kaggle/input/objectdetection/Baja'
print(f"📁 Dataset path: {dataset_path}")

# Step 4: Check dataset structure
print("📂 Dataset structure:")
for root, dirs, files in os.walk(dataset_path):
    level = root.replace(dataset_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files[:5]:  # Show first 5 files
        print(f"{subindent}{file}")
    if len(files) > 5:
        print(f"{subindent}... and {len(files) - 5} more files")

# Step 5: Load and verify data.yaml file
data_yaml_path = f"{dataset_path}/data.yaml"
print(f"📄 Using data.yaml: {data_yaml_path}")

# Check if data.yaml exists and display its contents
if os.path.exists(data_yaml_path):
    with open(data_yaml_path, 'r') as f:
        yaml_content = yaml.safe_load(f)
    print("📋 data.yaml contents:")
    print(yaml.dump(yaml_content, default_flow_style=False))
    print("✅ data.yaml found and loaded successfully")
    print(f"📊 Number of classes: {yaml_content.get('nc', 'Unknown')}")
    print(f"📝 Class names: {yaml_content.get('names', 'Unknown')}")
else:
    print("❌ data.yaml not found at specified path")
    print("Please check if the file exists at:", data_yaml_path)
    exit()

# Step 6: Load YOLOv8 model (Fresh Training)
print("🔄 Loading fresh YOLOv8 model...")
model = YOLO('yolov8s.pt')  # Using YOLOv8s for better accuracy
print("✅ YOLOv8s model loaded for fresh training")

# Step 7: Fresh Training with 100 Epochs and Early Stopping
print("🚀 Starting fresh training with 100 epochs and early stopping...")

training_results = model.train(
    data=data_yaml_path,
    epochs=100,  # 100 epochs as requested
    imgsz=640,
    batch=16,
    project='/kaggle/working',
    name='object_detection',  # Changed name as requested
    
    # Early Stopping Configuration
    patience=15,  # Stop if no improvement for 15 epochs
    
    # Optimization parameters
    lr0=0.01,      # Initial learning rate
    lrf=0.01,      # Final learning rate
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    
    # Enhanced Data Augmentation
    hsv_h=0.015,    # Hue augmentation
    hsv_s=0.7,      # Saturation augmentation
    hsv_v=0.4,      # Value augmentation
    degrees=10.0,   # Rotation degrees
    translate=0.1,  # Translation
    scale=0.9,      # Scale augmentation
    shear=2.0,      # Shear augmentation
    perspective=0.0, # Perspective augmentation
    flipud=0.0,     # Vertical flip probability
    fliplr=0.5,     # Horizontal flip probability
    mosaic=1.0,     # Mosaic augmentation
    mixup=0.15,     # Mixup augmentation
    copy_paste=0.3, # Copy paste augmentation
    
    # Validation and saving settings
    val=True,
    save_period=10,  # Save checkpoint every 10 epochs
    plots=True,      # Generate training plots
    
    # Other optimization settings
    workers=8,
    seed=42,
    deterministic=True,
    single_cls=False,
    rect=False,
    cos_lr=True,     # Cosine learning rate scheduler
    close_mosaic=10, # Disable mosaic in last 10 epochs
    
    # Fresh training (no resume)
    resume=False,
    exist_ok=True   # Overwrite existing project
)

print("✅ Training completed!")

# Step 8: Load Best Model and Comprehensive Evaluation
best_model_path = '/kaggle/working/object_detection/weights/best.pt'
last_model_path = '/kaggle/working/object_detection/weights/last.pt'

print("📊 Loading best model and running comprehensive evaluation...")

if os.path.exists(best_model_path):
    print(f"📁 Loading best model: {best_model_path}")
    best_model = YOLO(best_model_path)
    
    # Comprehensive validation
    print("📈 Running detailed validation...")
    val_metrics = best_model.val(
        data=data_yaml_path, 
        plots=True,
        save_json=True,
        split='val'
    )
    
    # Display comprehensive metrics
    print("\n" + "="*60)
    print("🏆 COMPREHENSIVE PERFORMANCE METRICS")
    print("="*60)
    
    # Overall Performance
    print(f"🎯 mAP@0.5 (IoU=0.5):        {val_metrics.box.map50:.4f} ({val_metrics.box.map50*100:.2f}%)")
    print(f"🎯 mAP@0.5-0.95:             {val_metrics.box.map:.4f} ({val_metrics.box.map*100:.2f}%)")
    print(f"🎯 Precision (P):            {val_metrics.box.mp:.4f} ({val_metrics.box.mp*100:.2f}%)")
    print(f"🎯 Recall (R):               {val_metrics.box.mr:.4f} ({val_metrics.box.mr*100:.2f}%)")
    
    # Calculate F1 Score
    if val_metrics.box.mp > 0 and val_metrics.box.mr > 0:
        f1_score = 2 * (val_metrics.box.mp * val_metrics.box.mr) / (val_metrics.box.mp + val_metrics.box.mr)
        print(f"🎯 F1-Score:                 {f1_score:.4f} ({f1_score*100:.2f}%)")
    
    # Per-class metrics
    print("\n📋 Per-Class Performance (mAP@0.5):")
    class_names = yaml_content.get('names', [])
    if hasattr(val_metrics.box, 'maps') and val_metrics.box.maps is not None:
        for i, (name, map_val) in enumerate(zip(class_names, val_metrics.box.maps)):
            print(f"   📌 {name}: {map_val:.4f} ({map_val*100:.2f}%)")
    
    print("="*60)

else:
    print("❌ Best model not found. Training may have failed.")

# Step 9: Save Model with Custom Name
if os.path.exists(best_model_path):
    print("\n💾 Saving model with custom name 'object_detection'...")
    
    # Create custom model directory
    custom_model_dir = '/kaggle/working/object_detection_model'
    os.makedirs(custom_model_dir, exist_ok=True)
    
    # Copy and rename best model
    custom_model_path = '/kaggle/working/object_detection_model/object_detection.pt'
    shutil.copy2(best_model_path, custom_model_path)
    print(f"✅ Model saved as: {custom_model_path}")
    
    # Save PyTorch state dict
    best_model = YOLO(best_model_path)
    torch.save(best_model.model.state_dict(), '/kaggle/working/object_detection_model/object_detection.pth')
    print("✅ PyTorch state dict saved: object_detection.pth")
    
    # Export to ONNX format
    try:
        onnx_path = best_model.export(format='onnx', 
                                     imgsz=640, 
                                     dynamic=True,
                                     optimize=True)
        # Move ONNX to custom directory
        if os.path.exists(onnx_path):
            shutil.move(onnx_path, '/kaggle/working/object_detection_model/object_detection.onnx')
            print("✅ ONNX model exported: object_detection.onnx")
    except Exception as e:
        print(f"⚠ ONNX export failed: {e}")

# Step 10: Sample Testing and Predictions
print("\n🔍 Running sample testing and predictions...")

# Test paths
test_path = f"{dataset_path}/test/images"
val_path = f"{dataset_path}/val/images"

if os.path.exists(best_model_path):
    best_model = YOLO(best_model_path)
    
    # Test on validation set
    if os.path.exists(val_path):
        print("📋 Running predictions on validation set...")
        val_results = best_model.predict(
            source=val_path,
            save=True,
            conf=0.25,          # Confidence threshold
            iou=0.45,           # IoU threshold for NMS
            project='/kaggle/working',
            name='object_detection_val_predictions',
            save_txt=True,      # Save prediction labels
            save_conf=True,     # Save confidence scores
            augment=True,       # Test time augmentation
            agnostic_nms=False,
            max_det=300,        # Maximum detections per image
            device='0' if torch.cuda.is_available() else 'cpu'
        )
        print("✅ Validation predictions completed")
        
        # Analyze predictions
        total_detections = 0
        confidences = []
        class_counts = {}
        
        for result in val_results:
            if result.boxes is not None:
                total_detections += len(result.boxes)
                confidences.extend(result.boxes.conf.cpu().numpy())
                
                # Count detections per class
                classes = result.boxes.cls.cpu().numpy()
                for cls in classes:
                    class_name = class_names[int(cls)] if int(cls) < len(class_names) else f"Class_{int(cls)}"
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Display prediction analysis
        if confidences:
            print(f"\n📊 Validation Prediction Analysis:")
            print(f"   🔢 Total Images Processed: {len(val_results)}")
            print(f"   🎯 Total Detections: {total_detections}")
            print(f"   📈 Average Confidence: {sum(confidences)/len(confidences):.3f}")
            print(f"   📉 Min Confidence: {min(confidences):.3f}")
            print(f"   📈 Max Confidence: {max(confidences):.3f}")
            print(f"   🎯 Detections per Image: {total_detections/len(val_results):.2f}")
            
            print(f"\n📋 Detections by Class:")
            for class_name, count in sorted(class_counts.items()):
                print(f"   📌 {class_name}: {count} detections")
    
    # Test on test set if available
    if os.path.exists(test_path):
        print("\n📋 Running predictions on test set...")
        test_results = best_model.predict(
            source=test_path,
            save=True,
            conf=0.25,
            iou=0.45,
            project='/kaggle/working',
            name='object_detection_test_predictions',
            save_txt=True,
            save_conf=True,
            augment=True,
            max_det=300
        )
        print("✅ Test predictions completed")
        
        # Test set analysis
        test_detections = sum(len(result.boxes) if result.boxes is not None else 0 for result in test_results)
        print(f"   🔢 Test Images Processed: {len(test_results)}")
        print(f"   🎯 Test Detections: {test_detections}")
        print(f"   🎯 Test Detections per Image: {test_detections/len(test_results):.2f}")

# Step 11: Display Sample Results
print("\n🖼 Displaying sample prediction results...")

prediction_dirs = [
    '/kaggle/working/object_detection_val_predictions',
    '/kaggle/working/object_detection_test_predictions'
]

for pred_dir in prediction_dirs:
    if os.path.exists(pred_dir):
        predictions = [f for f in os.listdir(pred_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if predictions:
            print(f"\n📸 Sample from {os.path.basename(pred_dir)}:")
            # Show first 3 predictions
            for i, pred_file in enumerate(predictions[:3]):
                print(f"   🖼 Sample {i+1}: {pred_file}")
                display(Image(filename=os.path.join(pred_dir, pred_file)))
            break

# Step 12: Training Summary and File Locations
print("\n" + "🎉" + "="*58 + "🎉")
print("             OBJECT DETECTION TRAINING COMPLETED")
print("🎉" + "="*58 + "🎉")

# Display training results location
results_dir = '/kaggle/working/object_detection'
if os.path.exists(results_dir):
    print(f"\n📁 Training results directory: {results_dir}")
    
    # List important training files
    important_files = [
        'results.png', 'confusion_matrix.png', 'F1_curve.png',
        'P_curve.png', 'PR_curve.png', 'R_curve.png',
        'val_batch0_labels.jpg', 'val_batch0_pred.jpg',
        'train_batch0.jpg', 'labels.jpg', 'labels_correlogram.jpg'
    ]
    
    print("\n📋 Generated Training Files:")
    for file in important_files:
        file_path = os.path.join(results_dir, file)
        if os.path.exists(file_path):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (not generated)")

# Final Summary
print(f"\n🎯 Final Model Performance:")
if 'val_metrics' in locals():
    print(f"   📊 mAP@0.5: {val_metrics.box.map50:.1%}")
    print(f"   📊 mAP@0.5-0.95: {val_metrics.box.map:.1%}")
    print(f"   📊 Precision: {val_metrics.box.mp:.1%}")
    print(f"   📊 Recall: {val_metrics.box.mr:.1%}")

print(f"\n💾 Saved Model Files:")
print(f"   🏆 Best Model: /kaggle/working/object_detection_model/object_detection.pt")
print(f"   🔧 PyTorch Dict: /kaggle/working/object_detection_model/object_detection.pth")
if os.path.exists('/kaggle/working/object_detection_model/object_detection.onnx'):
    print(f"   🔄 ONNX Model: /kaggle/working/object_detection_model/object_detection.onnx")

print(f"\n🔍 Prediction Results:")
print(f"   📂 Validation: /kaggle/working/object_detection_val_predictions/")
if os.path.exists(test_path):
    print(f"   📂 Test: /kaggle/working/object_detection_test_predictions/")

print(f"\n📊 Training Charts: /kaggle/working/object_detection/")
print("\n✨ Fresh training with 100 epochs completed successfully! ✨")
print("🎯 Model saved as 'object_detection' with comprehensive metrics and sample testing!")
