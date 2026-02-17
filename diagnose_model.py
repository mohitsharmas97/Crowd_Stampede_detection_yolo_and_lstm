"""
Diagnostic script to check what YOLOv8 model is actually detecting
This will help identify if the model is trained correctly
"""

import cv2
from ultralytics import YOLO
import numpy as np

# Load model
print("Loading YOLOv8 model...")
model = YOLO("best.pt")

# Load test video
video_path = r"c:\Users\Mohit Sharma\Downloads\853892-hd_1920_1080_25fps.mp4"
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video!")
    exit()

print(f"\nFrame shape: {frame.shape}")

# Run detection
print("\nRunning detection with confidence 0.25...")
results = model(frame, conf=0.25, verbose=True)

# Check what was detected
if len(results) > 0 and results[0].boxes is not None:
    boxes = results[0].boxes
    print(f"\nTotal detections: {len(boxes)}")
    
    if len(boxes) > 0:
        # Get all unique classes detected
        cls_ids = boxes.cls.cpu().numpy()
        unique_classes = np.unique(cls_ids)
        
        print(f"\nUnique class IDs detected: {unique_classes}")
        print(f"\nClass distribution:")
        for cls_id in unique_classes:
            count = np.sum(cls_ids == cls_id)
            conf_avg = boxes.conf[cls_ids == cls_id].mean()
            print(f"  Class {int(cls_id)}: {count} detections (avg conf: {conf_avg:.3f})")
        
        # Check model names
        print(f"\nModel class names:")
        if hasattr(model, 'names'):
            print(f"  {model.names}")
        
        # Show first few detections
        print(f"\nFirst 5 detections:")
        for i, box in enumerate(boxes[:5]):
            cls_id = int(box.cls.cpu().numpy()[0])
            conf = float(box.conf.cpu().numpy()[0])
            class_name = model.names.get(cls_id, f"class_{cls_id}") if hasattr(model, 'names') else f"class_{cls_id}"
            print(f"  {i+1}. Class: {class_name} (ID: {cls_id}), Confidence: {conf:.3f}")
    else:
        print("\nNo detections found!")
else:
    print("\nNo detections found!")

cap.release()

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
if len(results) > 0 and results[0].boxes is not None and len(boxes) > 0:
    cls_ids = boxes.cls.cpu().numpy()
    if 0 in cls_ids:
        print("✅ Model IS detecting class 0 (person)")
    else:
        print("❌ Model is NOT detecting class 0")
        print(f"   Detected classes: {unique_classes}")
        print("\nPOSSIBLE ISSUE:")
        print("   Your model was trained on a custom dataset")
        print("   that doesn't use class 0 for 'person'")
        print("\nSOLUTION:")
        print("   Modify app.py to accept ALL detections,")
        print("   not just class 0")
else:
    print("❌ Model detected NOTHING")
    print("\nPOSSIBLE ISSUES:")
    print("   1. Model not trained properly")
    print("   2. Confidence threshold too high")
    print("   3. Video content doesn't match training data")
