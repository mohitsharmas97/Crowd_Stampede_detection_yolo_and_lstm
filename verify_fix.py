"""
Quick test to verify the detection fix is working
Run this after restarting Streamlit
"""

import cv2
from ultralytics import YOLO
import numpy as np

print("=" * 70)
print(" TESTING DETECTION FIX - Quick Verification")
print("=" * 70)

# Load model
model = YOLO("best.pt")
video_path = r"c:\Users\Mohit Sharma\Downloads\853892-hd_1920_1080_25fps.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("❌ Failed to read video")
    exit(1)

print(f"\n📹 Video loaded: {frame.shape}")
print(f"📦 Model loaded: best.pt")
print(f"📋 Model classes: {model.names}")

# Test at confidence 0.25 (new default)
print(f"\n🔍 Testing detection at confidence 0.25...")
results = model(frame, conf=0.25, verbose=False)

if len(results) > 0 and results[0].boxes is not None:
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        cls_ids = boxes.cls.cpu().numpy()
        
        # Apply new logic (same as in app.py)
        has_person_class = False
        person_mask = None
        
        if hasattr(model, 'names'):
            if 0 in model.names and 'person' in str(model.names[0]).lower():
                person_mask = cls_ids == 0
                has_person_class = person_mask.any()
            else:
                for cls_id, cls_name in model.names.items():
                    if 'person' in str(cls_name).lower() or 'people' in str(cls_name).lower() \
                       or 'human' in str(cls_name).lower() or 'crowd' in str(cls_name).lower():
                        person_mask = cls_ids == cls_id
                        has_person_class = person_mask.any()
                        if has_person_class:
                            break
        
        if not has_person_class or person_mask is None:
            person_mask = np.ones(len(cls_ids), dtype=bool)
        
        person_count = np.sum(person_mask)
        
        print(f"\n✅ SUCCESS!")
        print(f"   Total detections: {len(boxes)}")
        print(f"   Person detections: {person_count}")
        
        if person_count > 0:
            person_confs = boxes.conf.cpu().numpy()[person_mask]
            print(f"   Confidence range: {person_confs.min():.3f} - {person_confs.max():.3f}")
            print(f"   Average confidence: {person_confs.mean():.3f}")
            
            print(f"\n🎯 FIX WORKING! App should now show People Count: {person_count}")
        else:
            print(f"\n⚠️  Still no people detected")
            print(f"   Detected classes: {np.unique(cls_ids)}")
            print(f"   Try lowering confidence further in the app")
    else:
        print("\n❌ No detections at confidence 0.25")
        print("   Try lowering confidence to 0.1 in the app")
else:
    print("\n❌ No detections at all")
    print("   Model may need retraining")

cap.release()

print(f"\n" + "=" * 70)
print("Next step: Restart Streamlit and test with the video!")
print("=" * 70)
