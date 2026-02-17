"""
Enhanced diagnostic to check detection at different confidence levels
"""

import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("best.pt")
video_path = r"c:\Users\Mohit Sharma\Downloads\853892-hd_1920_1080_25fps.mp4"

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()

if not ret:
    print("Failed to read video!")
    exit()

print(f"Frame shape: {frame.shape}\n")

# Test different confidence levels
for conf in [0.1, 0.25, 0.5, 0.7]:
    print(f"\n{'='*60}")
    print(f"Testing with confidence threshold: {conf}")
    print('='*60)
    
    results = model(frame, conf=conf, verbose=False)
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            cls_ids = boxes.cls.cpu().numpy()
            
            # Check for class 0 (person)
            person_mask = cls_ids == 0
            person_count = np.sum(person_mask)
            
            print(f"Total detections: {len(boxes)}")
            print(f"Class 0 (person) detections: {person_count}")
            
            if person_count > 0:
                person_confs = boxes.conf.cpu().numpy()[person_mask]
                print(f"Person confidences: min={person_confs.min():.3f}, "
                      f"max={person_confs.max():.3f}, avg={person_confs.mean():.3f}")
                
                # Show sample detections
                person_boxes = boxes.xyxy.cpu().numpy()[person_mask]
                print(f"\nSample person detections (first 3):")
                for i in range(min(3, len(person_boxes))):
                    x1, y1, x2, y2 = person_boxes[i]
                    conf = person_confs[i]
                    w, h = x2-x1, y2-y1
                    print(f"  Person {i+1}: conf={conf:.3f}, box=({int(x1)},{int(y1)},{int(x2)},{int(y2)}), size={int(w)}x{int(h)}")
            else:
                unique_classes = np.unique(cls_ids)
                print(f"Detected classes (not person): {unique_classes}")
        else:
            print("No detections at all")
    else:
        print("No detections at all")

cap.release()

print(f"\n\n{'='*60}")
print("RECOMMENDATION:")
print('='*60)
print("Based on the results above:")
print("1. If people detected at low conf (0.1-0.3): Lower confidence threshold in app")
print("2. If no people detected at any conf: Model needs retraining")
print("3. If detecting other classes: Model using wrong class mapping")
