import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO('/Users/sweetjichu/Documents/faris-lab/Roots/runs/detect/train3/weights/best.onnx')
print("Model loaded successfully")
print("Model classes:", model.names)

# Test on a frame from your video
cap = cv2.VideoCapture('wa.mp4')
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f"Frame shape: {frame.shape}")

        # Test with very low confidence threshold
        results = model(frame, conf=0.1, iou=0.3, verbose=True)
        print(f"Number of detections: {len(results[0].boxes)}")

        if len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                cls = int(box.cls)
                conf = float(box.conf)
                print(f"Detection {i}: Class={model.names[cls]}, Confidence={conf:.3f}")
        else:
            print("No detections found even with low confidence threshold")

            # Try with the PyTorch model
            print("\nTrying with PyTorch model...")
            model_pt = YOLO('/Users/sweetjichu/Documents/faris-lab/Roots/runs/detect/train3/weights/best.pt')
            results_pt = model_pt(frame, conf=0.1, iou=0.3, verbose=True)
            print(f"PyTorch detections: {len(results_pt[0].boxes)}")

            if len(results_pt[0].boxes) > 0:
                for i, box in enumerate(results_pt[0].boxes):
                    cls = int(box.cls)
                    conf = float(box.conf)
                    print(f"PyTorch Detection {i}: Class={model_pt.names[cls]}, Confidence={conf:.3f}")

    cap.release()
else:
    print("Could not open video file")