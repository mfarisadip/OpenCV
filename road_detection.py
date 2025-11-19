import cv2
import math
import cvzone
import numpy as np
from sort import Sort
from ultralytics import YOLO
from datetime import datetime
import time

# Initialize video capture
video_path = "pole.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model (using same path as app.py)
yolo_model = YOLO("runs/detect/train3/weights/best.onnx")

# Define class names (will be updated from model names to avoid mismatch)
class_labels = []  # Will be populated from model.names

# Initialize tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define line limits for detection zone (similar to app.py)
count_line = [0, 0, 0, 0]  # Will be set based on video dimensions

# Detection tracking variables
pothole_count = 0
crack_count = 0
detection_log = []
processed_trackers = set()
frame_count = 0

def get_timestamp(frame_index, fps):
    """Convert frame index to timestamp in MM:SS format"""
    if frame_index is None or fps == 0:
        return "00:00"
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Populate class labels from model to ensure correct mapping
class_labels = list(yolo_model.names.values())
print(f"Model classes loaded: {class_labels}")

# Get video properties and set detection line
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default fallback

# Set detection line at 55% from top (like app.py)
line_y_position = int(height * 0.55)
count_line = [0, line_y_position, width, line_y_position]

print(f"Processing video: {video_path}")
print(f"Resolution: {width}x{height}, FPS: {fps}")
print(f"Detection line at Y: {line_y_position} (55% from top)")
print(f"Model: yolov5s6.onnx")
print("-" * 60)

start_processing_time = time.time()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    timestamp = get_timestamp(frame_count, fps)

    # Perform object detection
    detection_results = yolo_model(frame, stream=True, conf=0.4, iou=0.3, verbose=False)

    # Collect detections for crack and pothole only
    detection_array = np.empty((0, 5))
    valid_detections = []  # Store valid detections with class info

    for result in detection_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2 - x1, y2 - y1
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id].lower()  # Convert to lowercase and use model.names directly

            # Filter for crack and pothole only (exactly like app.py)
            if class_name in ['crack', 'pothole']:
                detection_entry = np.array([x1, y1, x2, y2, confidence])
                detection_array = np.vstack((detection_array, detection_entry))
                valid_detections.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })

    # Update tracker
    tracked_objects = tracker.update(detection_array)

    # Draw detection line
    cv2.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (255, 255, 0), 2)

    for i, obj in enumerate(tracked_objects):
        x1, y1, x2, y2, obj_id = map(int, obj)
        width, height = x2 - x1, y2 - y1

        # Find the corresponding detection info by matching bbox
        class_name = "unknown"
        confidence = 0.0

        for detection in valid_detections:
            det_x1, det_y1, det_x2, det_y2 = detection['bbox']
            # Calculate IoU (Intersection over Union) for better matching
            # Calculate intersection
            x_left = max(x1, det_x1)
            y_top = max(y1, det_y1)
            x_right = min(x2, det_x2)
            y_bottom = min(y2, det_y2)

            if x_right > x_left and y_bottom > y_top:
                intersection_area = (x_right - x_left) * (y_bottom - y_top)

                # Calculate union
                box1_area = (x2 - x1) * (y2 - y1)
                box2_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                union_area = box1_area + box2_area - intersection_area

                # Calculate IoU
                iou = intersection_area / union_area

                # If IoU is high enough, consider it a match
                if iou > 0.7:  # 70% IoU threshold
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    break

        # Debug: print if still unknown (this will help identify the issue)
        if class_name == "unknown" and len(valid_detections) > 0:
            print(f"Warning: Could not match tracked object {obj_id} to any detection")
            print(f"Tracked bbox: [{x1}, {y1}, {x2}, {y2}]")
            print(f"Available detections: {len(valid_detections)}")
            for j, det in enumerate(valid_detections):
                det_x1, det_y1, det_x2, det_y2 = det['bbox']
                # Calculate IoU for debug
                x_left = max(x1, det_x1)
                y_top = max(y1, det_y1)
                x_right = min(x2, det_x2)
                y_bottom = min(y2, det_y2)
                if x_right > x_left and y_bottom > y_top:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                    box1_area = (x2 - x1) * (y2 - y1)
                    box2_area = (det_x2 - det_x1) * (det_y2 - det_y1)
                    union_area = box1_area + box2_area - intersection_area
                    iou = intersection_area / union_area
                    print(f"  Detection {j}: {det['bbox']} - {det['class_name']} (IoU: {iou:.2f})")
                else:
                    print(f"  Detection {j}: {det['bbox']} - {det['class_name']} (No overlap)")

        # Set colors based on class (consistent with app.py)
        class_name_lower = class_name.lower()
        if class_name_lower == "pothole":
            color = (0, 255, 255)  # Yellow for pothole
        elif class_name_lower == "crack":
            color = (255, 0, 255)  # Magenta for crack
        else:
            color = (255, 255, 0)  # Cyan for other/unknown classes

        # Draw bounding boxes and labels
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f'#{obj_id} {class_name} {confidence:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Calculate center of the box
        center_x, center_y = x1 + width // 2, y1 + height // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # Check if object crosses the detection line (exactly like app.py logic)
        if obj_id not in processed_trackers and center_y >= line_y_position:
            if class_name_lower == 'pothole':
                pothole_count += 1
                detection_log.append({
                    'type': 'pothole',
                    'tracker_id': obj_id,
                    'timestamp': timestamp,
                    'frame': frame_count,
                    'position': center_y,
                    'confidence': confidence,
                    'detection_type': 'detected_below_line'
                })
                print(f"🕳️ POTHOLE #{pothole_count} (ID:#{obj_id}) DETECTED at {timestamp} (Y: {center_y}, Conf: {confidence:.2f})")

            elif class_name_lower == 'crack':
                crack_count += 1
                detection_log.append({
                    'type': 'crack',
                    'tracker_id': obj_id,
                    'timestamp': timestamp,
                    'frame': frame_count,
                    'position': center_y,
                    'confidence': confidence,
                    'detection_type': 'detected_below_line'
                })
                print(f"〰️ CRACK #{crack_count} (ID:#{obj_id}) DETECTED at {timestamp} (Y: {center_y}, Conf: {confidence:.2f})")

            processed_trackers.add(obj_id)
            # Flash detection line
            cv2.line(frame, (count_line[0], count_line[1]), (count_line[2], count_line[3]), (0, 255, 0), 4)

    # Display statistics (similar to app.py)
    cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f'Potholes: {pothole_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Cracks: {crack_count}', (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f'Time: {timestamp}', (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display total detections
    # total_detections = pothole_count + crack_count
    # cvzone.putTextRect(frame, f'TOTAL: {total_detections}', (20, height - 30),
    #                   scale=1, thickness=2, colorT=(255, 255, 255), colorR=(255, 0, 0),
    #                   font=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.imshow("Crack and Pothole Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup and final summary
processing_duration = time.time() - start_processing_time
cap.release()
cv2.destroyAllWindows()

# Print final summary (similar to app.py)
print("\n" + "="*60)
print("🔍 FINAL DETECTION SUMMARY")
print("="*60)
print(f"📹 Source: {video_path}")
print(f"🕳️  Total Potholes Detected: {pothole_count}")
print(f"〰️  Total Cracks Detected: {crack_count}")
print(f"📊 Total Detections: {pothole_count + crack_count}")
print(f"⏱️  Processing Time: {processing_duration:.2f} seconds")
print(f"🎬 Total Frames Processed: {frame_count}")
print(f"🚀 Processing Speed: {frame_count/processing_duration:.2f} FPS")

if detection_log:
    print("\n📋 Detailed Detection Log:")
    print("-" * 60)
    pothole_logs = [log for log in detection_log if log['type'] == 'pothole']
    crack_logs = [log for log in detection_log if log['type'] == 'crack']

    if pothole_logs:
        print(f"\n🕳️  Potholes ({len(pothole_logs)}):")
        for i, log in enumerate(pothole_logs, 1):
            detection_type = log.get('detection_type', 'unknown')
            print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

    if crack_logs:
        print(f"\n〰️  Cracks ({len(crack_logs)}):")
        for i, log in enumerate(crack_logs, 1):
            detection_type = log.get('detection_type', 'unknown')
            print(f"   {i}. Tracker ID #{log['tracker_id']} at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Conf: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]")

print("="*60)

# Save detection report to file
report_content = f"""
CRACK AND POTHOLE DETECTION REPORT
Source: {video_path}
Model: yolov5s6.onnx
Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Line Position: Y={line_y_position} (55% from top)

SUMMARY:
- Total Potholes Detected: {pothole_count}
- Total Cracks Detected: {crack_count}
- Total Detections: {pothole_count + crack_count}
- Processing Time: {processing_duration:.2f} seconds
- Processing Speed: {frame_count/processing_duration:.2f} FPS

DETAILED LOG:
"""

if detection_log:
    for log in detection_log:
        detection_type = log.get('detection_type', 'unknown')
        report_content += f"- {log['type'].upper()} (Tracker #{log['tracker_id']}) at {log['timestamp']} (Frame {log['frame']}, Y: {log['position']}, Confidence: {log['confidence']:.2f}) [{detection_type.replace('_', ' ').title()}]\n"

with open("detection_report_road_detection.txt", "w") as f:
    f.write(report_content)

print(f"📄 Detection report saved to: detection_report_road_detection.txt")