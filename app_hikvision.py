import numpy as np
import supervision as sv
import cv2
import argparse
import os
import onnxruntime as ort

# Force CPU usage for ONNX Runtime
ort.set_default_logger_severity(3)  # Disable logging warnings
providers = ['CPUExecutionProvider']

# Configuration
MODEL_PATH = 'model_ir8_final.onnx'

# Parse command line arguments
parser = argparse.ArgumentParser(description='Crack and Pothole Detection for Hikvision (CPU Only)')
parser.add_argument('--source', type=str, default='camera',
                    help='Source: "camera" for live camera, "video" for video file')
parser.add_argument('--video-path', type=str, default='waki.mp4',
                    help='Path to video file when source is "video"')
parser.add_argument('--camera-id', type=int, default=0,
                    help='Camera ID (default: 0)')
parser.add_argument('--output', type=str, default='output_annotated_video_hikvision.mp4',
                    help='Output video file path')
parser.add_argument('--save-video', action='store_true',
                    help='Save processed video to file')
parser.add_argument('--confidence', type=float, default=0.4,
                    help='Confidence threshold (default: 0.4)')
args = parser.parse_args()

# Set source based on arguments
USE_CAMERA = args.source == 'camera'
SOURCE_VIDEO_PATH = args.video_path
TARGET_VIDEO_PATH = args.output
CAMERA_ID = args.camera_id
CONFIDENCE_THRESHOLD = args.confidence

print(f"Initializing ONNX Runtime with CPU providers: {providers}")

class HikvisionYOLO:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=providers)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Class names (based on your dataset)
        self.names = {0: 'crack', 1: 'pothole'}

        print(f"Model loaded successfully: {model_path}")
        print(f"Input name: {self.input_name}")
        print(f"Output names: {self.output_names}")

    def __call__(self, frame):
        # Preprocess frame
        original_shape = frame.shape[:2]  # (height, width)

        # Resize to 640x640
        resized = cv2.resize(frame, (640, 640))

        # Convert to RGB and normalize
        input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Process outputs (YOLOv12 format)
        output = outputs[0]  # Shape: (1, 6, 8400) where 6 = [x, y, w, h, conf, class]

        # Convert to detections
        detections = sv.Detections.empty()

        if output.shape[0] > 0:
            # Filter by confidence
            mask = output[0, 4, :] > CONFIDENCE_THRESHOLD
            filtered_output = output[0, :, mask]

            if filtered_output.shape[1] > 0:
                # Extract boxes and convert to xyxy format
                boxes = filtered_output[:4, :].T  # (N, 4) in xywh format
                confidences = filtered_output[4, :]
                class_ids = filtered_output[5, :].astype(int)

                # Convert from normalized 640x640 to original image coordinates
                scale_x = original_shape[1] / 640
                scale_y = original_shape[0] / 640

                # Convert xywh to xyxy
                xyxy_boxes = np.zeros_like(boxes)
                xyxy_boxes[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x  # x1
                xyxy_boxes[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y  # y1
                xyxy_boxes[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x  # x2
                xyxy_boxes[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y  # y2

                # Create detections
                detections = sv.Detections(
                    xyxy=xyxy_boxes,
                    confidence=confidences,
                    class_id=class_ids
                )

        return detections

# Initialize video source and get video info
if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_ID}")
        exit()

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback for camera

    video_info = sv.VideoInfo(width=width, height=height, fps=fps, total_frames=None)
    print(f"Camera {CAMERA_ID} opened: {width}x{height} @ {fps} FPS")
else:
    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

line_y_position = int(video_info.height * 0.55)  # 55% from top

# Create horizontal line zone
LINE_START = sv.Point(0, line_y_position)
LINE_END = sv.Point(video_info.width, line_y_position)

# Load YOLO model (CPU version)
model = HikvisionYOLO(MODEL_PATH)

# create BYTETracker instance for better object tracking
byte_tracker = sv.ByteTrack()

# LineZone untuk visualisasi garis (tanpa counter)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Detection tracking variables
pothole_count = 0
crack_count = 0
detection_log = []
processed_trackers = set()

def get_timestamp(frame_index, fps):
    """Convert frame index to timestamp in MM:SS format"""
    if frame_index is None or fps == 0:
        return "00:00"
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def filter_detections(detections):
    """Filter detections to only include crack and pothole classes"""
    filtered_indices = []
    for i, class_id in enumerate(detections.class_id):
        class_name = model.names[class_id].lower()
        if class_name in ['crack', 'pothole']:
            filtered_indices.append(i)

    if filtered_indices:
        return detections[filtered_indices]
    else:
        return sv.Detections.empty()

def process_frame(frame: np.ndarray, frame_index: int = None) -> np.ndarray:
    """Process a single frame for object detection and annotation"""

    # Run YOLOv12n inference
    results = model(frame)

    # Filter detections to only include cracks and potholes
    filtered_results = filter_detections(results)

    # Update tracker with filtered detections
    tracked_detections = byte_tracker.update_with_detections(filtered_results)

    # Annotate frame
    annotated_frame = frame.copy()

    # Draw detection line
    cv2.line(annotated_frame, (LINE_START.x, LINE_START.y), (LINE_END.x, LINE_END.y), (0, 255, 255), 2)

    if len(tracked_detections) > 0:
        # Generate labels
        labels = []
        for i, (class_id, confidence) in enumerate(zip(tracked_detections.class_id, tracked_detections.confidence)):
            class_name = model.names[class_id]
            labels.append(f"{class_name}: {confidence:.2f}")

            # Count unique detections crossing the line
            tracker_id = tracked_detections.tracker_id[i]
            if tracker_id not in processed_trackers:
                # Check if detection crosses the line
                box = tracked_detections.xyxy[i]
                box_center_y = (box[1] + box[3]) / 2

                if box_center_y > line_y_position:
                    if class_name.lower() == 'pothole':
                        pothole_count += 1
                    elif class_name.lower() == 'crack':
                        crack_count += 1

                    # Log detection with timestamp
                    timestamp = get_timestamp(frame_index, video_info.fps)
                    detection_log.append({
                        'timestamp': timestamp,
                        'class': class_name,
                        'confidence': confidence,
                        'frame': frame_index
                    })

                    processed_trackers.add(tracker_id)

        # Draw boxes and labels
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_detections)

    # Add statistics overlay
    stats_text = [
        f"Cracks: {crack_count}",
        f"Potholes: {pothole_count}",
        f"Total: {crack_count + pothole_count}"
    ]

    y_offset = 30
    for text in stats_text:
        cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25

    return annotated_frame

# Main processing loop
if USE_CAMERA:
    print("Starting camera processing...")
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Process frame
            processed_frame = process_frame(frame, frame_count)

            # Display frame
            cv2.imshow('Crack and Pothole Detection', processed_frame)

            # Save frame if requested
            if args.save_video and frame_count == 0:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info.fps,
                                    (processed_frame.shape[1], processed_frame.shape[0]))

            if args.save_video and 'out' in locals():
                out.write(processed_frame)

            # Check for 'q' key to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nProcessing stopped by user")

    finally:
        cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()

else:
    print(f"Processing video file: {SOURCE_VIDEO_PATH}")

    # Create video writer
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info.fps,
                            (video_info.width, video_info.height))

    # Process video file
    with sv.VideoSink(target_path=TARGET_VIDEO_PATH, video_info=video_info) as sink:
        for frame_index, frame in enumerate(sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)):
            processed_frame = process_frame(frame, frame_index)

            if args.save_video:
                sink.write_frame(processed_frame)

            # Display progress
            if frame_index % 30 == 0:
                print(f"Processed frame {frame_index}")

# Save detection report
report_path = 'detection_report_hikvision.txt'
with open(report_path, 'w') as f:
    f.write("Detection Report - Hikvision CPU Version\n")
    f.write("========================================\n\n")
    f.write(f"Total Cracks Detected: {crack_count}\n")
    f.write(f"Total Potholes Detected: {pothole_count}\n")
    f.write(f"Total Detections: {crack_count + pothole_count}\n\n")
    f.write("Detection Details:\n")
    f.write("Timestamp\t\tClass\t\tConfidence\n")
    f.write("--------\t\t-----\t\t----------\n")

    for detection in detection_log:
        f.write(f"{detection['timestamp']}\t\t{detection['class']}\t\t{detection['confidence']:.2f}\n")

print(f"\nProcessing completed!")
print(f"Detection report saved to: {report_path}")
print(f"Output video saved to: {TARGET_VIDEO_PATH}")
print(f"Cracks detected: {crack_count}")
print(f"Potholes detected: {pothole_count}")