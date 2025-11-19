import numpy as np
import cv2
import argparse
from datetime import datetime
import time
import os
import onnxruntime as ort

# Configuration
MODEL_PATH = 'best_ir8.onnx'  # Path to the optimized ONNX model

# Parse command line arguments
parser = argparse.ArgumentParser(description='Crack and Pothole Detection with Camera Input')
parser.add_argument('--source', type=str, default='camera',
                    help='Source: "camera" for live camera, "video" for video file')
parser.add_argument('--video-path', type=str, default='pole.mp4',
                    help='Path to video file when source is "video"')
parser.add_argument('--camera-id', type=int, default=0,
                    help='Camera ID (default: 0)')
parser.add_argument('--output', type=str, default='output_annotated_video_optimized.mp4',
                    help='Output video file path')
parser.add_argument('--save-video', action='store_true',
                    help='Save processed video to file')
parser.add_argument('--device', type=str, default='cpu',
                    help='Device to use: "cpu" or "cuda"')
args = parser.parse_args()

# Force CPU for Python 3.6 compatibility
USE_CUDA = False
DEVICE = 'cpu'

# Set source based on arguments
USE_CAMERA = args.source == 'camera'
SOURCE_VIDEO_PATH = args.video_path
TARGET_VIDEO_PATH = args.output
CAMERA_ID = args.camera_id

# Initialize video source and get video info
if USE_CAMERA:
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print("Error: Cannot open camera {}".format(CAMERA_ID))
        exit()

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # Default fallback for camera

    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': None
    }
    print("Camera {} opened: {}x{} @ {} FPS".format(CAMERA_ID, width, height, fps))
else:
    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        print("Error: Cannot open video file: {}".format(SOURCE_VIDEO_PATH))
        exit()
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    video_info = {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total_frames
    }
    cap.release()

line_y_position = int(video_info['height'] * 0.55)  # 55% from top

# Load ONNX model
try:
    # Set providers for ONNX Runtime
    providers = ['CPUExecutionProvider']  # Force CPU for Python 3.6 compatibility
    session = ort.InferenceSession(MODEL_PATH, providers=providers)
    print("ONNX model loaded successfully on CPU")
except Exception as e:
    print("Error loading ONNX model: {}".format(e))
    exit()

# Get input and output info
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
output_names = [output.name for output in session.get_outputs()]

print(f"Model input shape: {input_shape}")
print(f"Model output shape will be processed")

# Class names
CLASS_NAMES = ['crack', 'pothole']

# Detection tracking variables
pothole_count = 0
crack_count = 0
detection_log = []
processed_trackers = set()

# Simple centroid tracker implementation
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return []

        # Compute centroids for current rects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If we are currently not tracking any objects, register each of them
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
            return []

        # Compute the distance between each pair of object centroids and input centroids
        objectIDs = list(self.objects.keys())
        objectCentroids = np.array(list(self.objects.values()))
        D = np.linalg.norm(objectCentroids[:, np.newaxis] - inputCentroids, axis=2)

        # Find the smallest distances between each pair of centroids
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        # Keep track of which centroids have already been matched
        usedRows = set()
        usedCols = set()

        # Loop over the combination of the (row, column) index tuples
        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = inputCentroids[col]
            self.disappeared[objectID] = 0
            usedRows.add(row)
            usedCols.add(col)

        # Compute both the row and column index we have NOT yet used
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        unusedCols = set(range(D.shape[1])).difference(usedCols)

        # If the number of object centroids is greater than the number of input centroids, check for disappeared objects
        if D.shape[0] >= D.shape[1]:
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
        else:
            for col in unusedCols:
                self.register(inputCentroids[col])

        # Return the list of tracked objects with their IDs
        tracked_objects = []
        for objectID, centroid in self.objects.items():
            tracked_objects.append((objectID, centroid))

        return tracked_objects

# Create centroid tracker instance
centroid_tracker = CentroidTracker()

def get_timestamp(frame_index, fps):
    """Convert frame index to timestamp in MM:SS format"""
    if frame_index is None or fps == 0:
        return "00:00"
    seconds = frame_index / fps
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}".format(minutes, seconds)

def preprocess_image_yolo(image, input_size=640):
    """Preprocess image for YOLO model with letterboxing (maintain aspect ratio)"""
    # Get original dimensions
    h, w = image.shape[:2]

    # Calculate new dimensions maintaining aspect ratio
    scale = min(input_size / w, input_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h))

    # Create letterbox (black padding)
    letterbox = np.full((input_size, input_size, 3), 114, dtype=np.uint8)  # Gray padding
    y_offset = (input_size - new_h) // 2
    x_offset = (input_size - new_w) // 2

    # Place resized image on letterbox
    letterbox[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Convert BGR to RGB
    letterbox = cv2.cvtColor(letterbox, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    letterbox = letterbox.astype(np.float32) / 255.0

    # Add batch dimension: HWC -> BCHW
    input_tensor = np.transpose(letterbox, (2, 0, 1))  # CHW
    input_tensor = np.expand_dims(input_tensor, axis=0)  # BCHW

    return input_tensor, scale, x_offset, y_offset

def apply_nms(boxes, scores, iou_threshold=0.5):
    """Apply Non-Maximum Suppression to remove duplicate detections"""
    if len(boxes) == 0:
        return []

    # Convert boxes to [x1, y1, x2, y2] format if needed
    boxes = boxes.astype(float)

    # Calculate areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    # Sort by confidence score
    indices = np.argsort(scores)[::-1]

    keep = []
    while len(indices) > 0:
        # Pick the box with highest confidence
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        remaining_indices = indices[1:]
        current_box = boxes[current]
        remaining_boxes = boxes[remaining_indices]

        # Calculate intersection
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # Calculate union
        union = areas[current] + areas[remaining_indices] - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)

        # Keep boxes with IoU less than threshold
        indices = remaining_indices[iou < iou_threshold]

    return keep

def run_yolo_onnx_inference(frame, conf_threshold=0.002, input_size=640):  # Standard YOLO input size
    """Run YOLO ONNX inference manually"""
    try:
        # Preprocess image
        input_tensor, scale, x_offset, y_offset = preprocess_image_yolo(frame, input_size)

        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})

        # Process output (YOLO format)
        output = outputs[0]  # Model output

        # Handle different output formats
        if len(output.shape) == 3:
            if output.shape[1] == 6:  # Format: [1, 6, detections]
                output = output.transpose(0, 2, 1)[0]  # [detections, 6]
            else:
                output = output[0]  # Remove batch dimension

        # If output is in [features, detections] format, transpose it
        if output.shape[0] == 6 and output.shape[1] > 6:  # [6, detections]
            output = output.T  # [detections, 6]

        # Filter detections - try different interpretations
        detections = []
        for detection in output:
            if len(detection) >= 6:
                # Try different interpretations of the output format
                # Option 1: Normalized coordinates [0,1]
                x1, y1, x2, y2 = detection[:4]
                score1 = detection[4]  # Could be confidence or class score
                score2 = detection[5]  # Could be class score or confidence

                # Check if coordinates are normalized (0-1 range) or absolute pixels
                coord_max = max(abs(x1), abs(y1), abs(x2), abs(y2))

                # Try different confidence calculations
                confidence1 = abs(float(score1))  # Use absolute value
                confidence2 = abs(float(score2))
                confidence3 = abs(float(score1 * score2))  # Combined

                # Use the best confidence
                confidence = max(confidence1, confidence2, confidence3)

                if confidence > conf_threshold:
                    # Simple class determination
                    class_id = 0 if abs(score1) > abs(score2) else 1  # 0=crack, 1=pothole

                    # Scale coordinates back to original frame size
                    h, w = frame.shape[:2]
                    if coord_max <= 1.0:  # Normalized coordinates
                        # Scale to input size first
                        x1 = int(x1 * input_size)
                        y1 = int(y1 * input_size)
                        x2 = int(x2 * input_size)
                        y2 = int(y2 * input_size)
                    else:  # Already in pixel coordinates (input size)
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)

                    # Adjust for letterbox padding (remove offset and scale back to original)
                    x1 = int((x1 - x_offset) / scale)
                    y1 = int((y1 - y_offset) / scale)
                    x2 = int((x2 - x_offset) / scale)
                    y2 = int((y2 - y_offset) / scale)

                    # Clamp to image boundaries
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))

                    # Ensure valid box
                    if x2 > x1 and y2 > y1:
                        detections.append([x1, y1, x2, y2, confidence, class_id])

        if not detections:
            return [], [], []

        detections = np.array(detections)
        boxes = detections[:, :4].astype(int)
        scores = detections[:, 4]
        class_ids = detections[:, 5].astype(int)

        # Apply Non-Maximum Suppression to reduce duplicate detections
        filtered_indices = apply_nms(boxes, scores, iou_threshold=0.3)  # More aggressive

        if len(filtered_indices) == 0:
            return [], [], []

        boxes = boxes[filtered_indices]
        scores = scores[filtered_indices]
        class_ids = class_ids[filtered_indices]

        # Filter by bounding box size and position (remove sky detections)
        h, w = frame.shape[:2]
        valid_indices = []

        # Define sky region (top 30% of frame)
        sky_threshold = int(h * 0.3)  # Only process detections below 30% from top

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            box_area = box_width * box_height

            # Minimum size: 10x10 pixels, Maximum size: 80% of frame
            min_area = 10 * 10
            max_area = int(w * h * 0.8)

            # Also check aspect ratio (potholes and cracks are usually not extremely wide or tall)
            aspect_ratio = box_width / max(box_height, 1)

            # Filter out detections in sky region (top 30% of frame)
            # Potholes and cracks should be on the ground, not in the sky
            center_y = (y1 + y2) // 2

            # More strict confidence filtering
            min_confidence = 0.005  # Higher minimum confidence

            if (min_area <= box_area <= max_area and
                0.1 <= aspect_ratio <= 10 and  # Reasonable aspect ratio
                center_y > sky_threshold and  # Must be below sky threshold
                score > min_confidence):  # Minimum confidence threshold
                valid_indices.append(i)

        if len(valid_indices) == 0:
            return [], [], []

        # Additional filtering for potholes (should be more compact)
        final_indices = []
        for i in valid_indices:
            class_id = class_ids[i]
            box = boxes[i]
            score = scores[i]

            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            # For potholes: expect more square-ish shape (width/height ratio close to 1)
            if class_id == 1:  # pothole
                aspect_ratio = width / max(height, 1)
                if 0.5 <= aspect_ratio <= 2.0:  # Potholes should be reasonably square
                    # Also check if it's not too large (potholes are usually smaller)
                    area = width * height
                    max_pothole_area = int(w * h * 0.05)  # Max 5% of frame
                    if area <= max_pothole_area:
                        final_indices.append(i)
            else:  # crack - can be elongated
                final_indices.append(i)

        if len(final_indices) == 0:
            return [], [], []

        boxes = boxes[final_indices]
        scores = scores[final_indices]
        class_ids = class_ids[final_indices]

        return boxes, scores, class_ids

    except Exception as e:
        print("Error in YOLO ONNX inference: {}".format(e))
        return [], [], []

def draw_detections(image, detections, tracker_ids=None, confidences=None, class_ids=None):
    """Draw bounding boxes and labels on image"""
    for i in range(len(detections)):
        x1, y1, x2, y2 = detections[i]
        
        # Draw bounding box
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Create label
        label = ""
        if class_ids is not None and i < len(class_ids):
            class_name = CLASS_NAMES[class_ids[i]]
            label = class_name
        
        if tracker_ids is not None and i < len(tracker_ids):
            tracker_id = tracker_ids[i]
            label = "ID: {} {}".format(tracker_id, label)
            
        if confidences is not None and i < len(confidences):
            label += " {:.2f}".format(confidences[i])
        
        # Draw label
        cv2.putText(image, label, (int(x1), int(y1) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def process_frame(frame: np.ndarray, frame_index: int = None) -> np.ndarray:
    """Process a single frame for crack and pothole detection"""
    global pothole_count, crack_count, detection_log, processed_trackers

    start_time = time.time()

    # Run YOLO ONNX inference
    detections, confidences, class_ids = run_yolo_onnx_inference(frame)

    # Detections are already filtered by YOLO model
    filtered_detections = detections
    filtered_confidences = confidences
    filtered_class_ids = class_ids
    
    # Update tracker with filtered detections
    tracked_objects = centroid_tracker.update(filtered_detections)
    
    # Create tracker IDs and associate with detections
    tracker_ids = [-1] * len(filtered_detections)  # Initialize with -1 for all detections

    # Match tracked objects with detections
    used_detection_indices = set()
    for object_id, centroid in tracked_objects:
        # Find the detection closest to this centroid
        min_dist = float('inf')
        closest_idx = -1

        for i, detection in enumerate(filtered_detections):
            if i in used_detection_indices:
                continue  # Skip already matched detections

            cX = int((detection[0] + detection[2]) / 2.0)
            cY = int((detection[1] + detection[3]) / 2.0)
            dist = np.sqrt((cX - centroid[0])**2 + (cY - centroid[1])**2)

            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        if closest_idx != -1 and min_dist < 50:  # Threshold for matching
            tracker_ids[closest_idx] = object_id
            used_detection_indices.add(closest_idx)
    
    # Draw detections
    annotated_frame = frame.copy()
    if len(filtered_detections) > 0:
        annotated_frame = draw_detections(
            annotated_frame, 
            filtered_detections, 
            tracker_ids, 
            filtered_confidences,
            filtered_class_ids
        )
    
    # Check for objects below the line
    timestamp = get_timestamp(frame_index, video_info['fps']) if frame_index else "LIVE"
    
    for i, tracker_id in enumerate(tracker_ids):
        if tracker_id != -1 and tracker_id not in processed_trackers:
            detection = filtered_detections[i]
            center_y = int((detection[1] + detection[3]) / 2)
            
            if center_y >= line_y_position:  # Object is below the line
                class_id = filtered_class_ids[i]
                class_name = CLASS_NAMES[class_id]
                confidence = filtered_confidences[i]
                
                if class_name == 'pothole':
                    pothole_count += 1
                    detection_log.append({
                        'type': 'pothole',
                        'tracker_id': int(tracker_id),
                        'timestamp': timestamp,
                        'frame': frame_index if frame_index else 0,
                        'position': center_y,
                        'confidence': confidence,
                        'detection_type': 'detected_below_line'
                    })
                    print("🕳️ POTHOLE #{0} (ID:#{1}) DETECTED at {2} (Y: {3}, Conf: {4:.2f})".format(
                        pothole_count, tracker_id, timestamp, center_y, confidence))
                
                elif class_name == 'crack':
                    crack_count += 1
                    detection_log.append({
                        'type': 'crack',
                        'tracker_id': int(tracker_id),
                        'timestamp': timestamp,
                        'frame': frame_index if frame_index else 0,
                        'position': center_y,
                        'confidence': confidence,
                        'detection_type': 'detected_below_line'
                    })
                    print("〰️ CRACK #{0} (ID:#{1}) DETECTED at {2} (Y: {3}, Conf: {4:.2f})".format(
                        crack_count, tracker_id, timestamp, center_y, confidence))
                
                processed_trackers.add(tracker_id)
    
    # Add text overlay with statistics
    processing_time = time.time() - start_time
    
    # Frame info
    if frame_index and video_info['total_frames']:
        frame_text = 'Frame: {0}/{1}'.format(frame_index, video_info['total_frames'])
    else:
        frame_text = 'Frame: {0}'.format(frame_index if frame_index else "LIVE")
    
    cv2.putText(annotated_frame, frame_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(annotated_frame, 'Potholes: {0}'.format(pothole_count), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, 'Cracks: {0}'.format(crack_count), (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(annotated_frame, 'Time: {0}'.format(timestamp), (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, 'Process: {0:.3f}s'.format(processing_time), (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(annotated_frame, 'Device: {0}'.format(DEVICE.upper()), (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw detection line
    annotated_frame = cv2.line(annotated_frame,
                             (0, line_y_position),
                             (video_info['width'], line_y_position),
                             (255, 255, 0), 2)
    
    return annotated_frame

def process_camera_feed():
    """Process live camera feed"""
    global cap

    print("Processing camera feed from Camera {}".format(CAMERA_ID))
    print("Model: {}".format(MODEL_PATH))
    print("Line position: Y={} (55% from top)".format(line_y_position))
    print("Resolution: {}x{}".format(video_info['width'], video_info['height']))
    print("FPS: {}".format(video_info['fps']))
    print("Using CUDA: {}".format(USE_CUDA))
    print("Press 'q' to quit, 's' to save current frame")
    print("-" * 60)

    # Setup video writer if saving
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info['fps'],
                                     (video_info['width'], video_info['height']))
        print("Saving video to: {}".format(TARGET_VIDEO_PATH))

    frame_count = 0
    start_processing_time = time.time()

    try:
        cap = cv2.VideoCapture(CAMERA_ID)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera")
                break

            # Process frame
            processed_frame = process_frame(frame, frame_count)

            # Save frame if recording
            if video_writer:
                video_writer.write(processed_frame)

            # Display frame
            cv2.imshow('Crack and Pothole Detection - Live Feed', processed_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_path = "screenshot_{}.jpg".format(datetime.now().strftime('%Y%m%d_%H%M%S'))
                cv2.imwrite(screenshot_path, processed_frame)
                print("Screenshot saved: {}".format(screenshot_path))

            frame_count += 1

    except KeyboardInterrupt:
        print("\nCamera processing interrupted by user")

    finally:
        # Cleanup
        if video_writer:
            video_writer.release()
        cap.release()
        cv2.destroyAllWindows()

    processing_duration = time.time() - start_processing_time
    return processing_duration, frame_count

def process_video_file():
    """Process video file"""
    print("Processing video: {}".format(SOURCE_VIDEO_PATH))
    print("Model: {}".format(MODEL_PATH))
    print("Line position: Y={} (55% from top)".format(line_y_position))
    print("Output: {}".format(TARGET_VIDEO_PATH))
    print("Total frames: {}".format(video_info['total_frames']))
    print("FPS: {}".format(video_info['fps']))
    print("Resolution: {}x{}".format(video_info['width'], video_info['height']))
    print("Using CUDA: {}".format(USE_CUDA))
    print("-" * 60)

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, video_info['fps'],
                                 (video_info['width'], video_info['height']))

    frame_count = 0
    start_processing_time = time.time()

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = process_frame(frame, frame_count)

            # Save frame
            video_writer.write(processed_frame)

            frame_count += 1

            # Print progress
            if frame_count % 100 == 0:
                print("Processed {} frames".format(frame_count))

    except KeyboardInterrupt:
        print("\nVideo processing interrupted by user")

    finally:
        # Cleanup
        video_writer.release()
        cap.release()

    processing_duration = time.time() - start_processing_time
    return processing_duration, frame_count

# Main execution
if __name__ == "__main__":
    if USE_CAMERA:
        # Process camera feed
        processing_duration, total_frames = process_camera_feed()
        source_text = "Camera {}".format(CAMERA_ID)
    else:
        # Process video file
        processing_duration, total_frames = process_video_file()
        source_text = SOURCE_VIDEO_PATH

    # Print final summary
    print("\n" + "="*60)
    print("🔍 FINAL DETECTION SUMMARY")
    print("="*60)
    print("📹 Source: {}".format(source_text))
    print("🕳️  Total Potholes Detected: {}".format(pothole_count))
    print("〰️  Total Cracks Detected: {}".format(crack_count))
    print("📊 Total Detections: {}".format(pothole_count + crack_count))
    print("⏱️  Processing Time: {:.2f} seconds".format(processing_duration))

    if not USE_CAMERA:
        print("🎬 Video Duration: {:.2f} seconds".format(video_info['total_frames']/video_info['fps']))
        print("🚀 Processing Speed: {:.2f} FPS".format(video_info['total_frames']/processing_duration))
    else:
        print("🎬 Total Frames Processed: {}".format(total_frames))
        print("🚀 Processing Speed: {:.2f} FPS".format(total_frames/processing_duration))

    if detection_log:
        print("\n📋 Detailed Detection Log:")
        print("-" * 60)
        pothole_logs = [log for log in detection_log if log['type'] == 'pothole']
        crack_logs = [log for log in detection_log if log['type'] == 'crack']

        if pothole_logs:
            print("\n🕳️  Potholes ({}):".format(len(pothole_logs)))
            for i, log in enumerate(pothole_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print("   {}. Tracker ID #{} at {} (Frame {}, Y: {}, Conf: {:.2f}) [{}]".format(
                    i, log['tracker_id'], log['timestamp'], log['frame'],
                    log['position'], log['confidence'], detection_type.replace('_', ' ').title()))

        if crack_logs:
            print("\n〰️  Cracks ({}):".format(len(crack_logs)))
            for i, log in enumerate(crack_logs, 1):
                detection_type = log.get('detection_type', 'unknown')
                print("   {}. Tracker ID #{} at {} (Frame {}, Y: {}, Conf: {:.2f}) [{}]".format(
                    i, log['tracker_id'], log['timestamp'], log['frame'],
                    log['position'], log['confidence'], detection_type.replace('_', ' ').title()))

    print("="*60)

    # Save detection report to file
    report_content = """
OPTIMIZED CRACK AND POTHOLE DETECTION REPORT
Source: {0}
Model: {1}
Processing Date: {2}
Line Position: Y={3} (55% from top)
Using CUDA: {4}

SUMMARY:
- Total Potholes Detected: {5}
- Total Cracks Detected: {6}
- Total Detections: {7}
- Processing Time: {8:.2f} seconds
- Processing Speed: {9:.2f} FPS

DETAILED LOG:
""".format(
        source_text, MODEL_PATH, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        line_y_position, USE_CUDA, pothole_count, crack_count, 
        pothole_count + crack_count, processing_duration, total_frames/processing_duration
    )

    if detection_log:
        for log in detection_log:
            detection_type = log.get('detection_type', 'unknown')
            report_content += "- {0} (Tracker #{1}) at {2} (Frame {3}, Y: {4}, Confidence: {5:.2f}) [{6}]\n".format(
                log['type'].upper(), log['tracker_id'], log['timestamp'], 
                log['frame'], log['position'], log['confidence'], 
                detection_type.replace('_', ' ').title())

    with open("detection_report_optimized.txt", "w") as f:
        f.write(report_content)

    print("📄 Detection report saved to: detection_report_optimized.txt")
    print("🎥 Annotated video saved to: {}".format(TARGET_VIDEO_PATH))