import cv2
import requests
import threading
import time
import os
import sys
import argparse 
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict
import json #

# --- Configuration ---
SERVER_URL = "http://localhost:8000/traffic_data" # Updated endpoint? Needs to accept new fields
COOLDOWN_SEC = 5
# DETECTION_BATCH_THRESHOLD = 3 # Removed, sending based on cooldown per type now
RECORDING_DURATION_SEC = 300 # Only relevant for --live mode

# Congestion Configuration
# Adjust these thresholds based on your needs
CONGESTION_THRESHOLDS = {
    "Light": 30.0, # Below this percentage is Light
    "Moderate": 60.0, # Below this percentage is Moderate
    "High": 100.0 # Anything up to 100% is High
}

# Check model.names after loading if unsure
VEHICLE_CLASSES = ['Person', 'Cyclist', 'Bike', 'Cycle-Rickshaw', 'Auto-Rickshaw', 'Car', 'Taxi', 'Jeep', 'Van', 'Bus', 'Tempo', 'Truck']

# --- Detection Cache & Threading ---
LAST_SENT = defaultdict(lambda: 0)
active_threads = []
thread_lock = threading.Lock()

# --- Load YOLO Model ---
print("[INFO] Loading YOLO model...")
try:
    model_path = "vehicle_s.pt"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        sys.exit(1)
    model = YOLO(model_path)
    print("[INFO] YOLO model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load YOLO model: {e}")
    sys.exit(1)

# --- Functions ---

def calculate_congestion(vehicle_boxes, frame_width, frame_height):
    """Calculates congestion level based on vehicle bounding box area."""
    if frame_width == 0 or frame_height == 0:
        return 0.0, "Unknown"

    total_frame_area = frame_width * frame_height
    total_bbox_area = 0

    for box in vehicle_boxes:
        x1, y1, x2, y2 = box
        bbox_area = (x2 - x1) * (y2 - y1)
        total_bbox_area += bbox_area

    congestion_percent = (total_bbox_area / total_frame_area) * 100

    level = "High" 
    if congestion_percent < CONGESTION_THRESHOLDS["Light"]:
        level = "Light"
    elif congestion_percent < CONGESTION_THRESHOLDS["Moderate"]:
        level = "Moderate"

    return congestion_percent, level

def send_frame_data(frame, detections_to_send, congestion_percent, congestion_level, thread_list):
    """Sends frame data including detections and congestion info to the server."""
    current_thread = threading.current_thread()
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        files = {'image': ('detection.jpg', img_bytes, 'image/jpeg')}

        # Prepare individual detection data (filter out non-essential keys for sending)
        detections_payload = []
        for det in detections_to_send:
             detections_payload.append({
                 'type': det['type'],
                 'bbox': ','.join(map(str, det['bbox'])) # Send bbox as string
                 # Add 'confidence' if needed: 'confidence': det['confidence']
             })

        data = {
            'id' : "JM Road",
            'location': (18.5204,73.8567), # (latitude, longitude) 
            'timestamp': datetime.now().isoformat(),
            'congestion_level': congestion_level,
            'congestion_percent': round(congestion_percent, 2),
            'detections_json': json.dumps(detections_payload) # Send detections as JSON string
        }

        try:
            print(f"[SENDING] Data: {data} to {SERVER_URL}")
            response = requests.post(SERVER_URL, files=files, data=data, timeout=15) 
            print(f"[SERVER] Status: {response.status_code}")
            if response.ok:
                print(f"[RECEIVED] {response.text.strip()}")
            else:
                print(f"[ERROR] Server responded with status {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as req_err:
            print(f"[ERROR] Failed to send data: {req_err}")
        except Exception as inner_e:
            print(f"[ERROR] Unexpected error during send: {inner_e}")

    except cv2.error as cv_err:
        print(f"[ERROR] Failed to encode frame: {cv_err}")
    except Exception as e:
        print(f"[ERROR] Failed during send_frame_data preparation: {e}")
    finally:
        with thread_lock:
            if current_thread in thread_list:
                thread_list.remove(current_thread)
        print(f"[DEBUG] Thread {current_thread.name} finished and removed.") 

def process_frame(frame, current_time):
    """Processes a single frame for object detection and congestion."""
    if frame is None:
        print("[ERROR] Received None frame in process_frame")
        return None, [], 0.0, "Error"

    try:
        h, w, _ = frame.shape
    except ValueError:
        print("[ERROR] Could not unpack frame shape. Is it a valid image?")
        return frame, [], 0.0, "Error" # Return original frame if shape fails

    results = model.predict(source=frame.copy(), stream=False, conf=0.3, iou=0.5, verbose=False)[0] # Process a copy

    # Get frame with bounding boxes drawn by default plot function
    annotated_frame = results.plot()

    new_detections_for_send = []
    vehicle_bboxes_for_congestion = []

    for box in results.boxes:
        try:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Check if it's a vehicle for congestion calculation
            if label in VEHICLE_CLASSES:
                vehicle_bboxes_for_congestion.append((x1, y1, x2, y2))

            # Check cooldown for sending individual detection info
            key = f"{label}-jmroad" # Unique key per object type and user
            if current_time - LAST_SENT[key] > COOLDOWN_SEC:
                print(f"[DETECTED] {label} (Confidence: {conf:.2f}) at {datetime.now().strftime('%H:%M:%S')}")
                detection_data = {
                    "type": label,
                    "id": "jmroad",
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf 
                }
                new_detections_for_send.append(detection_data)
                LAST_SENT[key] = current_time # Update last sent time *only if added*
        except Exception as e:
            print(f"[ERROR] Error processing detection box: {e}")
            continue # Skip this box

    # Calculate congestion
    congestion_percent, congestion_level = calculate_congestion(vehicle_bboxes_for_congestion, w, h)

    # Display congestion info on the frame
    text = f"Congestion: {congestion_level} ({congestion_percent:.1f}%)"
    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Sending Logic ---
    # Send frame data if *any* new detection passed the cooldown
    if new_detections_for_send:
        # Use the *original* frame for sending, not the annotated one, unless annotation is desired on server
        thread = threading.Thread(target=send_frame_data, args=(frame.copy(), new_detections_for_send, congestion_percent, congestion_level, active_threads), daemon=True)
        thread.start()
        with thread_lock:
            active_threads.append(thread)

    return annotated_frame, new_detections_for_send, congestion_percent, congestion_level

def process_image(image_path):
    """Processes a single image file."""
    print(f"[INFO] Processing image: {image_path}")
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[ERROR] Failed to read image: {image_path}")
        return

    current_process_time = time.time()
    annotated_frame, detections, congestion_p, congestion_l = process_frame(frame, current_process_time)

    if annotated_frame is not None:
        print(f"[RESULT] Image: {os.path.basename(image_path)} | Congestion: {congestion_l} ({congestion_p:.1f}%) | Detections Sent: {len(detections)}")
        cv2.imshow(f"Detection Result - {os.path.basename(image_path)}", annotated_frame)
        print("[INFO] Press any key to close the image window.")
        cv2.waitKey(0) # Wait indefinitely until a key is pressed
        cv2.destroyWindow(f"Detection Result - {os.path.basename(image_path)}") # Close specific window
    else:
        print(f"[ERROR] Frame processing failed for {image_path}")


def process_video(video_path):
    """Processes a video file."""
    print(f"[INFO] Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video file reached.")
            break

        frame_count += 1
        current_process_time = time.time()
        annotated_frame, detections, congestion_p, congestion_l = process_frame(frame, current_process_time)

        if annotated_frame is not None:
             print(f"\r[VIDEO] Frame: {frame_count} | Congestion: {congestion_l} ({congestion_p:.1f}%) | Detections Sent: {len(detections)}", end="")
             cv2.imshow("Video Detection", annotated_frame)
             # Press 'q' to exit early
             if cv2.waitKey(1) & 0xFF == ord('q'): # Wait 1ms, check for 'q'
                 print("\n[INFO] 'q' pressed, stopping video processing.")
                 break
        else:
             print(f"\n[ERROR] Frame processing failed for frame {frame_count}")
             # Decide if you want to break or continue on error
             # break

    print("\n[INFO] Video processing finished.")
    cap.release()
    cv2.destroyWindow("Video Detection") # Close specific window


def process_live():
    """Processes live feed from camera."""
    print("[INFO] Starting live camera feed processing...")
    # Try different backends if default fails
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Windows DirectShow
    cap = cv2.VideoCapture(0) # Default backend

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    # Attempt to set resolution (camera might not support it)
    target_w, target_h = 1280, 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"[INFO] Requested resolution: {target_w}x{target_h}. Actual resolution: {int(actual_w)}x{int(actual_h)}")

    start_time = time.time()
    print(f"[INFO] Live processing started. Press 'q' to quit. Max duration: {RECORDING_DURATION_SEC}s.")

    frame_count = 0
    while True:
        # Check elapsed time for duration limit
        current_elapsed_time = time.time() - start_time
        if current_elapsed_time > RECORDING_DURATION_SEC:
            print(f"\n[INFO] Max duration ({RECORDING_DURATION_SEC}s) reached.")
            break

        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Can't receive frame (stream end?). Exiting ...")
            break

        frame_count += 1
        current_process_time = time.time()
        annotated_frame, detections, congestion_p, congestion_l = process_frame(frame, current_process_time)

        if annotated_frame is not None:
            print(f"\r[LIVE] Frame: {frame_count} | Congestion: {congestion_l} ({congestion_p:.1f}%) | Detections Sent: {len(detections)}", end="")
            cv2.imshow("Live Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q' pressed, stopping live feed.")
                break
        else:
            print(f"\n[ERROR] Frame processing failed for live frame {frame_count}")
            # break # Optional: stop on error

    print("\n[INFO] Live processing finished.")
    cap.release()
    cv2.destroyWindow("Live Detection")


def process_folder(folder_path):
    """Processes all images in a folder."""
    print(f"[INFO] Processing image folder: {folder_path}")
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"[WARN] No supported image files found in folder: {folder_path}")
        return

    print(f"[INFO] Found {len(image_files)} images to process.")
    for filename in image_files:
        image_full_path = os.path.join(folder_path, filename)
        process_image(image_full_path) # Reuse the single image processor
        # Add a small delay or waitkey if you want to see each image result longer
        # cv2.waitKey(500) # Wait 500ms

    print(f"[INFO] Finished processing folder: {folder_path}")
    # Explicitly destroy all windows that might be left open by process_image
    cv2.destroyAllWindows()


def main():
    global active_threads # Allow modification

    parser = argparse.ArgumentParser(description="Traffic Congestion Detection and Reporting")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--live', action='store_true', help='Use live camera feed.')
    group.add_argument('--video', type=str, help='Path to the video file.')
    group.add_argument('--image', type=str, help='Path to a single image file.')
    group.add_argument('--folder', type=str, help='Path to a folder containing images.')

    # Add optional arguments if needed (e.g., server URL, model path)
    # parser.add_argument('--server', type=str, default=SERVER_URL, help='Server URL for reporting.')
    # parser.add_argument('--model', type=str, default="vehicle_s.pt", help='Path to YOLO model file.')

    args = parser.parse_args()

    try:
        if args.live:
            process_live()
        elif args.video:
            if not os.path.exists(args.video):
                 print(f"[ERROR] Video file not found: {args.video}")
                 sys.exit(1)
            process_video(args.video)
        elif args.image:
            if not os.path.exists(args.image):
                 print(f"[ERROR] Image file not found: {args.image}")
                 sys.exit(1)
            process_image(args.image)
        elif args.folder:
            if not os.path.isdir(args.folder):
                 print(f"[ERROR] Folder not found: {args.folder}")
                 sys.exit(1)
            process_folder(args.folder)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt detected. Stopping.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred in main: {e}")
    finally:
        print("[INFO] Main processing finished or interrupted.")
        # Ensure any remaining OpenCV windows are closed (belt-and-suspenders)
        cv2.destroyAllWindows()

        print(f"[INFO] Waiting for {len(active_threads)} pending uploads to complete...")
        # Make a copy for safe iteration while threads might remove themselves
        threads_to_wait_for = []
        with thread_lock:
            threads_to_wait_for = active_threads[:]

        for thread in threads_to_wait_for:
             if thread.is_alive():
                  print(f"[INFO] Waiting for thread {thread.name}...")
                  thread.join() # Wait for the thread to complete

        # Final check on remaining threads (should be empty if join worked)
        with thread_lock:
            remaining_threads = len(active_threads)
        if remaining_threads == 0:
            print("[INFO] All uploads finished.")
        else:
             print(f"[WARN] {remaining_threads} threads still marked as active after join attempt.")

        print("[INFO] Script finished.")


if __name__ == "__main__":
    main()