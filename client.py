import cv2
import requests
import threading
import time
import os
import sys
from datetime import datetime
from ultralytics import YOLO
# Removed redirect_stdout, redirect_stderr, io as verbose=False is preferred
from collections import defaultdict

# Server config
SERVER_URL = "http://localhost:8000/detections"
COOLDOWN_SEC = 5
DETECTION_BATCH_THRESHOLD = 3
RECORDING_DURATION_SEC = 300 # Stop recording after 5 minutes (300 seconds)

# Detection cache
LAST_SENT = defaultdict(lambda: 0)
DETECTED_TYPES = set()

# --- Thread Management ---
# Keep track of active sending threads
active_threads = []
# Lock for safely modifying the active_threads list (optional but good practice)
thread_lock = threading.Lock()

# Load YOLO model (quietly)
print("[INFO] Loading YOLO model...")
# Note: Initial loading might still print some messages,
# but per-frame predictions will be silent with verbose=False
model = YOLO("face.pt")
model.fuse()
print("[INFO] YOLO model loaded.")

def send_detections(frame, detections, thread_list):
    """Sends detections and removes itself from the tracking list upon completion."""
    current_thread = threading.current_thread()
    try:
        _, img_encoded = cv2.imencode('.jpg', frame)
        # Ensure the image data is bytes
        img_bytes = img_encoded.tobytes()
        files = {'image': ('detection.jpg', img_bytes, 'image/jpeg')}

        # Important: Send the *same* image for all detections in this batch
        # The original code re-opened files for each detection, which is inefficient
        # and could lead to issues if the frame data changed.
        # Instead, we prepare the files dict once.

        sent_count = 0
        for det in detections:
            data = {
                'type': det['type'],
                'user': det['user'],
                'timestamp': datetime.now().isoformat(),
                'bbox': ','.join(map(str, det['bbox']))
            }
            # Make a copy of the files dictionary for each request
            files_copy = {'image': ('detection.jpg', img_bytes, 'image/jpeg')}
            try:
                # print data being sent
                print(f"[SENDING] Sending {data} detection to server...")
            
                response = requests.post(SERVER_URL, files=files_copy, data=data, timeout=10) # Added timeout
                print(f"[SERVER] Sent {data['type']} | Status: {response.status_code}")
                if response.ok:
                    print(f"[RECEIVED] {response.text.strip()}")
                    sent_count += 1
                else:
                    print(f"[ERROR] Server responded with status {response.status_code}")
            except requests.exceptions.RequestException as req_err:
                 print(f"[ERROR] Failed to send {data['type']} detection: {req_err}")
            except Exception as inner_e:
                 print(f"[ERROR] Unexpected error during send for {data['type']}: {inner_e}")

        print(f"[THREAD] Finished sending batch ({sent_count}/{len(detections)} successful).")

    except cv2.error as cv_err:
        print(f"[ERROR] Failed to encode frame: {cv_err}")
    except Exception as e:
        print(f"[ERROR] Failed to send detection batch: {e}")
    finally:
        # Remove this thread from the active list when done
        with thread_lock:
            if current_thread in thread_list:
                thread_list.remove(current_thread)
        # print(f"[DEBUG] Thread {current_thread.name} finished and removed.") # Optional debug print

def process_frame(frame, current_time):
    """Processes a single frame for object detection."""
    # Use verbose=False to suppress YOLO logs during prediction
    results = model.predict(source=frame, stream=False, conf=0.3, iou=0.5, verbose=False)[0]

    annotated_frame = results.plot() # Get frame with bounding boxes drawn
    new_detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detection = {
            "type": label,
            "user": "tanmay2964", # Hardcoded user
            "bbox": (x1, y1, x2, y2)
        }

        key = f"{label}-{detection['user']}" # Unique key per object type and user
        if current_time - LAST_SENT[key] > COOLDOWN_SEC:
            print(f"[DETECTED] {label} at {datetime.now().strftime('%H:%M:%S')}")
            new_detections.append(detection)
            DETECTED_TYPES.add(key) # Add to the set of types detected *since last send*

    # --- Sending Logic ---
    # Send if the batch threshold is met OR if there are *any* new detections
    # (The original logic might have missed sending single detections if threshold wasn't met)
    if new_detections: # Simplified condition: send if anything new is detected and cooled down
        # Create and start the sending thread
        thread = threading.Thread(target=send_detections, args=(annotated_frame.copy(), new_detections, active_threads), daemon=True)
        # Use daemon=True so threads don't block exit if script forced quit,
        # but we will explicitly wait for them in `finally`.
        thread.start()

        # Add thread to our tracking list
        with thread_lock:
            active_threads.append(thread)
        # print(f"[DEBUG] Started thread {thread.name}. Active threads: {len(active_threads)}") # Optional debug print


        # Update last sent time for items just sent
        for d in new_detections:
            key = f"{d['type']}-{d['user']}"
            LAST_SENT[key] = current_time
        DETECTED_TYPES.clear() # Clear the set for the next batch

    # Check if batch threshold logic is still needed (original had it, maybe remove if confusing?)
    # The current logic sends immediately when a cooled-down item is detected.
    # If you strictly want to *batch* sending until 3 *different types* are seen,
    # the logic needs adjustment. Let's keep the immediate send for now.

    return annotated_frame

def main():
    global active_threads # Allow modification of the global list

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)

    # Set desired frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"[INFO] Attempting to set resolution to {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    start_time = time.time()
    print(f"[INFO] Starting recording. Will stop after {RECORDING_DURATION_SEC} seconds or on 'q' press.")

    try:
        while True:
            # Check elapsed time
            current_elapsed_time = time.time() - start_time
            if current_elapsed_time > RECORDING_DURATION_SEC:
                print(f"\n[INFO] Recording time limit ({RECORDING_DURATION_SEC}s) reached.")
                break

            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Can't receive frame (stream end?). Exiting ...")
                break

            # Process the frame
            current_process_time = time.time()
            annotated_frame = process_frame(frame, current_process_time)

            # Display the resulting frame
            cv2.imshow("Live Detection", annotated_frame)

            # Press 'q' to exit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] 'q' pressed, stopping recording.")
                break

            # Optional: Clean up finished threads periodically from the list
            # This prevents the list from growing indefinitely if the script runs for a very long time
            # with thread_lock:
            #    active_threads = [t for t in active_threads if t.is_alive()]

    except KeyboardInterrupt:
         print("\n[INFO] KeyboardInterrupt detected. Stopping.")
    finally:
        print("[INFO] Stopping video capture.")
        cap.release()
        cv2.destroyAllWindows()

        print(f"[INFO] Main loop finished. Waiting for {len(active_threads)} pending uploads to complete...")
        # Make a copy of the list to iterate over, as threads remove themselves
        threads_to_wait_for = []
        with thread_lock:
            threads_to_wait_for = active_threads[:]

        for thread in threads_to_wait_for:
            thread.join() # Wait for the thread to complete its task

        print("[INFO] All uploads finished.")
        print("[INFO] Script finished cleanly.")

if __name__ == "__main__":
    main()