import os
import cv2
from ultralytics import YOLO
import time
import torch  

image_formats = ['bmp', 'jpg', 'png', 'mpo', 'tif', 'webp', 'heic', 'pfm', 'tiff', 'dng', 'jpeg']
video_formats = ['ts', 'mp4', 'mpg', 'asf', 'mov', 'mkv', 'gif', 'webm', 'm4v', 'wmv', 'mpeg', 'avi']

# Load YOLO model
def load_model(model_path="best12.pt"):
    """Loads the YOLO model from the given path."""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = YOLO(model_path)
    model.to(device) 
    return model, device

# Image inference  
def run_inference_on_image(model, image_path, output_dir, device):
    """Runs YOLO inference on a single image and saves results."""
    results = model.predict(image_path, device=device)  
    labeled_img = results[0].plot()
    title = os.path.basename(image_path).split(".")[0]

    output_labeled_path = os.path.join(output_dir, f"{title}_labeled.jpg")
    cv2.imwrite(output_labeled_path, labeled_img)

    results_file = os.path.join(output_dir, f"{title}_results.txt")
    with open(results_file, "w") as f:  
        for result in results:
            if result.boxes is not None:  
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    f.write(f"Class: {label}, Confidence: {conf:.2f}, BBox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})\n")
            else:
                 f.write("No objects detected.\n")


    print(f"Labeled image saved as: {output_labeled_path}")
    return output_labeled_path

# Video inference
def run_inference_on_video(model, video_path, output_dir, device,
                           conf_thresh=0.25, iou_thresh=0.7, imgsz=None):
    """
    Runs YOLO inference on a video using streaming mode for better performance
    and saves the output.

    Args:
        model: The loaded YOLO model object.
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the processed video and results.
        device (str): Device to run inference on ('cuda' or 'cpu').
        conf_thresh (float): Confidence threshold for predictions.
        iou_thresh (float): IoU threshold for Non-Maximum Suppression (NMS).
        imgsz (int, optional): Inference image size. Resizes input frames to
                               this size for prediction. Defaults to None (uses
                               original size or model's default).
    """
    start_time = time.time()

    # --- Get Video Properties ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()  

    # --- Set up Output Video Writer ---
    output_video_name = os.path.basename(video_path)
  
    output_video_name = os.path.splitext(output_video_name)[0] + ".mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # Codec for .mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}")
    print(f"Outputting to: {output_video_path}")

    # --- Run Inference using Stream Mode ---
 
    results_generator = model.predict(
        source=video_path,
        stream=True,
        device=device,
        conf=conf_thresh,
        iou=iou_thresh,
        imgsz=imgsz  
    )

    processed_frames = 0
    try:
        for results in results_generator:
           
            labeled_frame = results.plot()  


            if labeled_frame.shape[1] != width or labeled_frame.shape[0] != height:
                 labeled_frame = cv2.resize(labeled_frame, (width, height))

            out.write(labeled_frame)
            processed_frames += 1

            print(f"Processed frame {processed_frames}/{frame_count}", end='\r')

    except Exception as e:
        print(f"\nAn error occurred during video processing: {e}")
    finally:
        out.release()
        print(f"\nFinished processing. Processed {processed_frames} frames.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to process video: {elapsed_time:.2f} seconds")
    print(f"Processed video saved as: {output_video_path}")

    if elapsed_time > 0:
        processing_fps = processed_frames / elapsed_time
        print(f"Processing speed: {processing_fps:.2f} FPS")

    return output_video_path


# Folder inference 
def run_inference_on_folder(model, folder_path, output_dir, device):
    """Runs YOLO inference on all images/videos in a folder."""
    media_files = os.listdir(folder_path)

    if not media_files:
        print("No files found in folder!")
        return

    image_files = [f for f in media_files if any(f.lower().endswith(ext) for ext in image_formats)]
    video_files = [f for f in media_files if any(f.lower().endswith(ext) for ext in video_formats)]

    print(f"Found {len(image_files)} images and {len(video_files)} videos.")

    if image_files:
        print("\n--- Processing Images ---")
        img_output_dir = os.path.join(output_dir, "image_outputs")
        os.makedirs(img_output_dir, exist_ok=True)
        for img in image_files:
            image_path = os.path.join(folder_path, img)
            print(f"Processing image: {img}")
            run_inference_on_image(model, image_path, img_output_dir, device) # Pass device

    if video_files:
        print("\n--- Processing Videos ---")
        vid_output_dir = os.path.join(output_dir, "video_outputs")
        os.makedirs(vid_output_dir, exist_ok=True)
        for vid in video_files:
            video_path = os.path.join(folder_path, vid)
            print(f"Processing video: {vid}")
            run_inference_on_video(model, video_path, vid_output_dir, device) # Pass device


