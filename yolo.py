import os
import cv2
from ultralytics import YOLO


image_formats = ['bmp', 'jpg', 'png', 'mpo', 'tif', 'webp', 'heic', 'pfm', 'tiff', 'dng', 'jpeg']
video_formats = ['ts', 'mp4', 'mpg', 'asf', 'mov', 'mkv', 'gif', 'webm', 'm4v', 'wmv', 'mpeg', 'avi']
# Load YOLO model
def load_model(model_path="best.pt"):
    """Loads the YOLO model from the given path."""
    return YOLO(model_path)

# Image inference
def run_inference_on_image(model, image_path, output_dir):
    """Runs YOLO inference on a single image and saves results."""
    results = model.predict(image_path)
    labeled_img = results[0].plot()
    title = os.path.basename(image_path).split(".")[0]
    
    output_labeled_path = os.path.join(output_dir, f"{title}_labeled.jpg")
    cv2.imwrite(output_labeled_path, labeled_img)


    results_file = os.path.join(output_dir, f"{title}_results.txt")
    with open(results_file, "a") as f:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0] 
                conf = box.conf[0]  
                cls = int(box.cls[0])  
                label = result.names[cls] 
                f.write(f"Class: {label}, Confidence: {conf:.2f}, BBox: ({x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f})\n")

    print(f"Labeled image saved as: {output_labeled_path}")
    return output_labeled_path

# Video inference
def run_inference_on_video(model, video_path, output_dir):
    """Runs YOLO inference on a video and saves the output."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    output_video_path = os.path.join(output_dir, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.predict(frame)
        labeled_frame = results[0].plot()
        out.write(labeled_frame)

    cap.release()
    out.release()
    print(f"Processed video saved as: {output_video_path}")
    return output_video_path

# Folder inference
def run_inference_on_folder(model, folder_path, output_dir):
    """Runs YOLO inference on all images in a folder."""
    images = [f for f in os.listdir(folder_path) if any(f.endswith(ext) for ext in image_formats)]

    
    if not images:
        print("No images found in folder!")
        return
    
    for img in images:
        image_path = os.path.join(folder_path, img)
        run_inference_on_image(model, image_path, output_dir)
