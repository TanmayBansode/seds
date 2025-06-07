import os
from yolo import load_model, run_inference_on_image, run_inference_on_video, run_inference_on_folder
import argparse
from config import model_file

# Load model
model, device = load_model(model_file)
print("Model class names:", model.names)
print("Device:", device)

# Output directory
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run YOLOv5 inference on images, videos, or folders.")
    parser.add_argument("-i", "--image", type=str, help="Path to the input image.")
    parser.add_argument("-v", "--video", type=str, help="Path to the input video.")
    parser.add_argument("-f", "--folder", type=str, help="Path to the input folder containing images/videos.")
    args = parser.parse_args()

    if args.image:
        img_output_dir = os.path.join(output_dir, "image_outputs")
        os.makedirs(img_output_dir, exist_ok=True)
        run_inference_on_image(model, args.image, img_output_dir, device)

    elif args.video:
        vid_output_dir = os.path.join(output_dir, "video_outputs")
        os.makedirs(vid_output_dir, exist_ok=True)
        run_inference_on_video(model, args.video, vid_output_dir, device, imgsz=640) 

    elif args.folder:
        output_dir = os.path.join(output_dir, "folder_outputs")
        os.makedirs(output_dir, exist_ok=True)
        run_inference_on_folder(model, args.folder, output_dir, device)

    else:
        print("Please provide a valid input using -i (image), -v (video), or -f (folder).")


