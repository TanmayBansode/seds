import os
from yolo import load_model, run_inference_on_image, run_inference_on_video, run_inference_on_folder

# Load model
model = load_model("best.pt")

# Output directory
output_dir = os.path.join(os.getcwd(), "output")
os.makedirs(output_dir, exist_ok=True)

# Run inference on different sources
# run_inference_on_image(model, "test11.jpg", output_dir)
# run_inference_on_video(model, "video1.mp4", output_dir)
run_inference_on_folder(model, "test_images", output_dir)
