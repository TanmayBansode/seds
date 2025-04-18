# YOLO Inference Project

This project demonstrates how to perform inference on images, videos, and entire folders using a pre-trained YOLOv8 model. The trained model (`best.pt`) is loaded, and inference is run on specified input files or folders. The results (labeled images or videos) and detection details are saved in an output directory.

## Project Structure

```
yolo_project/
│── inference.py  # Main script to run inference
│── yolo.py  # Contains all YOLO-related functions
│── best.pt  # Trained YOLOv8 model
│── test2.jpg  # Sample image
│── test_video.mp4  # Sample video
│── images_folder/  # Folder containing multiple images
│── output/  # Stores results from inference
```

## Installation

### 1. Clone this repository:

```bash
git clone https://github.com/TanmayBansode/seds
cd seds
```

### 2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare the YOLOv8 Model:**
   Ensure that you have your **trained YOLOv8 model (`best.pt`)** in the project directory.

2. **Run Inference:**
   You can run inference on an image, a video, or an entire folder of images by executing the `inference.py` script. It will use the trained model (`best.pt`) for detection and save the results.

### Running inference on a single image:

Modify `inference.py` to point to your image (e.g., `test2.jpg`), then run:

```bash
python inference.py
```

### Running inference on a video:

Modify `inference.py` to point to your video (e.g., `test_video.mp4`), then run:

```bash
python inference.py
```

### Running inference on a folder of images:

Modify `inference.py` to point to a folder containing images, then run:

```bash
python inference.py
```

### Output

- **Labeled Image/Video**: The labeled images and videos will be saved in the `output/` folder.
- **Results File**: The detection results (classes, confidence scores, and bounding boxes) will be saved in `results.txt` within the `output/` folder.

---
