from ultralytics import YOLO

# Load the model
model = YOLO("best.pt")

# Get model details
model_info = model.model  # Access underlying PyTorch model
class_names = model.names  # Get class names (if available)
hyperparameters = model.overrides  # Get hyperparameters used for training

# Print extracted information
# print("\n🔹 Model Architecture:\n", model_info)
print("\n🔹 Class Names:\n", class_names)
print("\n🔹 Hyperparameters:\n", hyperparameters)
