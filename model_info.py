from yolo import load_model
from config import model_file
model, device = load_model(model_file)


# Get model details
model_info = model.model  # Access underlying PyTorch model
class_names = model.names  # Get class names (if available)
hyperparameters = model.overrides  # Get hyperparameters used for training


# Print extracted information
print("\n🔹 Model Architecture:\n", model_info)
print("\n🔹 Class Names:\n", class_names)
print("\n🔹 Hyperparameters:\n", hyperparameters)
print("\n🔹 Model Device:\n", device)
