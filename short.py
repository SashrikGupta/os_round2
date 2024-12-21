from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11s.pt")

# Export the model to TF.js format
model.export(format="tfjs")