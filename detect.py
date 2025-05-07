from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("best.pt")


# Perform object detection on an image
results = model("dataset/test/paper-waste-151-_jpg.rf.36443315daa95f22519e9ba50b899ba4.jpg")  # Predict on an image
results[0].show()  # Display results