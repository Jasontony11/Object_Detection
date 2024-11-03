import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8n model
model = YOLO("yolov8n.pt")

# Title of the app
st.title("Object Detection Web App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting objects...")

    # Perform object detection
    results = model(image)

    # Convert the detection results to a PIL image
    result_img = np.array(results[0].plot())
    result_pil = Image.fromarray(result_img)

    # Display the results
    st.image(result_pil, caption="Detected Objects", use_column_width=True)

# Run the app using: streamlit run app.py
