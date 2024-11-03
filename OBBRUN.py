import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Custom title with black font color
st.markdown(
    """
    <h1 style='color: black;'>Object Detection Web App</h1>
    """,
    unsafe_allow_html=True
)

import streamlit as st

# Add custom CSS for background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpaperboat.com/wp-content/uploads/2019/10/free-website-background-21.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Option to choose between image upload and live webcam
option = st.radio("Choose an option:", ("Upload Image", "Live Webcam"))

if option == "Upload Image":
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image using PIL
        image = Image.open(uploaded_file)

        # Convert PIL image to NumPy array for YOLO model
        image_np = np.array(image)

        # Display the uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Detecting objects...")

        # Perform object detection
        results = model.predict(image_np)

        # Convert the detection results to a format suitable for display
        annotated_frame = results[0].plot()
        result_pil = Image.fromarray(annotated_frame)

        # Display the results
        st.image(result_pil, caption="Detected Objects", use_column_width=True)

elif option == "Live Webcam":
    # Add buttons to start and stop webcam
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    # Initialize session state to control the webcam
    if "run_detection" not in st.session_state:
        st.session_state.run_detection = False

    # Update session state based on button clicks
    if start_button:
        st.session_state.run_detection = True
    if stop_button:
        st.session_state.run_detection = False

    # Placeholder for video feed
    video_placeholder = st.empty()

    # Function to capture and process frames
    def get_frame(cap):
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame.")
            return None

        # Perform object detection
        results = model.predict(frame)

        # Render detection results on the frame
        annotated_frame = results[0].plot()

        # Convert frame to RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame

    # Start video capture if detection is active
    cap = None
    if st.session_state.run_detection:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Error: Could not open webcam.")
        else:
            try:
                while st.session_state.run_detection:
                    frame = get_frame(cap)
                    if frame is not None:
                        # Display the frame
                        video_placeholder.image(frame, channels="RGB")
                    time.sleep(0.1)  # Slightly longer delay to reduce load
            finally:
                cap.release()
                video_placeholder.empty()  # Clear the placeholder when done
