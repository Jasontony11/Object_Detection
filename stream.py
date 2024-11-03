import streamlit as st
import cv2
from ultralytics import YOLO
import time

# Load YOLO model
model = YOLO("yolo11n.pt")

# Streamlit app
st.title("Real-Time Object Detection App")

# Checkbox to start/stop detection
run_detection = st.checkbox("Run Webcam Detection")

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
if run_detection:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
    else:
        while run_detection:
            frame = get_frame(cap)
            if frame is not None:
                # Display the frame
                video_placeholder.image(frame, channels="RGB")
            time.sleep(0.01)  # Small delay to simulate real-time processing

    # Release the webcam when done
    cap.release()
