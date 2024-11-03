import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # Load your YOLO model

# Open webcam feed (0 is for the default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform object detection on the frame
    results = model(frame)

    # Render the detection results on the frame
    annotated_frame = results[0].plot()  # Results[0].plot() to draw bounding boxes

    # Display the frame with detections
    cv2.imshow("YOLO Webcam Detection", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()