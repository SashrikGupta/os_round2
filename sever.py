import streamlit as st
import cv2
import base64
import time
import json
from ultralytics import YOLO
import os
from PIL import Image

# Path to the YOLO model
model_path = r'yolo11s.pt'
bb_box_model = YOLO(model_path)

# Global variable to control the camera streaming
streaming = False

# Create an upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def stream_video(video_path, bounding_boxes):
    """Stream video with bounding boxes for persons only."""
    global streaming
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file")
        return

    # Parse bounding boxes from the frontend (expecting a JSON array of boxes)
    bounding_boxes = json.loads(bounding_boxes)

    # Get the original frame dimensions to calculate the resize ratio
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resize target frame dimensions
    target_width = 640
    target_height = 360

    # Calculate the resize ratio for width and height
    width_ratio = target_width / original_width
    height_ratio = target_height / original_height

    frame_count = 0

    # Create a placeholder for the video stream in Streamlit
    frame_placeholder = st.empty()

    while cap.isOpened() and streaming:
        success, frame = cap.read()
        if not success:
            break

        # Resize the frame for processing
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Run YOLO model for detection and tracking
        results = bb_box_model.track(resized_frame, persist=True, verbose=False)

        # Manually draw bounding boxes with only the "person" class
        for result in results[0].boxes:
            track_id = int(result.id.item())  # Convert tensor to integer
            x1_res, y1_res, x2_res, y2_res = result.xyxy[0]  # Get the bounding box coordinates

            # Check if the detected object is a person (class ID 0)
            class_id = int(result.cls.item())  # Get the class ID
            if class_id == 0:  # 0 is usually the class ID for "person"
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1_res), int(y1_res)), (int(x2_res), int(y2_res)), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (int(x1_res), int(y1_res)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw each user-defined bounding box
        for box in bounding_boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

            # Resize the user-defined zone bounding box coordinates based on the resize ratio
            x1_resized = int(x1 * width_ratio)
            y1_resized = int(y1 * height_ratio)
            x2_resized = int(x2 * width_ratio)
            y2_resized = int(y2 * height_ratio)

            # Draw the resized user-defined zone bounding box
            cv2.rectangle(frame, (x1_resized, y1_resized), (x2_resized, y2_resized), (255, 255, 0), 2)

        # Encode frame as base64 for Streamlit display
        _, bb_box_data = cv2.imencode('.jpg', frame)
        bb_as_text = base64.b64encode(bb_box_data).decode('utf-8')

        # Update the image displayed in Streamlit by using the placeholder
        frame_placeholder.image("data:image/jpeg;base64," + bb_as_text, use_column_width=True)

        # Add a small delay to simulate frame rate (to control FPS)
        time.sleep(1 / 20)  # Simulate 20 FPS
        frame_count += 1

    cap.release()
    st.success("Video streaming completed")

def main():
    st.title("YOLO Object Detection with Bounding Boxes")

    # Video upload section
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Only process the video after it's uploaded
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
        print(video_path)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.success("Video uploaded successfully")
    else:
        video_path = None

    # Bounding box input section
    st.header("Define Bounding Boxes")

    bounding_boxes = []
    add_box = st.button("Add Bounding Box")
    if add_box:
        x1 = st.number_input("x1", min_value=0, value=0)
        y1 = st.number_input("y1", min_value=0, value=0)
        x2 = st.number_input("x2", min_value=0, value=640)
        y2 = st.number_input("y2", min_value=0, value=360)
        bounding_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        st.write(f"Bounding Box added: {bounding_boxes[-1]}")

    # Start streaming button
    if st.button("Start Video Streaming"):
            if video_path:
                global streaming
                streaming = True
                stream_video(video_path, json.dumps(bounding_boxes))
            else:
                st.error("Video upload failed. Please try again.")

    # Stop streaming button
    if st.button("Stop Streaming"):
        streaming = False
        st.success("Streaming stopped")

if __name__ == "__main__":
    main()

