import streamlit as st
import cv2
import base64
import time
import json
from ultralytics import YOLO
import os

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
    bounding_boxes = json.loads(bounding_boxes or "[]")

    frame_count = 0

    # Create a placeholder for the video stream in Streamlit
    frame_placeholder = st.empty()

    while cap.isOpened() and streaming:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLO model for detection and tracking
        results = bb_box_model.track(frame, persist=True, verbose=False)

        # Draw YOLO bounding boxes
        for result in results[0].boxes:
            class_id = int(result.cls.item())  # Get the class ID
            if class_id == 0:  # 0 is usually the class ID for "person"
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                cv2.putText(
                    frame,
                    f"ID: {class_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),  # Green text
                    2
                )

        # Draw each user-defined bounding box
        for box in bounding_boxes:
            x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow bounding box
            cv2.putText(
                frame,
                "User-defined",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),  # Yellow text
                2
            )

        # Encode frame as base64 for Streamlit display
        _, bb_box_data = cv2.imencode('.jpg', frame)
        bb_as_text = base64.b64encode(bb_box_data).decode('utf-8')
        frame_placeholder.image("data:image/jpeg;base64," + bb_as_text, use_container_width=True)

        # Add a small delay to simulate frame rate (to control FPS)
        time.sleep(0.03)
        frame_count += 1

    cap.release()
    st.success("Video streaming completed")


def main():
    st.title("YOLO Object Detection with Bounding Boxes")

    # Video upload section
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    video_path = None

    if uploaded_video is not None:
        video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.success("Video uploaded successfully")

    # Bounding box input section
    bounding_boxes = []

    if st.button("Add Bounding Box"):
        x1 = st.number_input("x1", min_value=0, value=0)
        y1 = st.number_input("y1", min_value=0, value=0)
        x2 = st.number_input("x2", min_value=0, value=640)
        y2 = st.number_input("y2", min_value=0, value=360)
        bounding_boxes.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
        st.write(f"Bounding Box added: {bounding_boxes[-1]}")

    # Start streaming button
    if st.button("Start Video Streaming"):
        if video_path and os.path.exists(video_path):
            global streaming
            streaming = True
            try:
                stream_video(video_path, json.dumps(bounding_boxes))
            except Exception as e:
                st.error("Error during streaming.")
                st.exception(e)
        else:
            st.error("Upload a valid video file first.")

    # Stop streaming button
    if st.button("Stop Streaming"):
        st.success("Streaming stopped")


if __name__ == "__main__":
    main()
