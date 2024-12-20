from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import base64
import time
from ultralytics import solutions, YOLO
import threading
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Path to the YOLO model
model_path = r'cctv-model.pt'

# Initialize YOLO heatmap
heatmap = solutions.Heatmap(
    show=False,
    model=model_path,
    colormap=cv2.COLORMAP_PARULA,
)

bb_box_model = YOLO(model_path)

# Global variable to control the camera streaming
streaming = False

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('start_camera_stream')
def start_camera_stream():
    global streaming
    if not streaming:
        streaming = True
        # Start the camera stream in the background
        socketio.start_background_task(target=stream_camera)

@socketio.on('stop_camera_stream')
def stop_camera_stream():
    global streaming
    streaming = False
    print("Camera feed stopped")
    
    # Reset heatmap or clear previous storage if required
    reset_heatmap()

def reset_heatmap():
    # Reinitialize the heatmap to clear previous state
    global heatmap
    heatmap = solutions.Heatmap(
        show=False,
        model=model_path,
        colormap=cv2.COLORMAP_PARULA,
    )
    print("Heatmap reset")

def stream_camera():
    global streaming
    cap = cv2.VideoCapture(0)  # Use 0 for the primary webcam
    assert cap.isOpened(), "Error accessing the webcam"

    frame_count = 0
    while cap.isOpened() and streaming:
        success, frame = cap.read()
        if not success:
            break

        # Skip every other frame for efficiency
        if frame_count % 2 != 0:
            frame_count += 1
            continue

        # Resize frame to smaller dimensions for YOLO processing
        resized_frame = cv2.resize(frame, (400, 200))
        bb_box_results = bb_box_model(frame, verbose=False)
        annotated_frame = bb_box_results[0].plot()  # Get annotated frame

        annotated_frame = cv2.resize(annotated_frame, (frame.shape[1], frame.shape[0]))
        # Display the frame in Jupyter Notebook
        _, bb_box_data = cv2.imencode('.jpg', annotated_frame)
        bb_as_text = base64.b64encode(bb_box_data).decode('utf-8')

        # Generate YOLO heatmap
        processed_frame = heatmap.generate_heatmap(resized_frame)

        # Resize back to original size for display
        original_size_frame = cv2.resize(processed_frame, (frame.shape[1], frame.shape[0]))

        # Convert processed frame to JPG and encode it as base64
        _, buffer = cv2.imencode('.jpg', original_size_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # Send the processed frame over WebSocket using socketio.emit
        socketio.emit('new_frame', {'image': jpg_as_text, "bb": bb_as_text})

        # Add a small delay to simulate frame rate
        time.sleep(1 / 30)  # Simulate 30 FPS
        frame_count += 1

    cap.release()
    print("Camera streaming completed")

if __name__ == '__main__':
    # Run the Flask app without auto-reloading in production
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
