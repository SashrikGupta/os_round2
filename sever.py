from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import base64
import time
from ultralytics import YOLO
import os
import json

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Path to the YOLO model
model_path = r'yolo11s.pt'
bb_box_model = YOLO(model_path)

# Global variable to control the camera streaming
streaming = False

# Create an upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and start streaming.""" 
    global streaming

    video = request.files.get('video')
    bounding_boxes = request.form.get('bounding_box')

    if video:
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        video.save(video_path)

        if not streaming:
            streaming = True
            # Send the bounding boxes along with the video path for processing
            socketio.start_background_task(target=stream_video, video_path=video_path, bounding_boxes=bounding_boxes)

        return jsonify({"success": True, "video_path": video_path})
    return jsonify({"success": False, "error": "No file uploaded"}), 400

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('stop_camera_stream')
def stop_camera_stream():
    global streaming
    streaming = False
    print("Camera feed stopped")

def stream_video(video_path, bounding_boxes):
    """Stream video with bounding boxes for persons only."""
    global streaming
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
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

        # Encode frame as base64
        _, bb_box_data = cv2.imencode('.jpg', frame)
        bb_as_text = base64.b64encode(bb_box_data).decode('utf-8')

        # Send the processed frame via WebSocket
        socketio.emit('new_frame', {'image': bb_as_text})

        # Add a small delay to simulate frame rate
        time.sleep(1 / 20)  # Simulate 20 FPS
        frame_count += 1

    cap.release()
    print("Video streaming completed")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=False ,  allow_unsafe_werkzeug=True)
