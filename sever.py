import streamlit as st
import cv2
import base64
import time
import json
from ultralytics import YOLO , solutions
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt
import streamlit.components.v1 as components

# Define the HTML, CSS, and JavaScript content for the AI Analysis section
bb_as_text = ""

st.set_page_config(layout="wide")

left, center, right = st.columns(3)  # Three columns for layout

# Path to the YOLO model
model_path = r'yolo11n.pt'
bb_box_model = YOLO(model_path)
heatmap_model = solutions.Heatmap(
    show=False,  # Streamlit will handle visualizatio
    model=model_path,  # Use the same model as bb_box_model
    colormap=cv2.COLORMAP_JET,  # Heatmap colormap
    classes = [0],
    verbose=False
)
# Global variable to control the camera streaming
streaming = False

# Create an upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
def generate_heatmap(frame):
    return heatmap_model.generate_heatmap(frame)
# Function to check if a point is inside a rectangular zone
def is_center_in_zone(center, zone):
    x, y = center
    return zone['x1'] <= x <= zone['x2'] and zone['y1'] <= y <= zone['y2']

# Stream video and process
def stream_video(video_path, bounding_boxes, zones):
    global streaming
    global bb_as_text
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error opening video file")
        return

    zones = json.loads(zones or "[]")
    threshold_chart_data = {
        'frame': [],
        'threshold': [],
    }

    zone_stats = {zone['id']: {'footfall': set(), 'current_persons': 0, 'high_density_time': [], 'above_threshold': False, 'history': []} for zone in zones}
    footfall_data = {zone['id']: [] for zone in zones}
    total_zones = len(zones)
    frame_count = 0

    with center:
        frame_placeholder = st.empty()
        threshold_chart_placeholder = st.empty()
    with right:
        heatmap_placeholder = st.empty()
        stats_placeholder = st.empty()
        pie_chart_placeholder = st.empty()

    zone_placeholders = {}
    num_columns = 3
    num_rows = (len(zones) + num_columns - 1) // num_columns

    for i, zone in enumerate(zones):
        col = i % num_columns
        row = i // num_columns
        if row not in zone_placeholders:
            zone_placeholders[row] = []

        zone_placeholders[row].append(st.empty())

    runner = 0
    while cap.isOpened() and streaming:
        success, frame = cap.read()
        if not success:
            break

        heatmap_frame = generate_heatmap(frame)
        _, heatmap_data = cv2.imencode('.jpg', heatmap_frame)
        heatmap_as_text = base64.b64encode(heatmap_data).decode('utf-8')
        heatmap_placeholder.image(f"data:image/jpeg;base64,{heatmap_as_text}", use_container_width=True)

        results = bb_box_model.track(frame, persist=True, verbose=False)

        person_count = 0
        tracked_centers = []

        for result in results[0].boxes:
            class_id = int(result.cls.item())
            track_id = int(result.id.item())

            if class_id == 0:
                person_count += 1
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                tracked_centers.append({'track_id': track_id, 'center': (cx, cy)})

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        zone_person_count = []
        threshold = person_count / total_zones if total_zones > 0 else 0

        threshold_chart_data['frame'].append(runner)
        threshold_chart_data['threshold'].append(threshold)

        with center:
            threshold_chart_placeholder.line_chart(threshold_chart_data)

        for zone in zones:
            zone_id = zone['id']
            zone_info = zone_stats[zone_id]
            current_zone_count = 0

            for person in tracked_centers:
                if is_center_in_zone(person['center'], zone):
                    current_zone_count += 1
                    if person['track_id'] not in zone_info['footfall']:
                        zone_info['footfall'].add(person['track_id'])

            zone_info['current_persons'] = current_zone_count
            zone_person_count.append(current_zone_count)

            if current_zone_count > threshold:
                if not zone_info['above_threshold']:
                    zone_info['high_density_time'].append({'start_time': frame_count})
                    zone_info['above_threshold'] = True
            else:
                if zone_info['above_threshold']:
                    zone_info['high_density_time'][-1]['end_time'] = frame_count
                    zone_info['above_threshold'] = False

            zone_info['history'].append((frame_count, current_zone_count, len(zone_info['footfall'])))

            for person in tracked_centers:
                if is_center_in_zone(person['center'], zone):
                    footfall_data[zone_id].append(person['track_id'])

            if current_zone_count > threshold:
                cv2.rectangle(frame, (zone['x1'], zone['y1']), (zone['x2'], zone['y2']), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (zone['x1'], zone['y1']), (zone['x2'], zone['y2']), (255, 255, 0), 2)

            cv2.putText(
                frame,
                f"Zone {zone_id}: {current_zone_count} Persons",
                (zone['x1'], zone['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            history = np.array(zone_info['history'])
            if history.size > 0:
                footfall = history[:, 2]
                current_count = history[:, 1]

                chart_data = {
                    'Footfall': footfall,
                    'Current Persons': current_count
                }

                zone_index = int(zone_id.split('_')[-1]) - 1
                row = zone_index // num_columns
                col = zone_index % num_columns

                zone_placeholders[row][col].line_chart(chart_data)

        _, bb_box_data = cv2.imencode('.jpg', frame)
        bb_as_text = base64.b64encode(bb_box_data).decode('utf-8')
        with center:
            frame_placeholder.image("data:image/jpeg;base64," + bb_as_text, use_container_width=True)

        pie_chart_data = {
            'Zone': [f"Zone {i+1}" for i in range(total_zones)],
            'Count': zone_person_count
        }
        pie_chart_fig = px.pie(
            pie_chart_data,
            values='Count',
            names='Zone',
        )
        pie_chart_placeholder.plotly_chart(pie_chart_fig, use_column_width=True, key=f"pie_chart_{runner}")

        runner += 1

    cap.release()
    st.success("Video streaming completed")


# Main function
def main():
    with left: 
        st.title("YOLO Object Detection with Real-Time Zone Footfall")

        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        video_path = None

        if uploaded_video is not None:
            video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.success("Video uploaded successfully")

        num_zones = st.number_input("Enter the number of zones", min_value=0, value=0, step=1)
        zones = []
        for i in range(num_zones):
            st.write(f"Zone {i + 1}")
            cols = st.columns(5)
            zone_id = cols[0].text_input(f"ID for Zone {i + 1}", value=f"Zone_{i + 1}", key=f"id_{i}")
            x1 = cols[1].number_input(f"x1 for Zone {i + 1}", min_value=0, value=0, key=f"x1_{i}")
            y1 = cols[2].number_input(f"y1 for Zone {i + 1}", min_value=0, value=0, key=f"y1_{i}")
            x2 = cols[3].number_input(f"x2 for Zone {i + 1}", min_value=0, value=640, key=f"x2_{i}")
            y2 = cols[4].number_input(f"y2 for Zone {i + 1}", min_value=0, value=360, key=f"y2_{i}")
            zones.append({'id': zone_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

    if st.button("Start Video Streaming"):
        # Clear the left column content
        

        if video_path and os.path.exists(video_path):
            global streaming
            streaming = True
            
            # Start video streaming after clearing the left column
            try:
                stream_video(video_path, json.dumps(zones), json.dumps(zones))
            except Exception as e:
                st.error("Error during streaming.")
                st.exception(e)
        else:
            st.error("Upload a valid video file first.")


    
    # Container where AI Analysis content will appear


if __name__ == "__main__":
    main()
