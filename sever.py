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
import requests
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


def highlight_density(row):
    if row['density'] == 'UNSTABLE':
        return [f'background-color: rgba(255, 0, 0, 0.3)'] * len(row)  # Red with 30% opacity
    else:
        return [f'background-color: rgba(0, 255, 0, 0.3)'] * len(row)  # Green with 30% opacity

def generate_heatmap(frame):
    return heatmap_model.generate_heatmap(frame)
# Function to check if a point is inside a rectangular zone
def is_center_in_zone(center, zone):
    x, y = center
    return zone['x1'] <= x <= zone['x2'] and zone['y1'] <= y <= zone['y2']
def download_video_from_url(video_url, save_path):
    """Downloads video from a URL and saves it to the specified path."""
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True, "Video downloaded successfully"
    except Exception as e:
        return False, str(e)
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
        'total_person': [],
    }

    zone_stats = {zone['id']: {'footfall': set(), 'current_persons': 0, 'unique_persons': set(), 'high_density_time': [], 'above_threshold': False, 'history': []} for zone in zones}
    footfall_data = {zone['id']: [] for zone in zones}
    total_zones = len(zones)
    frame_count = 0

    with center:
        frame_placeholder = st.empty()
        st.write(f"**total count**")
        threshold_chart_placeholder = st.empty()
    with right:
        heatmap_placeholder = st.empty()
        stats_placeholder = st.empty()
        st.write(f"**Zone wise Pie Chart**")
        pie_chart_placeholder = st.empty()

    zone_placeholders = {}

    with st.container():  # Use a full-width container
        for i, zone in enumerate(zones):
                st.write(f"**Graph for {zone['id']}**")
                zone_placeholders[zone["id"]] = st.empty() 

    runner = 0

    with st.sidebar : 
        st.title("Analytics Tab")
        st.write(f"**Current Statistics**")
        table_placeholder = st.empty()
        info_placeholder = st.empty()
        st.write(f"**High density Zone Distribution**")
        zone_threshold_placeholder = st.empty()

    zone_boke = {f'{zone["id"]}' : [] for zone in zones}


    


    while cap.isOpened() and streaming:
        unstability_flag = False 
        df = {
                'zones' :   [] , 
                'count' :   [] , 
                'density' : [] ,  
                'emoji' :   []
            }
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
        threshold = 0.4

        threshold_chart_data['total_person'].append(person_count)

        with center:
            threshold_chart_placeholder.line_chart(threshold_chart_data)

        df['zones'].append('full')
        df['count'].append(person_count)
        print(person_count)
        df['density'].append('')
        df['emoji'].append('')

        for zone in zones:
            zone_id = zone['id']
            df['zones'].append(zone_id)
            zone_info = zone_stats[zone_id]
            current_zone_count = 0

            for person in tracked_centers:
                if is_center_in_zone(person['center'], zone):
                    current_zone_count += 1

                    if person['track_id'] not in zone_info['footfall']:
                        zone_info['footfall'].add(person['track_id'])

                    if person['track_id'] not in zone_info['unique_persons']:
                        zone_info['unique_persons'].add(person['track_id'])

            zone_info['current_persons'] = current_zone_count
            zone_person_count.append(current_zone_count)
            df['count'].append(current_zone_count)

            if current_zone_count/person_count > threshold:
                if not zone_info['above_threshold']:
                    zone_info['high_density_time'].append({'start_time': frame_count})
                    zone_info['above_threshold'] = True
                df['density'].append('UNSTABLE')
                df['emoji'].append('❌')
                zone_boke[zone_id].append(1)
                unstability_flag = True 
            else:
                if zone_info['above_threshold']:
                    zone_info['high_density_time'][-1]['end_time'] = frame_count
                    zone_info['above_threshold'] = False
                df['density'].append('STABLE')
                df['emoji'].append('👍')
                zone_boke[zone_id].append(0)

            zone_info['history'].append((frame_count, current_zone_count, len(zone_info['footfall']), len(zone_info['unique_persons'])))

            color = (0, 0, 255) if current_zone_count/person_count > threshold else (255, 255, 0)
            cv2.rectangle(frame, (zone['x1'], zone['y1']), (zone['x2'], zone['y2']), color, 2)
            cv2.putText(
                frame,
                f"Zone {zone_id}: {current_zone_count} Persons, {len(zone_info['unique_persons'])} Unique",
                (zone['x1'], zone['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            history = np.array(zone_info['history'])
            if history.size > 0:
                footfall = history[:, 2]
                current_count = history[:, 1]

                chart_data = {
                    'Current Persons': current_count
                }

            
            zone_placeholders[zone["id"]].line_chart(chart_data)

        if unstability_flag:
            show_string = """ALERT : High Density at areas : 
"""
            for row in df:
                try:
                    # Ensure `row` is a dictionary before accessing keys
                    if isinstance(row, dict) and 'count' in row and 'zones' in row:
                        if row['count'] / person_count > threshold:
                            show_string += f"Zone {row['zones']}, "
                except Exception as e:
                    print(f"Error processing row: {row}, Error: {e}")
            show_string += """ 
Please Send STAFF ❗❗"""

            info_placeholder.error(show_string)
        else : 
            info_placeholder.success("all Good 👍")


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
        pie_chart_placeholder.plotly_chart(pie_chart_fig, use_container_width=True, key=f"pie_chart_{runner}")

        runner += 1
        table_placeholder.table(pd.DataFrame(df))
        zone_threshold_placeholder.area_chart(zone_boke)
    cap.release()
    st.success("Video streaming completed")


# Main function
def main():
    # Initialize session state
    with left : 
        st.title("Realtime crowd Footfall and zone Analytics ")
        # File uploader
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        video_path = None

        # URL input for video download
        video_url = st.text_input("Or, enter video URL to download:")
        if video_url:
            video_name = os.path.basename(video_url)
            video_path = os.path.join(UPLOAD_FOLDER, video_name)
            if st.button("Download Video"):
                success, message = download_video_from_url(video_url, video_path)
                if success:
                    st.success(f"Video downloaded successfully: {video_name}")
                else:
                    st.error(f"Failed to download video: {message}")

        # Check if uploaded or downloaded video exists
        if uploaded_video is not None:
            video_path = os.path.join(UPLOAD_FOLDER, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.success(f"Video uploaded successfully: {uploaded_video.name}")

        # Zone input within an expander (dropdown)
        zones = []
        with st.expander("Zone Configuration", expanded=True):
            num_zones = st.number_input("Enter the number of zones", min_value=0, value=0, step=1)
            for i in range(num_zones):
                st.write(f"Zone {i + 1}")
                cols = st.columns(5)
                zone_id = cols[0].text_input(f"ID for Zone {i + 1}", value=f"Zone_{i + 1}", key=f"id_{i}")
                x1 = cols[1].number_input(f"x1 for Zone {i + 1}", min_value=0, value=0, key=f"x1_{i}")
                y1 = cols[2].number_input(f"y1 for Zone {i + 1}", min_value=0, value=0, key=f"y1_{i}")
                x2 = cols[3].number_input(f"x2 for Zone {i + 1}", min_value=0, value=640, key=f"x2_{i}")
                y2 = cols[4].number_input(f"y2 for Zone {i + 1}", min_value=0, value=360, key=f"y2_{i}")
                zones.append({'id': zone_id, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})

        # Button to start video streaming
    if st.button("Start Video Streaming"):
        if video_path and os.path.exists(video_path):
            st.session_state.stream_started = True  # Set stream_started flag
            global streaming
            streaming = True

            try:
                stream_video(video_path, json.dumps(zones), json.dumps(zones))
            except Exception as e:
                st.error("Error during streaming.")
                st.exception(e)
        else:
                st.error("Upload a valid video file first.")


    # Analytics Tab
    with st.sidebar:
        st.title("Analytics Tab")

if __name__ == "__main__":
    main()
