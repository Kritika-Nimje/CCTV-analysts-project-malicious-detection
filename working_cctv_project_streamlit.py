import streamlit as st
import os, time, cv2, csv, shutil, math, tempfile
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from google.cloud import storage
import json

# === STREAMLIT UI ===
st.title("🎥 CCTV Intelligent Video Analytics Dashboard")

# Step 1 — Upload Google Cloud service account JSON
st.header("1️⃣ Upload Google Cloud Service Account JSON")
uploaded_file = st.file_uploader("Upload your JSON key file", type=["json"])
if not uploaded_file:
    st.warning("Please upload a Google Cloud service account JSON file to continue.")
    st.stop()

# Save JSON temporarily
with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
    tmp.write(uploaded_file.getbuffer())
    service_account_file = tmp.name
st.success("✅ Service account loaded successfully.")

# Step 2 — Define configuration
BUCKET_NAME = st.text_input("Enter your GCS Bucket Name:", "cctv-motion-clips-kritika-nimje")
VIDEO_EXTS = ('.mp4', '.mov', '.avi', '.mkv')

LOCAL_VIDEO_DIR = "./gcs_videos"
os.makedirs(LOCAL_VIDEO_DIR, exist_ok=True)

# Step 3 — Download videos
if st.button("⬇️ Download Videos from GCS"):
    try:
        client = storage.Client.from_service_account_json(service_account_file)
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs())
        video_blobs = [b for b in blobs if b.name.lower().endswith(VIDEO_EXTS)]

        progress = st.progress(0)
        for i, b in enumerate(video_blobs):
            local_path = os.path.join(LOCAL_VIDEO_DIR, b.name.replace("/", os.sep))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path) and os.path.getsize(local_path) == int(b.size):
                continue
            b.download_to_filename(local_path)
            progress.progress((i + 1) / len(video_blobs))
        st.success(f"✅ Downloaded {len(video_blobs)} videos.")
    except Exception as e:
        st.error(f"Failed to download videos: {e}")

# Step 4 — Motion detection
st.header("2️⃣ Motion Detection")
sensitivity = st.slider("Motion sensitivity", 1000, 20000, 5000, 500)
skip_frames = st.slider("Skip frames", 1, 10, 2)

LOCAL_MOTION_DIR = "./motion_frames"
os.makedirs(LOCAL_MOTION_DIR, exist_ok=True)

def detect_motion(video_path, output_folder, sensitivity=5000, skip_frames=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    ret, prev = cap.read()
    if not ret:
        return 0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip_frames != 0:
            count += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        if cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]) > sensitivity:
            fname = f"{os.path.basename(video_path)}_motion_{count:05d}.jpg"
            cv2.imwrite(os.path.join(output_folder, fname), frame)
            saved += 1
        prev_gray = gray
        count += 1
    cap.release()
    return saved

if st.button("▶️ Run Motion Detection"):
    total_motion_frames = 0
    for root, _, files in os.walk(LOCAL_VIDEO_DIR):
        for f in files:
            if f.lower().endswith(VIDEO_EXTS):
                total_motion_frames += detect_motion(
                    os.path.join(root, f), LOCAL_MOTION_DIR, sensitivity, skip_frames
                )
    st.success(f"🎯 Total motion frames saved: {total_motion_frames}")

# Step 5 — Object Detection
st.header("3️⃣ Object Detection with YOLO")
CONF_THRESH, IMG_SZ, DEVICE = 0.25, 640, 0
model_path = st.text_input("Enter YOLO model path (default: yolov8n.pt)", "yolov8n.pt")

if st.button("🚀 Run YOLO Detections"):
    try:
        model = YOLO(model_path)
        videos = [
            os.path.join(root, f)
            for root, _, files in os.walk(LOCAL_VIDEO_DIR)
            for f in files if f.lower().endswith(VIDEO_EXTS)
        ]

        output_dir = "./detections"
        os.makedirs(output_dir, exist_ok=True)

        total_detections = 0
        progress = st.progress(0)
        for i, video_path in enumerate(videos):
            cap = cv2.VideoCapture(video_path)
            frame_idx = 0
            vname = Path(video_path).stem
            out_dir = os.path.join(output_dir, vname)
            os.makedirs(out_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % 2 != 0:
                    frame_idx += 1
                    continue
                results = model.predict(frame, imgsz=IMG_SZ, conf=CONF_THRESH, device=DEVICE, verbose=False)
                for r in results:
                    annotated = r.plot()
                    out_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.jpg")
                    cv2.imwrite(out_path, annotated)
                    total_detections += len(r.boxes)
                frame_idx += 1
            cap.release()
            progress.progress((i + 1) / len(videos))
        st.success(f"✅ Detection complete. Total detections: {total_detections}")
    except Exception as e:
        st.error(f"Detection failed: {e}")

# Step 6 — Visualization
st.header("4️⃣ Visualize Results")
if os.path.exists("./detections"):
    all_dets = []
    for root, _, files in os.walk("./detections"):
        all_dets.extend([os.path.join(root, f) for f in files if f.endswith(".jpg")])
    if all_dets:
        st.image(all_dets[:10], caption=[Path(f).name for f in all_dets[:10]], width=250)
else:
    st.info("No detections yet — run motion or YOLO detection first.")

# Clean up temporary JSON
if os.path.exists(service_account_file):
    os.remove(service_account_file)
