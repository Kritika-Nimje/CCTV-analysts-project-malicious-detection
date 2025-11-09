import streamlit as st
import os, time, cv2, csv, shutil, math, tempfile
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from google.cloud import storage
import plotly.express as px
import seaborn as sns

# ==================== STREAMLIT UI ====================

st.title("🎥 CCTV AI Analytics Dashboard")

# --- SECTION 1: Upload JSON or skip GCS ---
st.header("1️⃣ Google Cloud Setup (Optional)")

uploaded_json = st.file_uploader("Upload your Google Cloud Service Account JSON", type=["json"])
use_gcs = False
if uploaded_json:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_json.getbuffer())
        service_account_file = tmp.name
    BUCKET_NAME = st.text_input("Enter GCS Bucket Name", "cctv-motion-clips-kritika-nimje")
    use_gcs = st.button("Connect and Download from GCS")

# --- LOCAL FOLDERS ---
BASE_DIR = Path("./cctv_data")
VIDEO_DIR = BASE_DIR / "videos"
MOTION_DIR = BASE_DIR / "motion_frames"
OUTPUT_DIRS = {
    "weapon": BASE_DIR / "weapon_detections",
    "mask": BASE_DIR / "mask_detections",
    "person": BASE_DIR / "person_detections",
    "fire": BASE_DIR / "fire_detections",
    "posture": BASE_DIR / "posture_detections",
}
for d in OUTPUT_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
MOTION_DIR.mkdir(parents=True, exist_ok=True)

# --- GCS DOWNLOAD LOGIC ---
if use_gcs:
    try:
        st.info("Connecting to Google Cloud Storage...")
        client = storage.Client.from_service_account_json(service_account_file)
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs())
        video_blobs = [b for b in blobs if b.name.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
        for b in tqdm(video_blobs, desc="Downloading videos"):
            local_path = VIDEO_DIR / b.name.replace("/", "_")
            b.download_to_filename(local_path)
        st.success(f"✅ Downloaded {len(video_blobs)} videos from {BUCKET_NAME}")
    except Exception as e:
        st.error(f"GCS Error: {e}")

# --- Manual Video Upload ---
st.header("2️⃣ Upload Videos Manually (if not using GCS)")
uploaded_videos = st.file_uploader("Upload your CCTV videos", type=["mp4", "avi", "mov", "mkv"], accept_multiple_files=True)
if uploaded_videos:
    for video in uploaded_videos:
        with open(VIDEO_DIR / video.name, "wb") as f:
            f.write(video.getbuffer())
    st.success(f"✅ Uploaded {len(uploaded_videos)} video(s) to local folder.")

# ==================== MOTION DETECTION ====================
st.header("3️⃣ Motion Detection")

sensitivity = st.slider("Detection Sensitivity", 1000, 20000, 5000, 500)
skip_frames = st.slider("Frame Skip Interval", 1, 10, 2)
run_motion = st.button("▶️ Run Motion Detection")

def detect_motion(video_path, output_folder, sensitivity=5000, skip_frames=2):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return 0
    ret, prev = cap.read()
    if not ret: return 0
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if count % skip_frames != 0:
            count += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        if cv2.countNonZero(cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]) > sensitivity:
            fname = f"{Path(video_path).stem}_motion_{count:05d}.jpg"
            cv2.imwrite(str(output_folder / fname), frame)
            saved += 1
        prev_gray = gray
        count += 1
    cap.release()
    return saved

if run_motion:
    total_frames = 0
    videos = [VIDEO_DIR / f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    progress = st.progress(0)
    for i, vid in enumerate(videos):
        total_frames += detect_motion(vid, MOTION_DIR, sensitivity, skip_frames)
        progress.progress((i + 1) / len(videos))
    st.success(f"🎯 Total motion frames detected: {total_frames}")

# ==================== YOLO DETECTION ====================
st.header("4️⃣ YOLO Object Detection")

model_path = st.text_input("Enter YOLO model path", "yolov8n.pt")
run_yolo = st.button("🚀 Run YOLO Detections")

if run_yolo:
    model = YOLO(model_path)
    videos = [VIDEO_DIR / f for f in os.listdir(VIDEO_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    CONF_THRESH, IMG_SZ, DEVICE = 0.25, 640, 0
    st.info("Running detections...")
    total_dets = 0
    progress = st.progress(0)
    for i, vid in enumerate(videos):
        vname = Path(vid).stem
        out_dir = OUTPUT_DIRS["weapon"] / vname
        out_dir.mkdir(exist_ok=True, parents=True)
        cap = cv2.VideoCapture(str(vid))
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_idx % 2 != 0:
                frame_idx += 1
                continue
            results = model.predict(frame, imgsz=IMG_SZ, conf=CONF_THRESH, device=DEVICE, verbose=False)
            for r in results:
                annotated = r.plot()
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.jpg"), annotated)
                total_dets += len(r.boxes)
            frame_idx += 1
        cap.release()
        progress.progress((i + 1) / len(videos))
    st.success(f"✅ YOLO detections complete: {total_dets} objects found.")

# ==================== VISUALIZATION ====================
st.header("5️⃣ Detection Visualization")

if os.path.exists(MOTION_DIR):
    motion_imgs = list(MOTION_DIR.glob("*.jpg"))
    if motion_imgs:
        st.image(motion_imgs[:5], caption=[img.name for img in motion_imgs[:5]], width=250)

# ==================== SUMMARY STATS ====================
st.header("6️⃣ Summary Analytics")

data_summary = []
for label, folder in OUTPUT_DIRS.items():
    count = sum(len(files) for _, _, files in os.walk(folder))
    data_summary.append({"type": label, "count": count})
df_summary = pd.DataFrame(data_summary)

if not df_summary.empty:
    fig = px.bar(df_summary, x="type", y="count", color="type", title="📊 Total Detections per Module")
    st.plotly_chart(fig)
else:
    st.info("No detection data available yet.")

# ==================== CLEANUP ====================
if uploaded_json and os.path.exists(service_account_file):
    os.remove(service_account_file)
