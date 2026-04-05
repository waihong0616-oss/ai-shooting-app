import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pathlib
import platform

# --- THE FIX: MUST BE AT THE TOP ---
# This forces Linux to understand Windows-style file paths saved in your model
plt_system = platform.system()
if plt_system != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Shooting App", layout="wide")
st.title("🎯 AI Shooting Detector")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # 1. Check if the file actually exists on GitHub
    if not os.path.exists('best.pt'):
        st.error("File 'best.pt' not found! Make sure it is in the root folder of your GitHub repo.")
        return None
    
    try:
        # 2. Load model with trust_repo to allow YOLOv5 scripts to run
        model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path='best.pt', 
            force_reload=True,
            trust_repo=True
        )
        return model
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None

model = load_model()

# --- APP INTERFACE ---
if model is not None:
    st.sidebar.header("Settings")
    conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.45)
    model.conf = conf_threshold 

    source = st.radio("Choose Source:", ("Image", "Video"))
    uploaded_file = st.file_uploader(f"Upload {source}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

    if uploaded_file is not None:
        if source == "Image":
            img = Image.open(uploaded_file)
            if st.button("Start Detection"):
                results = model(np.array(img))
                results.render()
                st.image(results.ims[0], use_container_width=True)
                st.dataframe(results.pandas().xyxy[0])

        elif source == "Video":
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                results.render()
                st_frame.image(results.ims[0], channels="RGB", use_container_width=True)
            
            vf.release()
            os.remove(tfile.name)
