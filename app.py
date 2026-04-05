import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Shooting App", layout="wide")
st.title("🎯 AI Shooting Detector")
st.write("Upload an image or video to detect targets using YOLOv5.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Loading the model from the local 'best.pt' file
    # We use ultralytics/yolov5 as the base repository
    model = torch.hub.load(
        'ultralytics/yolov5', 
        'custom', 
        path='best.pt', 
        force_reload=True
    )
    return model

model = load_model()

# --- SIDEBAR ---
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)

# --- UPLOAD SECTION ---
source = st.radio("Select Source:", ("Image", "Video"))
uploaded_file = st.file_uploader(f"Upload {source}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

if uploaded_file is not None:
    if source == "Image":
        # Process Image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Run Detection"):
            results = model(img)
            results.render()  # updates results.ims with boxes and labels
            
            st.subheader("Detection Result")
            st.image(results.ims[0], caption="Processed Image", use_container_width=True)
            st.write(results.pandas().xyxy[0])  # Show table of detections

    elif source == "Video":
        # Process Video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        vf = cv2.VideoCapture(tfile.name)
        
        st.write("Processing video... this may take a moment.")
        st_frame = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inference
            results = model(frame_rgb)
            results.render()
            
            # Display frame
            st_frame.image(results.ims[0], channels="RGB", use_container_width=True)
        
        vf.release()
        os.remove(tfile.name)
