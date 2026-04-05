import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import pathlib
import platform  # Added this to fix the NameError

# --- CRITICAL FIX FOR WINDOWS-TRAINED MODELS ---
# This rebinds WindowsPath to PosixPath so the model can load on Linux/Streamlit Cloud
plt_system = platform.system()
if plt_system != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Shooting App", layout="wide")
st.title("🎯 AI Shooting Detector")
st.write("Upload an image or video to detect targets using your custom YOLOv5 model.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Verify file exists to prevent generic loading errors
    if not os.path.exists('best.pt'):
        st.error("File 'best.pt' not found in the root directory of your GitHub repo!")
        return None
    
    try:
        # Load the custom model using torch.hub
        # force_reload=True ensures the latest YOLOv5 scripts are pulled
        model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            path='best.pt', 
            force_reload=True,
            trust_repo=True
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# --- APP LOGIC ---
if model is not None:
    st.sidebar.header("Detection Settings")
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
    model.conf = conf_threshold 

    source = st.radio("Select Source:", ("Image", "Video"))
    uploaded_file = st.file_uploader(f"Upload {source}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

    if uploaded_file is not None:
        if source == "Image":
            img = Image.open(uploaded_file)
            img_array = np.array(img)
            
            if st.button("Run Detection"):
                with st.spinner("Analyzing image..."):
                    results = model(img_array)
                    results.render()  
                    
                    st.subheader("Result")
                    st.image(results.ims[0], caption="Processed Image", use_container_width=True)
                    st.write("Detections found:")
                    st.dataframe(results.pandas().xyxy[0])

        elif source == "Video":
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vf = cv2.VideoCapture(tfile.name)
            
            st.info("Processing video... frames will appear below.")
            st_frame = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                # Convert BGR (OpenCV) to RGB (Streamlit)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                results.render()
                
                st_frame.image(results.ims[0], channels="RGB", use_container_width=True)
            
            vf.release()
            os.remove(tfile.name)
else:
    st.warning("Model is not loaded. Ensure 'best.pt' is uploaded to GitHub correctly.")
