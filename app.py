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

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Check if file exists first to avoid generic exceptions
    if not os.path.exists('best.pt'):
        st.error("Error: 'best.pt' not found in the root directory! Please check your GitHub upload.")
        return None
    
    try:
        # We load the model using torch.hub
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
    st.sidebar.header("Settings")
    confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.45)
    model.conf = confidence # Apply slider to model

    source = st.radio("Select Source:", ("Image", "Video"))
    uploaded_file = st.file_uploader(f"Upload {source}", type=['jpg', 'jpeg', 'png', 'mp4', 'mov'])

    if uploaded_file is not None:
        if source == "Image":
            img = Image.open(uploaded_file)
            # Convert PIL to OpenCV format
            img_array = np.array(img)
            
            if st.button("Run Detection"):
                results = model(img_array)
                results.render() # draws boxes
                
                st.subheader("Detection Result")
                # YOLOv5 render stores results in .ims (list of numpy arrays)
                st.image(results.ims[0], caption="Processed Image", use_container_width=True)
                st.write(results.pandas().xyxy[0]) 

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
                
                # Inference
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model(frame_rgb)
                results.render()
                
                # Display
                st_frame.image(results.ims[0], channels="RGB", use_container_width=True)
            
            vf.release()
            os.remove(tfile.name)
else:
    st.warning("Model is not loaded. Please fix the errors above.")
