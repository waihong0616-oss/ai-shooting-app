import pathlib
import platform
import sys

# --- THE FIX: MUST BE AT THE TOP ---
if platform.system() != 'Windows':
    pathlib.WindowsPath = pathlib.PosixPath

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
    if not os.path.exists('best.pt'):
        st.error("File 'best.pt' not found in your GitHub repo!")
        return None
    
    try:
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

# ... (Rest of your detection code)
