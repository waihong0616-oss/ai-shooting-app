import streamlit as st
import torch
import cv2
import numpy as np
import math
from PIL import Image

# =========================
# TITLE
# =========================
st.set_page_config(page_title="AI Shooting Scoring", layout="centered")

st.title("🎯 AI Smart Shooting Scoring System")
st.write("Upload your shooting target image")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path='best.pt',
        trust_repo=True
    )
    model.conf = 0.25
    return model

model = load_model()

# =========================
# TARGET DETECTION
# =========================
def detect_target_contour(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        largest = max(contours, key=cv2.contourArea)

        if cv2.contourArea(largest) > 5000:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            radius = min(w, h) // 2
            return (cx, cy, radius)

    return None

# =========================
# PROCESS IMAGE
# =========================
def process_image(image):
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]

    # ===== TARGET =====
    target = detect_target_contour(frame)

    if target:
        cx, cy, R = target
    else:
        cx, cy = w//2, h//2
        R = 200

    center = (cx, cy)

    # ===== RINGS =====
    radii = [
        int(R * 0.2),
        int(R * 0.5),
        int(R * 0.7),
        int(R * 0.9),
        int(R * 1.1)
    ]

    scores = [10, 8, 6, 4, 2]
    colors = [(0,255,255),(255,0,255),(0,255,0),(0,128,255),(255,0,0)]

    overlay = frame.copy()
    for r, c in zip(radii, colors):
        cv2.circle(overlay, center, r, c, -1)

    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # ===== YOLO =====
    results = model(frame, size=1024)
    detections = results.xyxy[0]

    total_score = 0

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        bx, by = (x1+x2)//2, (y1+y2)//2

        dist = math.hypot(bx-cx, by-cy)

        score = 0
        for r, s in zip(radii, scores):
            if dist <= r:
                score = s
                break

        total_score += score

        cv2.circle(frame, (bx,by), 6, (0,255,0), -1)
        cv2.putText(frame, str(score), (bx-10,by-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"TOTAL: {total_score}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, total_score

# =========================
# UPLOAD UI
# =========================
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Original Image")

    if st.button("🚀 Process"):
        result, score = process_image(image)

        st.image(result, caption="Processed Result")
        st.success(f"🎯 Total Score: {score}")