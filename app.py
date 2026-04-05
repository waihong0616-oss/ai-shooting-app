import cv2
import numpy as np
import gradio as gr
import math

# =========================
# TARGET DETECTION
# =========================
def detect_target_circle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=300,
        param1=120,
        param2=50,
        minRadius=150,
        maxRadius=800
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0][0]
        return (cx, cy), r

    h, w = img.shape[:2]
    return (w//2, h//2), min(h, w)//3


# =========================
# BULLET DETECTION (ROBUST)
# =========================
def detect_bullets(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 🔥 Step 1: blur (smooth paper)
    blur = cv2.GaussianBlur(gray, (21,21), 0)

    # 🔥 Step 2: difference (highlight holes)
    diff = cv2.absdiff(gray, blur)

    # 🔥 Step 3: normalize
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 🔥 Step 4: threshold
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # 🔥 Step 5: morphology
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    bullets = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # ✅ tuned for your image
        if 50 < area < 5000:
            x, y, w, h = cv2.boundingRect(cnt)

            # filter weird shapes
            if 0.3 < w/h < 3:
                cx = x + w//2
                cy = y + h//2
                bullets.append((cx, cy))

    return bullets


# =========================
# CLUSTER
# =========================
def cluster_points(points, dist_thresh=20):
    clusters = []

    for p in points:
        found = False
        for c in clusters:
            if math.hypot(p[0]-c[0], p[1]-c[1]) < dist_thresh:
                c[0] = int((c[0] + p[0]) / 2)
                c[1] = int((c[1] + p[1]) / 2)
                found = True
                break

        if not found:
            clusters.append([p[0], p[1]])

    return [(c[0], c[1]) for c in clusters]


# =========================
# SCORE
# =========================
def calculate_score(img):
    center, base_radius = detect_target_circle(img)

    bullets = detect_bullets(img)
    bullets = cluster_points(bullets)

    radii = [
        int(base_radius * 0.2),
        int(base_radius * 0.4),
        int(base_radius * 0.6),
        int(base_radius * 0.8),
        int(base_radius * 1.0)
    ]

    scores = [10, 8, 6, 4, 2]

    total_score = 0

    overlay = img.copy()
    for r in radii:
        cv2.circle(overlay, center, r, (255,255,0), 2)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    for bx, by in bullets:
        dist = math.hypot(bx - center[0], by - center[1])

        if dist > base_radius * 1.1:
            continue

        score = 0
        for r, s in zip(radii, scores):
            if dist <= r:
                score = s
                break

        total_score += score

        cv2.circle(img, (bx, by), 7, (0,0,255), -1)
        cv2.putText(img, str(score), (bx-10, by-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    print("Detected bullets:", len(bullets))

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), total_score


# =========================
# GRADIO
# =========================
iface = gr.Interface(
    fn=calculate_score,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(), gr.Textbox()],
    title="🎯 AI Shooting Scoring (FINAL FIX)"
)

iface.launch()
