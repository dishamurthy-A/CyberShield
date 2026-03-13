
#DeepShield — Deepfake Detection Dashboard
#Save as: C:\Users\UMME\Downloads\deepfake_dashboard.py
#Run with: streamlit run deepfake_dashboard.py

import streamlit as st
import torch
import timm
import torch.nn as nn
import cv2
import numpy as np
import time
import tempfile
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="DeepShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-primary: #050a0f;
    --bg-secondary: #0a1520;
    --bg-card: #0d1f2d;
    --accent-cyan: #00d4ff;
    --accent-green: #00ff88;
    --accent-red: #ff3355;
    --accent-yellow: #ffcc00;
    --text-primary: #e8f4f8;
    --text-muted: #4a7a8a;
    --border: #1a3a4a;
}

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* Hide streamlit defaults */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 1rem 2rem !important; }

/* Hero header */
.hero-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    position: relative;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.5rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    color: var(--text-primary);
    margin: 0;
    text-shadow: 0 0 40px rgba(0, 212, 255, 0.3);
}
.hero-title span {
    color: var(--accent-cyan);
}
.hero-subtitle {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-muted);
    font-size: 0.85rem;
    letter-spacing: 0.2em;
    margin-top: 0.5rem;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
    margin: 1.5rem auto;
    max-width: 600px;
}

/* Cards */
.shield-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.shield-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
}

/* Verdict displays */
.verdict-fake {
    background: linear-gradient(135deg, #1a0510, #2d0818);
    border: 2px solid var(--accent-red);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(255, 51, 85, 0.2);
}
.verdict-real {
    background: linear-gradient(135deg, #051a0f, #082d18);
    border: 2px solid var(--accent-green);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(0, 255, 136, 0.2);
}
.verdict-uncertain {
    background: linear-gradient(135deg, #1a1505, #2d2208);
    border: 2px solid var(--accent-yellow);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 0 30px rgba(255, 204, 0, 0.2);
}
.verdict-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    margin: 0;
}
.verdict-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem;
    margin-top: 0.5rem;
    opacity: 0.8;
}

/* Confidence bar */
.conf-bar-container {
    background: var(--bg-secondary);
    border-radius: 4px;
    height: 8px;
    margin: 0.5rem 0;
    overflow: hidden;
}
.conf-bar-fill-fake {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #ff3355, #ff6680);
    transition: width 0.5s ease;
}
.conf-bar-fill-real {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #00ff88, #00ffaa);
    transition: width 0.5s ease;
}

/* Model breakdown */
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.4rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.model-name {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-muted);
}
.model-score {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 1rem;
}

/* Warning badge */
.badge-warning {
    background: rgba(255, 204, 0, 0.1);
    border: 1px solid var(--accent-yellow);
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    color: var(--accent-yellow);
    font-family: 'Share Tech Mono', monospace;
    display: inline-block;
    margin: 0.2rem;
}
.badge-info {
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid var(--accent-cyan);
    border-radius: 6px;
    padding: 0.3rem 0.8rem;
    font-size: 0.75rem;
    color: var(--accent-cyan);
    font-family: 'Share Tech Mono', monospace;
    display: inline-block;
    margin: 0.2rem;
}

/* Upload area */
.upload-hint {
    text-align: center;
    color: var(--text-muted);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    padding: 1rem;
    border: 1px dashed var(--border);
    border-radius: 8px;
    margin-bottom: 1rem;
}

/* Section headers */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: var(--accent-cyan);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    font-size: 1rem !important;
    border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border-bottom: 2px solid var(--accent-cyan) !important;
}
.stSlider label {
    color: var(--text-muted) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.8rem !important;
}
div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important;
    border-radius: 8px !important;
}
.stButton button {
    background: linear-gradient(135deg, #00d4ff22, #00d4ff11) !important;
    border: 1px solid var(--accent-cyan) !important;
    color: var(--accent-cyan) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    border-radius: 6px !important;
    width: 100% !important;
}
.stButton button:hover {
    background: linear-gradient(135deg, #00d4ff33, #00d4ff22) !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.3) !important;
}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────
DOWNLOADS = Path(r'C:\Users\UMME\Downloads')
DEVICE    = torch.device('cpu')

# ── MODEL ─────────────────────────────────────────────────────
class EfficientNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=2)
    def forward(self, x): return self.efficientnet(x)

@st.cache_resource
def load_models():
    def _load(path):
        m    = EfficientNetDetector().to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        sd   = ckpt.get('state_dict') or ckpt
        m.load_state_dict(sd, strict=True)
        m.eval()
        return m
    models = {
        'Final (v3)':    _load(DOWNLOADS / 'EfficientNet_final.pth'),
        'Finetuned (v2)':_load(DOWNLOADS / 'EfficientNet_finetuned_v2.pth'),
        'Adversarial':   _load(DOWNLOADS / 'EfficientNet_adversarial.pth'),
        'Video':         _load(DOWNLOADS / 'EfficientNet_video.pth'),
    }
    return models

# ── TRANSFORMS ────────────────────────────────────────────────
val_tf = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── FACE DETECTOR ─────────────────────────────────────────────
@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    cascade = load_face_cascade()
    faces  = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    if len(faces) == 0:
        faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(20,20))
    return faces

def crop_face(img_pil, face):
    x, y, w, h = face
    pad_x = int(w*0.2)
    pad_y = int(h*0.2)
    x1 = max(0, x-pad_x)
    y1 = max(0, y-pad_y)
    x2 = min(img_pil.width,  x+w+pad_x)
    y2 = min(img_pil.height, y+h+pad_y)
    return img_pil.crop((x1,y1,x2,y2))

# ── IMAGE QUALITY CHECK ───────────────────────────────────────
def check_image_quality(img_pil):
    warnings = []
    img_array = np.array(img_pil.convert('L'))

    # Brightness
    brightness = img_array.mean()
    if brightness < 60:
        warnings.append(" Dark image — results may be less accurate")
    elif brightness > 220:
        warnings.append(" Overexposed image — results may be less accurate")

    # Blurriness (Laplacian variance)
    blur_score = cv2.Laplacian(img_array, cv2.CV_64F).var()
    if blur_score < 50:
        warnings.append(" Blurry image — results may be less accurate")

    # Resolution
    if img_pil.width < 100 or img_pil.height < 100:
        warnings.append(" Very low resolution image")

    return warnings

# ── PREDICT ───────────────────────────────────────────────────
def predict_image(img_pil, models, threshold=0.5):
    tensor = val_tf(img_pil).unsqueeze(0).to(DEVICE)
    scores = {}
    with torch.no_grad():
        for name, model in models.items():
            probs = torch.softmax(model(tensor), dim=1)[0]
            scores[name] = {
                'real': probs[0].item(),
                'fake': probs[1].item()
            }

    # Weighted ensemble
    weights = {
        'Final (v3)':     0.50,
        'Finetuned (v2)': 0.30,
        'Adversarial':    0.10,
        'Video':          0.10,
    }
    fake_prob = sum(scores[n]['fake'] * weights[n]
                    for n in scores)
    real_prob = 1 - fake_prob
    verdict   = 'FAKE' if fake_prob > threshold else 'REAL'

    return verdict, fake_prob, real_prob, scores

# ── GRADCAM HEATMAP ───────────────────────────────────────────
def generate_heatmap(img_pil, model):
    try:
        img_array = np.array(img_pil.resize((224,224)))
        tensor    = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        out  = model(tensor)
        pred = out.argmax(1)
        out[0, pred].backward()

        grads   = tensor.grad.data.abs().squeeze(0)
        heatmap = grads.mean(0).numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        overlay = np.array(img_pil.resize((224,224)))
        blended = cv2.addWeighted(overlay, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(blended)
    except:
        return img_pil.resize((224,224))

# ── DRAW FACE BOXES ───────────────────────────────────────────
def draw_face_boxes(img_pil, faces, verdicts):
    img_draw = img_pil.copy()
    draw     = ImageDraw.Draw(img_draw)
    colors   = {'FAKE': '#ff3355', 'REAL': '#00ff88', 'UNCERTAIN': '#ffcc00'}

    for i, (face, verdict) in enumerate(zip(faces, verdicts)):
        x, y, w, h = face
        color = colors.get(verdict, '#00d4ff')
        # Draw box
        for t in range(3):
            draw.rectangle([x-t, y-t, x+w+t, y+h+t], outline=color)
        # Label
        draw.rectangle([x, y-22, x+w, y], fill=color)
        draw.text((x+4, y-20), f"Face {i+1}: {verdict}", fill='black')

    return img_draw

# ── VIDEO ANALYSIS ────────────────────────────────────────────
def analyze_video(video_path, models, threshold=0.5, max_frames=15, frame_skip=10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0

    cascade     = load_face_cascade()
    frame_data  = []
    frame_count = 0
    extracted   = 0
    thumbnails  = []

    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) == 0:
            faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(20,20))

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            pad_x = int(w*0.2)
            pad_y = int(h*0.2)
            x1 = max(0, x-pad_x)
            y1 = max(0, y-pad_y)
            x2 = min(frame.shape[1], x+w+pad_x)
            y2 = min(frame.shape[0], y+h+pad_y)
            crop = rgb[y1:y2, x1:x2]
        else:
            hh, ww = rgb.shape[:2]
            s  = min(hh, ww)
            y1 = (hh-s)//2
            x1 = (ww-s)//2
            crop = rgb[y1:y1+s, x1:x1+s]

        img_pil = Image.fromarray(crop)

        # Predict
        weights = {'Final (v3)': 0.20, 'Finetuned (v2)': 0.10,
                   'Adversarial': 0.10, 'Video': 0.60}
        tensor  = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake_prob = sum(
                torch.softmax(m(tensor), dim=1)[0][1].item() * w
                for (n, m), w in zip(models.items(), weights.values())
            )

        frame_data.append({
            'frame':     frame_count,
            'time':      frame_count / fps if fps > 0 else extracted,
            'fake_prob': fake_prob,
            'verdict':   'FAKE' if fake_prob > threshold else 'REAL'
        })

        # Save thumbnail every 3 frames
        if extracted % 3 == 0:
            thumb = Image.fromarray(rgb).resize((120, 68))
            thumbnails.append((thumb, fake_prob))

        extracted += 1

    cap.release()

    if not frame_data:
        return None

    avg_fake    = np.mean([f['fake_prob'] for f in frame_data])
    fake_ratio  = np.mean([f['verdict'] == 'FAKE' for f in frame_data])
    verdict     = 'FAKE' if fake_ratio >= 0.4 else 'REAL'

    return {
        'verdict':     verdict,
        'avg_fake':    avg_fake,
        'fake_ratio':  fake_ratio,
        'frame_data':  frame_data,
        'thumbnails':  thumbnails,
        'duration':    duration,
        'total_frames': total_frames,
        'fps':         fps,
    }

# ── PLOT FRAME TIMELINE ───────────────────────────────────────
def plot_timeline(frame_data, threshold):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#0d1f2d')
    ax.set_facecolor('#0a1520')

    times      = [f['time'] for f in frame_data]
    fake_probs = [f['fake_prob'] for f in frame_data]

    # Fill areas
    ax.fill_between(times, fake_probs, threshold,
                    where=[p > threshold for p in fake_probs],
                    color='#ff3355', alpha=0.3, label='Fake region')
    ax.fill_between(times, fake_probs, threshold,
                    where=[p <= threshold for p in fake_probs],
                    color='#00ff88', alpha=0.3, label='Real region')

    ax.plot(times, fake_probs, color='#00d4ff',
            linewidth=2, zorder=5)
    ax.axhline(y=threshold, color='#ffcc00',
               linestyle='--', linewidth=1, alpha=0.7, label='Threshold')

    ax.set_xlim(min(times), max(times))
    ax.set_ylim(0, 1)
    ax.set_xlabel('Time (seconds)', color='#4a7a8a', fontsize=8)
    ax.set_ylabel('Fake Probability', color='#4a7a8a', fontsize=8)
    ax.tick_params(colors='#4a7a8a', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a3a4a')

    ax.legend(loc='upper right', fontsize=7,
              facecolor='#0d1f2d', edgecolor='#1a3a4a',
              labelcolor='#e8f4f8')

    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">DEEP<span>SHIELD</span></h1>
    <p class="hero-subtitle">// AI-POWERED DEEPFAKE DETECTION SYSTEM //</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# Load models
with st.spinner("Initializing detection models..."):
    try:
        models = load_models()
        st.markdown('<p style="text-align:center; font-family: Share Tech Mono; font-size:0.75rem; color:#00ff88;">✓ 4 MODELS LOADED — SYSTEM READY</p>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["📸  IMAGE ANALYSIS", "🎥  VIDEO ANALYSIS"])

# ══════════════════════════════════════════════════════════════
# IMAGE TAB
# ══════════════════════════════════════════════════════════════
with tab_img:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">UPLOAD IMAGE</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop image here",
            type=['jpg','jpeg','png','bmp','webp'],
            key='img_upload',
            label_visibility='collapsed'
        )

        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, use_container_width=True,
                     caption="Uploaded Image")

            # Settings
            st.markdown('<div class="section-header">DETECTION SETTINGS</div>',
                       unsafe_allow_html=True)

            sensitivity = st.select_slider(
                "SENSITIVITY",
                options=["LOW", "MEDIUM", "HIGH"],
                value="MEDIUM"
            )
            threshold_map = {"LOW": 0.55, "MEDIUM": 0.40, "HIGH": 0.25}
            threshold = threshold_map[sensitivity]

            show_heatmap = st.checkbox("🔥 Show GradCAM Heatmap", value=True)
            show_faces   = st.checkbox("🔍 Show Face Detection", value=True)

            analyze_btn = st.button("⚡ ANALYZE IMAGE", key='analyze_img')

    with col_right:
        st.markdown('<div class="section-header">ANALYSIS RESULTS</div>',
                   unsafe_allow_html=True)

        if uploaded and analyze_btn:
            with st.spinner("Analyzing..."):
                # Quality check
                quality_warnings = check_image_quality(img)

                # Face detection
                faces = detect_faces(img)

                # Predict
                if len(faces) > 0:
                    # Predict on largest face
                    largest_face = max(faces, key=lambda f: f[2]*f[3])
                    face_img     = crop_face(img, largest_face)
                    verdict, fake_prob, real_prob, scores = predict_image(
                        face_img, models, threshold)
                    face_verdicts = [verdict]
                else:
                    verdict, fake_prob, real_prob, scores = predict_image(
                        img, models, threshold)
                    face_verdicts = []

                # Verdict display
                conf = max(fake_prob, real_prob) * 100
                if verdict == 'FAKE':
                    card_class = 'verdict-fake'
                    color      = '#ff3355'
                    icon       = '🔴'
                else:
                    card_class = 'verdict-real'
                    color      = '#00ff88'
                    icon       = '🟢'

                st.markdown(f"""
                <div class="{card_class}">
                    <p class="verdict-label" style="color:{color}">
                        {icon} {verdict}
                    </p>
                    <p class="verdict-conf">
                        Confidence: {conf:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Confidence bars
                st.markdown(f"""
                <div style="margin-bottom:0.3rem">
                    <div style="display:flex;justify-content:space-between;
                                font-family:Share Tech Mono;font-size:0.75rem;
                                color:#4a7a8a;margin-bottom:3px">
                        <span>FAKE PROBABILITY</span>
                        <span style="color:#ff3355">{fake_prob*100:.1f}%</span>
                    </div>
                    <div class="conf-bar-container">
                        <div class="conf-bar-fill-fake"
                             style="width:{fake_prob*100:.1f}%"></div>
                    </div>
                </div>
                <div>
                    <div style="display:flex;justify-content:space-between;
                                font-family:Share Tech Mono;font-size:0.75rem;
                                color:#4a7a8a;margin-bottom:3px">
                        <span>REAL PROBABILITY</span>
                        <span style="color:#00ff88">{real_prob*100:.1f}%</span>
                    </div>
                    <div class="conf-bar-container">
                        <div class="conf-bar-fill-real"
                             style="width:{real_prob*100:.1f}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Quality warnings
                if quality_warnings:
                    for w in quality_warnings:
                        st.markdown(f'<span class="badge-warning">{w}</span>',
                                   unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                # Face info
                if len(faces) > 0:
                    st.markdown(f'<span class="badge-info">✓ {len(faces)} face(s) detected</span>',
                               unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-warning">⚠️ No face detected — full image analyzed</span>',
                               unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Model breakdown
                st.markdown('<div class="section-header">MODEL BREAKDOWN</div>',
                           unsafe_allow_html=True)

                for model_name, score in scores.items():
                    fp = score['fake'] * 100
                    color_m = '#ff3355' if fp > 50 else '#00ff88'
                    st.markdown(f"""
                    <div class="model-row">
                        <span class="model-name">{model_name}</span>
                        <span class="model-score" style="color:{color_m}">
                            {fp:.1f}% fake
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Face boxes + heatmap
                vis_col1, vis_col2 = st.columns(2)

                with vis_col1:
                    if show_faces and len(faces) > 0:
                        st.markdown('<p style="font-family:Share Tech Mono;font-size:0.7rem;color:#4a7a8a">FACE DETECTION</p>',
                                   unsafe_allow_html=True)
                        boxed = draw_face_boxes(img, faces, face_verdicts * len(faces))
                        st.image(boxed, use_container_width=True)

                with vis_col2:
                    if show_heatmap:
                        st.markdown('<p style="font-family:Share Tech Mono;font-size:0.7rem;color:#4a7a8a">GRADCAM HEATMAP</p>',
                                   unsafe_allow_html=True)
                        heatmap_img = generate_heatmap(
                            face_img if len(faces) > 0 else img,
                            models['Final (v3)'])
                        st.image(heatmap_img, use_container_width=True)

        elif not uploaded:
            st.markdown("""
            <div class="upload-hint">
                <p style="font-size:2rem;margin:0">🛡️</p>
                <p>Upload an image to begin analysis</p>
                <p style="font-size:0.7rem;opacity:0.5">
                    Supports JPG, PNG, BMP, WEBP
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# VIDEO TAB
# ══════════════════════════════════════════════════════════════
with tab_vid:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">UPLOAD VIDEO</div>',
                   unsafe_allow_html=True)

        vid_uploaded = st.file_uploader(
            "Drop video here",
            type=['mp4','avi','mov'],
            key='vid_upload',
            label_visibility='collapsed'
        )

        if vid_uploaded:
            st.video(vid_uploaded)

            st.markdown('<div class="section-header">DETECTION SETTINGS</div>',
                       unsafe_allow_html=True)

            vid_sensitivity = st.select_slider(
                "SENSITIVITY",
                options=["LOW", "MEDIUM", "HIGH"],
                value="MEDIUM",
                key='vid_sens'
            )
            vid_threshold = threshold_map[vid_sensitivity]

            max_frames = st.slider(
                "MAX FRAMES TO ANALYZE",
                min_value=5, max_value=30, value=15, step=5
            )

            analyze_vid_btn = st.button("⚡ ANALYZE VIDEO", key='analyze_vid')

    with col_right:
        st.markdown('<div class="section-header">ANALYSIS RESULTS</div>',
                   unsafe_allow_html=True)

        if vid_uploaded and analyze_vid_btn:
            # Save to temp file
            with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(vid_uploaded.name).suffix) as tmp:
                tmp.write(vid_uploaded.read())
                tmp_path = tmp.name

            progress = st.progress(0, text="Extracting frames...")

            with st.spinner("Analyzing video frames..."):
                result = analyze_video(
                    tmp_path, models,
                    threshold=vid_threshold,
                    max_frames=max_frames,
                    frame_skip=8
                )
                progress.progress(100, text="Analysis complete!")

            os.unlink(tmp_path)

            if result:
                verdict   = result['verdict']
                fake_prob = result['avg_fake']
                real_prob = 1 - fake_prob
                conf      = max(fake_prob, real_prob) * 100

                if verdict == 'FAKE':
                    card_class = 'verdict-fake'
                    color      = '#ff3355'
                    icon       = '🔴'
                else:
                    card_class = 'verdict-real'
                    color      = '#00ff88'
                    icon       = '🟢'

                st.markdown(f"""
                <div class="{card_class}">
                    <p class="verdict-label" style="color:{color}">
                        {icon} {verdict}
                    </p>
                    <p class="verdict-conf">
                        Confidence: {conf:.1f}% |
                        Fake frames: {result['fake_ratio']*100:.0f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Stats row
                stat1, stat2, stat3 = st.columns(3)
                with stat1:
                    st.markdown(f"""
                    <div style="text-align:center;background:#0d1f2d;
                                border:1px solid #1a3a4a;border-radius:8px;padding:0.8rem">
                        <p style="font-family:Share Tech Mono;font-size:0.7rem;
                                  color:#4a7a8a;margin:0">DURATION</p>
                        <p style="font-family:Rajdhani;font-size:1.3rem;
                                  font-weight:700;color:#00d4ff;margin:0">
                            {result['duration']:.1f}s
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with stat2:
                    st.markdown(f"""
                    <div style="text-align:center;background:#0d1f2d;
                                border:1px solid #1a3a4a;border-radius:8px;padding:0.8rem">
                        <p style="font-family:Share Tech Mono;font-size:0.7rem;
                                  color:#4a7a8a;margin:0">FRAMES ANALYZED</p>
                        <p style="font-family:Rajdhani;font-size:1.3rem;
                                  font-weight:700;color:#00d4ff;margin:0">
                            {len(result['frame_data'])}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                with stat3:
                    fake_color = '#ff3355' if result['fake_ratio'] > 0.4 else '#00ff88'
                    st.markdown(f"""
                    <div style="text-align:center;background:#0d1f2d;
                                border:1px solid #1a3a4a;border-radius:8px;padding:0.8rem">
                        <p style="font-family:Share Tech Mono;font-size:0.7rem;
                                  color:#4a7a8a;margin:0">FAKE FRAMES</p>
                        <p style="font-family:Rajdhani;font-size:1.3rem;
                                  font-weight:700;color:{fake_color};margin:0">
                            {result['fake_ratio']*100:.0f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Timeline chart
                st.markdown('<div class="section-header">FRAME-BY-FRAME TIMELINE</div>',
                           unsafe_allow_html=True)
                fig = plot_timeline(result['frame_data'], vid_threshold)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Frame thumbnails
                if result['thumbnails']:
                    st.markdown('<div class="section-header">SAMPLE FRAMES</div>',
                               unsafe_allow_html=True)
                    thumb_cols = st.columns(min(len(result['thumbnails']), 5))
                    for i, (thumb, fp) in enumerate(result['thumbnails'][:5]):
                        with thumb_cols[i]:
                            border = '#ff3355' if fp > vid_threshold else '#00ff88'
                            st.image(thumb, use_container_width=True)
                            st.markdown(
                                f'<p style="text-align:center;font-family:Share Tech Mono;'
                                f'font-size:0.65rem;color:{border};margin:0">'
                                f'{"FAKE" if fp > vid_threshold else "REAL"} {fp*100:.0f}%</p>',
                                unsafe_allow_html=True)
            else:
                st.error("Could not analyze video — check file format")

        elif not vid_uploaded:
            st.markdown("""
            <div class="upload-hint">
                <p style="font-size:2rem;margin:0">🎥</p>
                <p>Upload a video to begin analysis</p>
                <p style="font-size:0.7rem;opacity:0.5">
                    Supports MP4, AVI, MOV
                </p>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding:1.5rem;
            border-top:1px solid #1a3a4a">
    <p style="font-family:Share Tech Mono;font-size:0.7rem;color:#4a7a8a;margin:0">
        DEEPSHIELD v1.0 — EfficientNet-B0 + FaceForensics++ |
        IEEE Research Project 2026
    </p>
</div>
""", unsafe_allow_html=True)
