# ============================================================
# CyberShield — Combined Dashboard
# Save as: C:\Users\UMME\Downloads\CyberShield\main_app.py
#
# Folder structure needed:
# CyberShield\
#   main_app.py              ← this file
#   deepfake_dashboard.py    ← your existing file (copy here)
#   modules\
#     email_detector.py      ← Disha's file
#   models\                  ← Disha's model files
#     bert_lstm_phishing.pt
#     xgb_model_v2.pkl
#     rf_model_v2.pkl
#     tfidf_v2.pkl
#     scaler_v2.pkl
#     feature_columns_v2.pkl
#     tokenizer\
# ============================================================

import streamlit as st
import torch
import timm
import torch.nn as nn
import cv2
import numpy as np
import time
import tempfile
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add modules to path
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'modules'))

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-primary:   #050a0f;
    --bg-secondary: #0a1520;
    --bg-card:      #0d1f2d;
    --cyan:         #00d4ff;
    --green:        #00ff88;
    --red:          #ff3355;
    --yellow:       #ffcc00;
    --orange:       #ff8c00;
    --text:         #e8f4f8;
    --muted:        #4a7a8a;
    --border:       #1a3a4a;
}

* { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-primary); color: var(--text); }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.block-container { padding: 1rem 2rem !important; }

/* Hero */
.hero { text-align: center; padding: 1.5rem 0 0.5rem; }
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3.8rem; font-weight: 700;
    letter-spacing: 0.2em; color: var(--text); margin: 0;
    text-shadow: 0 0 40px rgba(0,212,255,0.3);
}
.hero-title span { color: var(--cyan); }
.hero-sub {
    font-family: 'Share Tech Mono', monospace;
    color: var(--muted); font-size: 0.8rem;
    letter-spacing: 0.25em; margin-top: 0.4rem;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    margin: 1rem auto; max-width: 700px;
}

/* Cards */
.card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem;
    position: relative; overflow: hidden;
}
.card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--cyan), transparent);
}

/* Verdict boxes */
.verdict-fake {
    background: linear-gradient(135deg, #1a0510, #2d0818);
    border: 2px solid var(--red); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 30px rgba(255,51,85,0.2);
}
.verdict-real {
    background: linear-gradient(135deg, #051a0f, #082d18);
    border: 2px solid var(--green); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 30px rgba(0,255,136,0.2);
}
.verdict-phishing {
    background: linear-gradient(135deg, #1a0510, #2d0818);
    border: 2px solid var(--red); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 30px rgba(255,51,85,0.2);
}
.verdict-legit {
    background: linear-gradient(135deg, #051a0f, #082d18);
    border: 2px solid var(--green); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 30px rgba(0,255,136,0.2);
}
.verdict-medium {
    background: linear-gradient(135deg, #1a1200, #2d1f00);
    border: 2px solid var(--yellow); border-radius: 12px;
    padding: 1.5rem; text-align: center;
    box-shadow: 0 0 30px rgba(255,204,0,0.2);
}
.verdict-label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    letter-spacing: 0.1em; margin: 0;
}
.verdict-conf {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem; margin-top: 0.4rem; opacity: 0.85;
}

/* Section headers */
.sec-hdr {
    font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;
    font-weight: 600; letter-spacing: 0.18em;
    color: var(--cyan); text-transform: uppercase;
    margin-bottom: 0.6rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
}

/* Confidence bars */
.bar-wrap { background: var(--bg-secondary); border-radius: 4px; height: 7px; margin: 3px 0; overflow: hidden; }
.bar-fake  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #ff3355, #ff6680); }
.bar-real  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #00ff88, #00ffaa); }
.bar-warn  { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #ffcc00, #ffe066); }

/* Badges */
.badge-warn {
    background: rgba(255,204,0,0.1); border: 1px solid var(--yellow);
    border-radius: 6px; padding: 0.25rem 0.7rem; font-size: 0.72rem;
    color: var(--yellow); font-family: 'Share Tech Mono', monospace;
    display: inline-block; margin: 0.15rem;
}
.badge-info {
    background: rgba(0,212,255,0.1); border: 1px solid var(--cyan);
    border-radius: 6px; padding: 0.25rem 0.7rem; font-size: 0.72rem;
    color: var(--cyan); font-family: 'Share Tech Mono', monospace;
    display: inline-block; margin: 0.15rem;
}
.badge-ok {
    background: rgba(0,255,136,0.1); border: 1px solid var(--green);
    border-radius: 6px; padding: 0.25rem 0.7rem; font-size: 0.72rem;
    color: var(--green); font-family: 'Share Tech Mono', monospace;
    display: inline-block; margin: 0.15rem;
}
.badge-danger {
    background: rgba(255,51,85,0.1); border: 1px solid var(--red);
    border-radius: 6px; padding: 0.25rem 0.7rem; font-size: 0.72rem;
    color: var(--red); font-family: 'Share Tech Mono', monospace;
    display: inline-block; margin: 0.15rem;
}

/* Model breakdown row */
.model-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.35rem 0; border-bottom: 1px solid var(--border); font-size: 0.82rem;
}
.model-name { font-family: 'Share Tech Mono', monospace; color: var(--muted); }

/* Flag items */
.flag-red {
    background: rgba(255,51,85,0.1); border-left: 3px solid var(--red);
    border-radius: 4px; padding: 0.4rem 0.7rem; margin: 0.2rem 0;
    font-size: 0.82rem; color: #ffaaaa;
}
.flag-green {
    background: rgba(0,255,136,0.1); border-left: 3px solid var(--green);
    border-radius: 4px; padding: 0.4rem 0.7rem; margin: 0.2rem 0;
    font-size: 0.82rem; color: #aaffcc;
}

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-radius: 8px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: var(--muted) !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; font-size: 1rem !important;
    border-radius: 6px !important; padding: 0.5rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important; color: var(--cyan) !important;
    border-bottom: 2px solid var(--cyan) !important;
}
.stButton button {
    background: linear-gradient(135deg, #00d4ff22, #00d4ff11) !important;
    border: 1px solid var(--cyan) !important; color: var(--cyan) !important;
    font-family: 'Rajdhani', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.1em !important; border-radius: 6px !important;
    width: 100% !important;
}
.stTextArea textarea {
    background: var(--bg-card) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.82rem !important;
}
div[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important; border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PERFORMANCE GRAPH FUNCTIONS (static — based on training results)
# ══════════════════════════════════════════════════════════════

def _style_ax(ax, title):
    ax.set_facecolor('#0a1520')
    ax.set_title(title, color='#00d4ff', fontsize=9, fontweight='bold', pad=8)
    ax.tick_params(colors='#4a7a8a', labelsize=7)
    ax.set_ylabel('Score (%)', color='#4a7a8a', fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#1a3a4a')

def _bar_labels(ax, bars, color='white', fontsize=7):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.4,
                f'{h:.1f}%', ha='center', va='bottom',
                color=color, fontsize=fontsize, fontweight='bold')

def plot_deepfake_graphs():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.patch.set_facecolor('#0d1f2d')

    # Graph 1 — Cross-dataset accuracy vs published
    ax = axes[0]
    _style_ax(ax, 'Cross-Dataset Accuracy vs Published Methods')
    datasets = ['Midjourney', 'StyleGAN', 'CelebDF\n(Unseen)', 'FF++ Video', 'DFD Video\n(Unseen)']
    ours     = [99.90, 97.30, 81.50, 95.50, 72.86]
    pub      = [None,  None,  51.20, None,  51.20]
    x = np.arange(len(datasets)); w = 0.35
    b1 = ax.bar(x-w/2, ours, w, color='#00d4ff', alpha=0.85, edgecolor='#1a3a4a', label='DeepShield (Ours)')
    pub_v = [v if v else 0 for v in pub]
    b2 = ax.bar(x+w/2, pub_v, w, color='#ff3355', alpha=0.75, edgecolor='#1a3a4a', label='Best Published')
    for bar, val in zip(b2, pub):
        if not val: bar.set_visible(False)
    _bar_labels(ax, b1, color='#00d4ff')
    for bar, val in zip(b2, pub):
        if val:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f'{val}%', ha='center', va='bottom', color='#ff3355', fontsize=7, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=7)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=6, facecolor='#0d1f2d', edgecolor='#1a3a4a', labelcolor='#e8f4f8', loc='lower right')
    ax.annotate('+30pp', xy=(x[2]+w/2, 81.5), xytext=(x[2]+w/2+0.4, 68),
                fontsize=7, color='#00ff88', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.2))
    ax.annotate('+21pp', xy=(x[4]+w/2, 72.86), xytext=(x[4]+w/2+0.4, 59),
                fontsize=7, color='#00ff88', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#00ff88', lw=1.2))

    # Graph 2 — Architecture comparison
    ax2 = axes[1]
    _style_ax(ax2, 'Architecture Comparison')
    arch = ['EfficientNet-B0\n(Ours)', 'ViT', 'Custom CNN']
    acc_v = [93.50, 85.00, 78.00]; par_v = [4.01, 86.00, 25.00]
    x2 = np.arange(len(arch)); w2 = 0.35
    b3 = ax2.bar(x2-w2/2, acc_v, w2, color='#00d4ff', alpha=0.85, edgecolor='#1a3a4a', label='Accuracy (%)')
    b4 = ax2.bar(x2+w2/2, par_v, w2, color='#ff3355', alpha=0.75, edgecolor='#1a3a4a', label='Params (M)')
    _bar_labels(ax2, b3, color='#00d4ff')
    for bar, val in zip(b4, par_v):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                 f'{val}M', ha='center', va='bottom', color='#ff9999', fontsize=6, fontweight='bold')
    ax2.set_xticks(x2); ax2.set_xticklabels(arch, fontsize=7)
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=6, facecolor='#0d1f2d', edgecolor='#1a3a4a', labelcolor='#e8f4f8')
    ax2.axvspan(-0.5, 0.5, alpha=0.04, color='#00ff88')
    ax2.text(0, 105, '★ Selected', ha='center', fontsize=7, color='#00ff88', fontweight='bold')

    # Graph 3 — Adversarial robustness
    ax3 = axes[2]
    _style_ax(ax3, 'Adversarial Robustness (FGSM Defense)')
    conds = ['Clean\nAccuracy', 'Under\nFGSM Attack', 'After RIP\nDefense']
    vals  = [95.50, 9.50, 53.00]
    bcolors = ['#00ff88', '#ff3355', '#ffcc00']
    bars3 = ax3.bar(conds, vals, color=bcolors, alpha=0.85, edgecolor='#1a3a4a', width=0.5)
    ax3.set_ylim(0, 115)
    for bar, val in zip(bars3, vals):
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{val}%', ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
    ax3.annotate('+74pp\nrecovery', xy=(2, 53), xytext=(1.45, 78),
                 arrowprops=dict(arrowstyle='->', color='#ffcc00', lw=1.5),
                 fontsize=8, color='#ffcc00', fontweight='bold')
    plt.tight_layout(pad=1.5)
    return fig

def plot_deepfake_video_graph():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor('#0d1f2d')
    _style_ax(ax, 'Video Detection Accuracy by Manipulation Type')
    types  = ['FF++\nOverall', 'Real', 'Deepfakes', 'Face2Face', 'FaceSwap', 'Neural\nTextures', 'DFD\n(Unseen)']
    accs   = [95.50, 100.0, 96.0, 92.0, 88.0, 88.0, 72.86]
    colors = ['#00d4ff','#00ff88','#00d4ff','#00d4ff','#ffcc00','#ffcc00','#ff8c00']
    bars = ax.bar(types, accs, color=colors, alpha=0.85, edgecolor='#1a3a4a')
    ax.axhline(51.2, color='#ff3355', linestyle='--', lw=1.2, alpha=0.8, label='Best Published (51.2%)')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=7, facecolor='#0d1f2d', edgecolor='#1a3a4a', labelcolor='#ff3355')
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f'{val}%', ha='center', va='bottom', color='white', fontsize=8, fontweight='bold')
    plt.tight_layout(pad=1.5)
    return fig

def plot_phishing_graphs():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.patch.set_facecolor('#0d1f2d')
    models    = ['BERT-LSTM', 'XGBoost v2', 'Random\nForest v2']
    accuracy  = [99.28, 94.17, 92.05]
    precision = [99.61, 93.80, 91.50]
    recall    = [99.55, 92.50, 90.20]
    f1        = [99.24, 93.10, 90.80]
    x = np.arange(len(models)); w = 0.18

    ax1 = axes[0]
    _style_ax(ax1, 'Phishing Detection — All Model Metrics')
    ax1.bar(x-1.5*w, accuracy,  w, label='Accuracy',  color='#00d4ff', alpha=0.85, edgecolor='#1a3a4a')
    ax1.bar(x-0.5*w, precision, w, label='Precision', color='#00ff88', alpha=0.85, edgecolor='#1a3a4a')
    ax1.bar(x+0.5*w, recall,    w, label='Recall',    color='#ffcc00', alpha=0.85, edgecolor='#1a3a4a')
    ax1.bar(x+1.5*w, f1,        w, label='F1-Score',  color='#ff8c00', alpha=0.85, edgecolor='#1a3a4a')
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=8)
    ax1.set_ylim(85, 103)
    ax1.legend(fontsize=7, facecolor='#0d1f2d', edgecolor='#1a3a4a', labelcolor='#e8f4f8', ncol=2, loc='lower right')

    ax2 = axes[1]
    _style_ax(ax2, 'Accuracy vs F1-Score Comparison')
    bars = ax2.bar(['BERT-LSTM','XGBoost v2','Random Forest v2'], accuracy,
                   color=['#00ff88','#00d4ff','#ffcc00'], alpha=0.85, edgecolor='#1a3a4a', width=0.5)
    ax2.plot(['BERT-LSTM','XGBoost v2','Random Forest v2'], f1,
             color='#ff8c00', marker='o', linewidth=2, markersize=7, label='F1-Score', zorder=5)
    ax2.set_ylim(85, 103)
    ax2.legend(fontsize=7, facecolor='#0d1f2d', edgecolor='#1a3a4a', labelcolor='#e8f4f8')
    _bar_labels(ax2, bars)
    for i, val in enumerate(f1):
        ax2.text(i, val+0.3, f'{val}%', ha='center', va='bottom', color='#ff8c00', fontsize=7, fontweight='bold')
    plt.tight_layout(pad=1.5)
    return fig

def plot_fakenews_graphs():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.patch.set_facecolor('#0d1f2d')
    models    = ['DistilBERT', 'TF-IDF + LR']
    accuracy  = [98.0, 96.5]; precision = [99.0, 97.0]
    recall    = [99.0, 96.0]; f1        = [99.0, 97.0]
    x = np.arange(len(models)); w = 0.18

    ax1 = axes[0]
    _style_ax(ax1, 'Fake News Detection — All Model Metrics')
    ax1.bar(x-1.5*w, accuracy,  w, label='Accuracy',  color='#00d4ff', alpha=0.85, edgecolor='#1a3a4a')
    ax1.bar(x-0.5*w, precision, w, label='Precision', color='#00ff88', alpha=0.85, edgecolor='#1a3a4a')
    ax1.bar(x+0.5*w, recall,    w, label='Recall',    color='#ffcc00', alpha=0.85, edgecolor='#1a3a4a')
    ax1.bar(x+1.5*w, f1,        w, label='F1-Score',  color='#ff8c00', alpha=0.85, edgecolor='#1a3a4a')
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=10)
    ax1.set_ylim(88, 102)
    ax1.legend(fontsize=7, facecolor='#0d1f2d', edgecolor='#1a3a4a', labelcolor='#e8f4f8', ncol=2, loc='lower right')

    ax2 = axes[1]
    ax2.set_facecolor('#0d1f2d')
    wedges, texts, autotexts = ax2.pie(
        [70, 30], labels=['DistilBERT\n(70%)', 'TF-IDF+LR\n(30%)'],
        colors=['#00d4ff','#00ff88'], explode=(0.05,0.05), autopct='%1.0f%%', startangle=90,
        textprops={'color':'#e8f4f8','fontsize':9},
        wedgeprops={'edgecolor':'#1a3a4a','linewidth':1.5}
    )
    for at in autotexts: at.set_color('white'); at.set_fontweight('bold'); at.set_fontsize(11)
    ax2.set_title('Ensemble Weight Distribution\n(DistilBERT 98% | TF-IDF 96.5%)',
                  color='#00d4ff', fontsize=9, fontweight='bold')
    plt.tight_layout(pad=1.5)
    return fig

def _metrics_badges(acc, prec, rec, f1, params=None, size=None):
    badges = f"""
    <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:0.5rem;">
        <span class="badge-ok">✓ Accuracy: {acc}</span>
        <span class="badge-ok">✓ Precision: {prec}</span>
        <span class="badge-ok">✓ Recall: {rec}</span>
        <span class="badge-ok">✓ F1: {f1}</span>
        {'<span class="badge-info">⚙️ Params: '+params+'</span>' if params else ''}
        {'<span class="badge-info">📦 Size: '+size+'</span>' if size else ''}
    </div>"""
    return badges

# ══════════════════════════════════════════════════════════════
# DEEPFAKE MODEL SETUP
# ══════════════════════════════════════════════════════════════
DOWNLOADS = Path(r'C:\Users\UMME\Downloads')
DEVICE    = torch.device('cpu')

class EfficientNetDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientnet = timm.create_model(
            'efficientnet_b0', pretrained=False, num_classes=2)
    def forward(self, x): return self.efficientnet(x)

@st.cache_resource
def load_deepfake_models():
    def _load(path):
        m    = EfficientNetDetector().to(DEVICE)
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        sd   = ckpt.get('state_dict') or ckpt
        m.load_state_dict(sd, strict=True)
        m.eval()
        return m
    return {
        'Final (v3)':     _load(DOWNLOADS / 'EfficientNet_final.pth'),
        'Finetuned (v2)': _load(DOWNLOADS / 'EfficientNet_finetuned_v2.pth'),
        'Adversarial':    _load(DOWNLOADS / 'EfficientNet_adversarial.pth'),
        'Video':          _load(DOWNLOADS / 'EfficientNet_video.pth'),
    }

@st.cache_resource
def load_phishing_detector():
    try:
        from email_detector import PhishingEmailDetector, detect_phishing_rules_only
        models_path = str(BASE_DIR / 'models')
        detector    = PhishingEmailDetector(models_path)
        loaded      = detector.load_models()
        return detector, loaded, detect_phishing_rules_only
    except Exception as e:
        try:
            from email_detector import detect_phishing_rules_only
            return None, False, detect_phishing_rules_only
        except:
            return None, False, None

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

val_tf = T.Compose([
    T.Resize((224,224)), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── DEEPFAKE HELPERS ──────────────────────────────────────────
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
    pad_x = int(w*0.2); pad_y = int(h*0.2)
    x1 = max(0, x-pad_x); y1 = max(0, y-pad_y)
    x2 = min(img_pil.width, x+w+pad_x)
    y2 = min(img_pil.height, y+h+pad_y)
    return img_pil.crop((x1,y1,x2,y2))

def check_image_quality(img_pil):
    warnings = []
    arr = np.array(img_pil.convert('L'))
    if arr.mean() < 60:   warnings.append("⚠️ Dark image")
    if arr.mean() > 220:  warnings.append("⚠️ Overexposed")
    if cv2.Laplacian(arr, cv2.CV_64F).var() < 50:
        warnings.append("⚠️ Blurry image")
    if img_pil.width < 100 or img_pil.height < 100:
        warnings.append("⚠️ Very low resolution")
    return warnings

def enhance_image(img_pil):
    arr  = np.array(img_pil.convert('L'))
    mean = arr.mean()
    blur = cv2.Laplacian(arr, cv2.CV_64F).var()
    enhanced = img_pil.copy()
    applied  = []

    if mean < 60:
        factor = min(2.5, 120.0 / (mean + 1e-5))
        enhanced = ImageEnhance.Brightness(enhanced).enhance(factor)
        applied.append(f"Brightness +{factor:.1f}x")

    arr2 = np.array(enhanced.convert('L'))
    std  = arr2.std()
    if std < 40:
        enhanced = ImageEnhance.Contrast(enhanced).enhance(1.8)
        applied.append("Contrast +1.8x")

    if blur < 50:
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.5)
        applied.append("Sharpness +2.5x")

    arr3 = np.array(enhanced)
    denoised = cv2.fastNlMeansDenoisingColored(arr3, None, 6, 6, 7, 21)
    enhanced = Image.fromarray(denoised)
    if blur < 30:
        applied.append("Denoised")

    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)

    return enhanced, applied

def predict_image(img_pil, models, threshold=0.40):
    tensor  = val_tf(img_pil).unsqueeze(0).to(DEVICE)
    scores  = {}

    image_models = {n: m for n, m in models.items() if n != 'Video'}
    base_weights = {'Final (v3)': 0.55, 'Finetuned (v2)': 0.30, 'Adversarial': 0.15}

    with torch.no_grad():
        for name, model in image_models.items():
            probs = torch.softmax(model(tensor), dim=1)[0]
            scores[name] = {'real': probs[0].item(), 'fake': probs[1].item()}

    confidences = {n: abs(scores[n]['fake'] - 0.5) * 2 for n in scores}
    total_conf  = sum(confidences.values()) + 1e-8

    dynamic_weights = {}
    for n in scores:
        conf_w = confidences[n] / total_conf
        dynamic_weights[n] = 0.60 * base_weights[n] + 0.40 * conf_w

    total_w = sum(dynamic_weights.values())
    dynamic_weights = {n: dynamic_weights[n] / total_w for n in dynamic_weights}

    fake_prob = sum(scores[n]['fake'] * dynamic_weights[n] for n in scores)
    real_prob = 1 - fake_prob

    max_fake_score = max(scores[n]['fake'] for n in scores)
    if max_fake_score > 0.75:
        verdict = 'FAKE'
    else:
        verdict = 'FAKE' if fake_prob > threshold else 'REAL'

    for n in scores:
        scores[n]['confidence'] = round(confidences[n] * 100, 1)
        scores[n]['dyn_weight'] = round(dynamic_weights[n] * 100, 1)

    return verdict, fake_prob, real_prob, scores

def generate_heatmap(img_pil, model):
    try:
        tensor = val_tf(img_pil).unsqueeze(0).to(DEVICE)
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

def draw_face_boxes(img_pil, faces, verdict):
    img_draw = img_pil.copy()
    draw     = ImageDraw.Draw(img_draw)
    color    = '#ff3355' if verdict == 'FAKE' else '#00ff88'
    for i, (x, y, w, h) in enumerate(faces):
        for t in range(3):
            draw.rectangle([x-t, y-t, x+w+t, y+h+t], outline=color)
        draw.rectangle([x, y-22, x+w, y], fill=color)
        draw.text((x+4, y-20), f"Face {i+1}: {verdict}", fill='black')
    return img_draw

def analyze_video(video_path, models, threshold=0.40, max_frames=15, frame_skip=10):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    duration     = total_frames / fps if fps > 0 else 0
    cascade      = load_face_cascade()
    frame_data   = []
    thumbnails   = []
    frame_count  = 0
    extracted    = 0
    weights      = {'Final (v3)':0.20,'Finetuned (v2)':0.10,'Adversarial':0.10,'Video':0.60}
    while extracted < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % frame_skip != 0: continue
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(faces) == 0:
            faces = cascade.detectMultiScale(gray, 1.05, 3, minSize=(20,20))
        if len(faces) > 0:
            x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
            pad  = int(min(w,h)*0.2)
            x1   = max(0,x-pad); y1 = max(0,y-pad)
            x2   = min(frame.shape[1],x+w+pad)
            y2   = min(frame.shape[0],y+h+pad)
            crop = rgb[y1:y2, x1:x2]
        else:
            hh,ww = rgb.shape[:2]; s = min(hh,ww)
            crop  = rgb[(hh-s)//2:(hh-s)//2+s, (ww-s)//2:(ww-s)//2+s]
        img_pil  = Image.fromarray(crop)
        tensor   = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            fake_prob = sum(
                torch.softmax(m(tensor), dim=1)[0][1].item() * w
                for (n,m), w in zip(models.items(), weights.values())
            )
        frame_data.append({'frame': frame_count,
                           'time': frame_count/fps if fps>0 else extracted,
                           'fake_prob': fake_prob,
                           'verdict': 'FAKE' if fake_prob > threshold else 'REAL'})
        if extracted % 3 == 0:
            thumbnails.append((Image.fromarray(rgb).resize((120,68)), fake_prob))
        extracted += 1
    cap.release()
    if not frame_data: return None
    avg_fake   = np.mean([f['fake_prob'] for f in frame_data])
    fake_ratio = np.mean([f['verdict']=='FAKE' for f in frame_data])
    return {
        'verdict':      'FAKE' if fake_ratio >= 0.4 else 'REAL',
        'avg_fake':     avg_fake,
        'fake_ratio':   fake_ratio,
        'frame_data':   frame_data,
        'thumbnails':   thumbnails,
        'duration':     duration,
        'total_frames': total_frames,
        'fps':          fps,
    }

def plot_timeline(frame_data, threshold):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    fig.patch.set_facecolor('#0d1f2d')
    ax.set_facecolor('#0a1520')
    times = [f['time'] for f in frame_data]
    probs = [f['fake_prob'] for f in frame_data]
    ax.fill_between(times, probs, threshold,
                    where=[p > threshold for p in probs],
                    color='#ff3355', alpha=0.3)
    ax.fill_between(times, probs, threshold,
                    where=[p <= threshold for p in probs],
                    color='#00ff88', alpha=0.3)
    ax.plot(times, probs, color='#00d4ff', linewidth=2)
    ax.axhline(y=threshold, color='#ffcc00', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlim(min(times), max(times)); ax.set_ylim(0,1)
    ax.set_xlabel('Time (s)', color='#4a7a8a', fontsize=8)
    ax.set_ylabel('Fake Prob', color='#4a7a8a', fontsize=8)
    ax.tick_params(colors='#4a7a8a', labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor('#1a3a4a')
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════════
# LOAD ALL MODELS
# ══════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <h1 class="hero-title">CYBER<span>SHIELD</span></h1>
    <p class="hero-sub">// AI-POWERED CYBER THREAT DETECTION SYSTEM //</p>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# Load models
col_s1, col_s2 = st.columns(2)
with col_s1:
    with st.spinner("Loading deepfake models..."):
        try:
            df_models = load_deepfake_models()
            st.markdown('<p style="text-align:center;font-family:Share Tech Mono;font-size:0.72rem;color:#00ff88">✓ DEEPFAKE MODELS READY</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Deepfake models failed: {e}")
            df_models = None

with col_s2:
    with st.spinner("Loading phishing models..."):
        ph_detector, ph_loaded, ph_rules_only = load_phishing_detector()
        if ph_loaded:
            st.markdown('<p style="text-align:center;font-family:Share Tech Mono;font-size:0.72rem;color:#00ff88">✓ PHISHING MODELS READY</p>', unsafe_allow_html=True)
        elif ph_rules_only:
            st.markdown('<p style="text-align:center;font-family:Share Tech Mono;font-size:0.72rem;color:#ffcc00">⚡ PHISHING RULES-ONLY MODE</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="text-align:center;font-family:Share Tech Mono;font-size:0.72rem;color:#ff3355">✗ PHISHING MODULE NOT FOUND</p>', unsafe_allow_html=True)

# Load fake news detector
@st.cache_resource
def load_fakenews_detector():
    try:
        import importlib.util, sys
        fn_dir  = str(BASE_DIR / 'fakenews')
        fn_file = str(BASE_DIR / 'fakenews' / 'fakenews_detector.py')
        spec    = importlib.util.spec_from_file_location("fakenews_detector", fn_file)
        mod     = importlib.util.module_from_spec(spec)
        import os; orig = os.getcwd(); os.chdir(fn_dir)
        spec.loader.exec_module(mod)
        os.chdir(orig)
        return mod.detect_fake_news, True
    except Exception as e:
        return None, False

col_s3, _ = st.columns(2)
with col_s3:
    with st.spinner("Loading fake news models..."):
        fn_detect, fn_loaded = load_fakenews_detector()
        if fn_loaded:
            st.markdown('<p style="text-align:center;font-family:Share Tech Mono;font-size:0.72rem;color:#00ff88">✓ FAKE NEWS MODELS READY</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="text-align:center;font-family:Share Tech Mono;font-size:0.72rem;color:#ff3355">✗ FAKE NEWS MODULE NOT FOUND</p>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════
tab_deep, tab_phish, tab_news = st.tabs([
    "🎭  DEEPFAKE DETECTION",
    "📧  PHISHING DETECTION",
    "📰  FAKE NEWS DETECTION",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DEEPFAKE DETECTION
# ══════════════════════════════════════════════════════════════
with tab_deep:
    subtab_img, subtab_vid, subtab_perf = st.tabs(["📸  IMAGE", "🎥  VIDEO", "📊  PERFORMANCE GRAPHS"])

    # ── IMAGE ─────────────────────────────────────────────────
    with subtab_img:
        col_l, col_r = st.columns([1,1], gap="large")
        with col_l:
            st.markdown('<div class="sec-hdr">IMAGE INPUT</div>', unsafe_allow_html=True)

            MENTOR_REAL = Path(r'C:\Users\UMME\Downloads\MENTOR_DEMO\images\real')
            MENTOR_FAKE = Path(r'C:\Users\UMME\Downloads\MENTOR_DEMO\images\fake')

            def get_sample_images(folder, n=4):
                exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
                files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
                import random; random.seed(42)
                return sorted(files)[:n] if len(files) >= n else sorted(files)

            st.markdown('<p style="font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a">QUICK DEMO SAMPLES</p>', unsafe_allow_html=True)
            demo_c1, demo_c2, demo_c3, demo_c4 = st.columns(4)

            real_samples = get_sample_images(MENTOR_REAL) if MENTOR_REAL.exists() else []
            fake_samples = get_sample_images(MENTOR_FAKE) if MENTOR_FAKE.exists() else []

            for i, (col_, stype, slist) in enumerate([
                (demo_c1, 'real', real_samples),
                (demo_c2, 'real', real_samples),
                (demo_c3, 'fake', fake_samples),
                (demo_c4, 'fake', fake_samples),
            ]):
                idx = i % 2
                with col_:
                    clr  = '#00ff88' if stype == 'real' else '#ff3355'
                    icon = '🟢' if stype == 'real' else '🔴'
                    lbl  = f'{icon} Real {idx+1}' if stype == 'real' else f'{icon} Fake {idx+1}'
                    if slist and len(slist) > idx:
                        thumb = Image.open(slist[idx]).convert('RGB')
                        thumb.thumbnail((80, 80))
                        st.image(thumb, use_container_width=True)
                        if st.button(lbl, key=f'demo_{stype}_{idx}', use_container_width=True):
                            st.session_state['demo_img'] = str(slist[idx])
                            st.session_state['demo_label'] = stype.upper()
                    else:
                        st.markdown(f'<p style="font-family:Share Tech Mono;font-size:.6rem;color:#4a7a8a">No {stype} samples found</p>', unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr">OR UPLOAD YOUR OWN</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("Drop image",
                type=['jpg','jpeg','png','bmp','webp'],
                key='img_up', label_visibility='collapsed')

            if uploaded:
                img = Image.open(uploaded).convert('RGB')
                st.session_state.pop('demo_img', None)
            elif 'demo_img' in st.session_state:
                img = Image.open(st.session_state['demo_img']).convert('RGB')
                uploaded = True
            else:
                img = None

            if img:
                st.image(img, use_container_width=True)

                st.markdown('<div class="sec-hdr">SETTINGS</div>', unsafe_allow_html=True)
                sensitivity  = st.select_slider("SENSITIVITY",
                    options=["LOW","MEDIUM","HIGH"], value="MEDIUM")
                tmap         = {"LOW":0.55,"MEDIUM":0.35,"HIGH":0.20}
                threshold    = tmap[sensitivity]
                show_heatmap = st.checkbox("🔥 GradCAM Heatmap", value=True)
                show_boxes   = st.checkbox("🔍 Face Detection Boxes", value=True)
                auto_enhance = st.checkbox("✨ Auto Enhance Low Quality", value=True,
                    help="Automatically improves dark/blurry images before prediction")
                analyze_img  = st.button("⚡ ANALYZE IMAGE", key='btn_img')

        with col_r:
            st.markdown('<div class="sec-hdr">RESULTS</div>', unsafe_allow_html=True)
            if img and analyze_img and df_models:
                with st.spinner("Analyzing..."):
                    quality_w = check_image_quality(img)

                    process_img = img
                    enhance_log = []
                    if auto_enhance and len(quality_w) > 0:
                        process_img, enhance_log = enhance_image(img)

                    faces = detect_faces(process_img)
                    if len(faces) == 0:
                        faces = detect_faces(img)
                        process_img = img

                    if len(faces) > 0:
                        largest  = max(faces, key=lambda f: f[2]*f[3])
                        face_img = crop_face(process_img, largest)
                        verdict, fake_prob, real_prob, scores = predict_image(
                            face_img, df_models, threshold)
                    else:
                        verdict, fake_prob, real_prob, scores = predict_image(
                            process_img, df_models, threshold)
                        face_img = process_img

                conf = max(fake_prob, real_prob)*100
                if verdict == 'FAKE':
                    card_class, color, icon = 'verdict-fake', '#ff3355', '🔴'
                else:
                    card_class, color, icon = 'verdict-real', '#00ff88', '🟢'

                st.markdown(f"""
                <div class="{card_class}">
                    <p class="verdict-label" style="color:{color}">{icon} {verdict}</p>
                    <p class="verdict-conf">Confidence: {conf:.1f}%</p>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if fake_prob >= real_prob:
                    bar_html = f"""
                    <div style="margin-bottom:.3rem">
                        <div style="display:flex;justify-content:space-between;
                                    font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:2px">
                            <span>FAKE PROBABILITY</span><span style="color:#ff3355">{fake_prob*100:.1f}%</span>
                        </div>
                        <div class="bar-wrap"><div class="bar-fake" style="width:{fake_prob*100:.1f}%"></div></div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;
                                    font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:2px">
                            <span>REAL PROBABILITY</span><span style="color:#00ff88">{real_prob*100:.1f}%</span>
                        </div>
                        <div class="bar-wrap"><div class="bar-real" style="width:{real_prob*100:.1f}%"></div></div>
                    </div>"""
                else:
                    bar_html = f"""
                    <div style="margin-bottom:.3rem">
                        <div style="display:flex;justify-content:space-between;
                                    font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:2px">
                            <span>REAL PROBABILITY</span><span style="color:#00ff88">{real_prob*100:.1f}%</span>
                        </div>
                        <div class="bar-wrap"><div class="bar-real" style="width:{real_prob*100:.1f}%"></div></div>
                    </div>
                    <div>
                        <div style="display:flex;justify-content:space-between;
                                    font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:2px">
                            <span>FAKE PROBABILITY</span><span style="color:#ff3355">{fake_prob*100:.1f}%</span>
                        </div>
                        <div class="bar-wrap"><div class="bar-fake" style="width:{fake_prob*100:.1f}%"></div></div>
                    </div>"""
                st.markdown(bar_html, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                for w in quality_w:
                    st.markdown(f'<span class="badge-warn">{w}</span>', unsafe_allow_html=True)
                if enhance_log:
                    for e in enhance_log:
                        st.markdown(f'<span class="badge-info">✨ Enhanced: {e}</span>', unsafe_allow_html=True)
                if len(faces) > 0:
                    st.markdown(f'<span class="badge-info">✓ {len(faces)} face(s) detected</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-warn">⚠️ No face detected</span>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec-hdr">MODEL BREAKDOWN</div>', unsafe_allow_html=True)

                sorted_scores = sorted(
                    scores.items(),
                    key=lambda x: x[1].get('confidence', 0),
                    reverse=True
                )

                for i, (name, score) in enumerate(sorted_scores):
                    fp   = score['fake'] * 100
                    conf = score.get('confidence', 0)
                    dw   = score.get('dyn_weight', 0)
                    clr  = '#ff3355' if fp > 50 else '#00ff88'
                    crown = ' 👑' if i == 0 else ''
                    st.markdown(f"""
                    <div class="model-row" style="{'border-left:2px solid #00d4ff;padding-left:6px;' if i==0 else ''}">
                        <span class="model-name">{name}{crown}</span>
                        <span style="font-family:Share Tech Mono;font-size:.68rem;color:#4a7a8a">
                            conf:{conf:.0f}% w:{dw:.0f}%
                        </span>
                        <span style="font-family:Rajdhani;font-weight:600;font-size:.95rem;color:{clr}">{fp:.1f}% fake</span>
                    </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                v1, v2 = st.columns(2)
                with v1:
                    if show_boxes and len(faces) > 0:
                        st.markdown('<p style="font-family:Share Tech Mono;font-size:.68rem;color:#4a7a8a">FACE DETECTION</p>', unsafe_allow_html=True)
                        st.image(draw_face_boxes(img, faces, verdict), use_container_width=True)
                with v2:
                    if show_heatmap:
                        st.markdown('<p style="font-family:Share Tech Mono;font-size:.68rem;color:#4a7a8a">GRADCAM HEATMAP</p>', unsafe_allow_html=True)
                        st.image(generate_heatmap(face_img, df_models['Final (v3)']), use_container_width=True)

            elif not img:
                st.markdown("""
                <div style="text-align:center;padding:2rem;border:1px dashed #1a3a4a;border-radius:8px;">
                    <p style="font-size:2rem;margin:0">🎭</p>
                    <p style="font-family:Share Tech Mono;font-size:.78rem;color:#4a7a8a">Click a sample above or upload your own image</p>
                </div>""", unsafe_allow_html=True)

    # ── VIDEO ─────────────────────────────────────────────────
    with subtab_vid:
        col_l, col_r = st.columns([1,1], gap="large")
        with col_l:
            st.markdown('<div class="sec-hdr">UPLOAD VIDEO</div>', unsafe_allow_html=True)
            vid_up = st.file_uploader("Drop video",
                type=['mp4','avi','mov'], key='vid_up',
                label_visibility='collapsed')
            if vid_up:
                st.video(vid_up)
                vid_sens  = st.select_slider("SENSITIVITY",
                    options=["LOW","MEDIUM","HIGH"], value="MEDIUM", key='vid_sens')
                vid_thr   = {"LOW":0.65,"MEDIUM":0.40,"HIGH":0.25}[vid_sens]
                max_frames= st.slider("MAX FRAMES", 5, 30, 15, 5)
                analyze_v = st.button("⚡ ANALYZE VIDEO", key='btn_vid')

        with col_r:
            st.markdown('<div class="sec-hdr">RESULTS</div>', unsafe_allow_html=True)
            if vid_up and analyze_v and df_models:
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix=Path(vid_up.name).suffix) as tmp:
                    tmp.write(vid_up.read()); tmp_path = tmp.name

                prog = st.progress(0, text="Analyzing frames...")
                result = analyze_video(tmp_path, df_models,
                                       threshold=vid_thr, max_frames=max_frames)
                prog.progress(100, text="Done!")
                os.unlink(tmp_path)

                if result:
                    verdict = result['verdict']
                    fp      = result['avg_fake']
                    conf    = max(fp, 1-fp)*100
                    if verdict == 'FAKE':
                        card_c, color, icon = 'verdict-fake', '#ff3355', '🔴'
                    else:
                        card_c, color, icon = 'verdict-real', '#00ff88', '🟢'

                    st.markdown(f"""
                    <div class="{card_c}">
                        <p class="verdict-label" style="color:{color}">{icon} {verdict}</p>
                        <p class="verdict-conf">Confidence: {conf:.1f}% | Fake frames: {result['fake_ratio']*100:.0f}%</p>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    s1, s2, s3 = st.columns(3)
                    for col_, label, val, clr in [
                        (s1, "DURATION",    f"{result['duration']:.1f}s", '#00d4ff'),
                        (s2, "FRAMES",      str(len(result['frame_data'])), '#00d4ff'),
                        (s3, "FAKE FRAMES", f"{result['fake_ratio']*100:.0f}%",
                         '#ff3355' if result['fake_ratio']>0.4 else '#00ff88')
                    ]:
                        with col_:
                            st.markdown(f"""
                            <div style="text-align:center;background:#0d1f2d;
                                        border:1px solid #1a3a4a;border-radius:8px;padding:.7rem">
                                <p style="font-family:Share Tech Mono;font-size:.68rem;color:#4a7a8a;margin:0">{label}</p>
                                <p style="font-family:Rajdhani;font-size:1.2rem;font-weight:700;color:{clr};margin:0">{val}</p>
                            </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="sec-hdr">FRAME TIMELINE</div>', unsafe_allow_html=True)
                    fig = plot_timeline(result['frame_data'], vid_thr)
                    st.pyplot(fig, use_container_width=True); plt.close()

                    if result['thumbnails']:
                        st.markdown('<div class="sec-hdr">SAMPLE FRAMES</div>', unsafe_allow_html=True)
                        thumb_cols = st.columns(min(len(result['thumbnails']),5))
                        for i, (thumb, prob) in enumerate(result['thumbnails'][:5]):
                            with thumb_cols[i]:
                                clr = '#ff3355' if prob > vid_thr else '#00ff88'
                                st.image(thumb, use_container_width=True)
                                st.markdown(
                                    f'<p style="text-align:center;font-family:Share Tech Mono;'
                                    f'font-size:.62rem;color:{clr};margin:0">'
                                    f'{"FAKE" if prob>vid_thr else "REAL"} {prob*100:.0f}%</p>',
                                    unsafe_allow_html=True)
            elif not vid_up:
                st.markdown("""
                <div style="text-align:center;padding:2rem;border:1px dashed #1a3a4a;border-radius:8px;">
                    <p style="font-size:2rem;margin:0">🎥</p>
                    <p style="font-family:Share Tech Mono;font-size:.78rem;color:#4a7a8a">Upload a video to begin</p>
                </div>""", unsafe_allow_html=True)

    # ── DEEPFAKE PERFORMANCE GRAPHS ────────────────────────────
    with subtab_perf:
        st.markdown('<div class="sec-hdr">DEEPFAKE DETECTION — MODEL PERFORMANCE (Conference / IEEE)</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:0.5rem">Static evaluation results — based on training across 5 benchmark datasets. These are the numbers submitted in the IEEE paper.</p>', unsafe_allow_html=True)

        st.markdown('<div class="sec-hdr" style="margin-top:1rem">FIG. 5 — ACCURACY COMPARISON & ARCHITECTURE</div>', unsafe_allow_html=True)
        fig5 = plot_deepfake_graphs()
        st.pyplot(fig5, use_container_width=True)
        plt.close()

        st.markdown('<div class="sec-hdr" style="margin-top:1rem">FIG. 6 — VIDEO DETECTION BY MANIPULATION TYPE</div>', unsafe_allow_html=True)
        fig6 = plot_deepfake_video_graph()
        st.pyplot(fig6, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1f2d;border:1px solid #1a3a4a;border-radius:8px;padding:1rem;">
            <p class="sec-hdr" style="margin:0 0 0.5rem 0">KEY METRICS SUMMARY — EfficientNet-B0 (Final Model)</p>
            {_metrics_badges('93.50%', '93.80%', '93.19%', '93.19%', '4.01M', '15.59MB')}
            <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:0.4rem;">
                <span class="badge-ok">✓ CelebDF Unseen: 81.50%</span>
                <span class="badge-ok">✓ DFD Video Unseen: 72.86%</span>
                <span class="badge-warn">⚡ Adversarial Recovery: +74pp</span>
                <span class="badge-info">📊 Beats published baselines by up to +30pp</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — PHISHING DETECTION
# ══════════════════════════════════════════════════════════════
with tab_phish:
    ph_det, ph_perf = st.tabs(["📧  DETECTION", "📊  PERFORMANCE GRAPHS"])
    with ph_det:
        col_l, col_r = st.columns([1,1], gap="large")

        with col_l:
            st.markdown('<div class="sec-hdr">EMAIL INPUT</div>', unsafe_allow_html=True)

            ph_samples = {
                "Phishing — Internship Scam": "Subject: Congratulations! You've been shortlisted!\n\nDear Student, you have been selected for our 2-month internship with WIPRO, IBM, CISCO, META certifications. 100% Placement Guarantee! Fill form: https://forms.gle/fake123 Pay registration fee ₹2,999.",
                "Phishing — Bank Scam": "Subject: URGENT: Your SBI Account Will Be Blocked\n\nYour SBI account KYC is incomplete. Update immediately at http://sbi-kyc-update.in/verify — enter account number and ATM PIN. Account will be permanently blocked in 24 hours.",
                "Legitimate — GitHub Alert": "From: noreply@github.com\nSubject: New sign-in to your account\n\nHi, we noticed a new sign-in to your GitHub account from Chrome on Windows, Bengaluru, India. If this was you, no action needed. https://github.com/settings/security",
                "Legitimate — University Notice": "From: examcell@alliance.edu.in\nSubject: End Semester Exam Schedule April 2026\n\nDear Students, exams scheduled April 15-30 2026. Carry University ID. Download hall ticket from student portal. Contact examcell@alliance.edu.in for queries."
            }

            input_tab1, input_tab2, input_tab3 = st.tabs(["✏️ Type / Paste", "📂 Upload File", "📋 Load Sample"])

            with input_tab1:
                typed_email = st.text_area(
                    "Type or paste email",
                    height=220,
                    placeholder="Paste the full email here (subject + body)...\n\nExample:\nSubject: URGENT! Your account is suspended\n\nDear Customer, click here to verify...",
                    label_visibility='collapsed',
                    key='email_typed'
                )
                if typed_email.strip():
                    st.session_state['email_draft'] = typed_email

            with input_tab2:
                st.markdown('<p style="font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a">Upload .txt, .eml or .csv file</p>', unsafe_allow_html=True)
                email_file = st.file_uploader("Upload email file",
                    type=['txt','eml','csv'],
                    label_visibility='collapsed',
                    key='email_file')
                if email_file is not None:
                    try:
                        if email_file.name.endswith('.csv'):
                            import pandas as pd
                            df_email = pd.read_csv(email_file)
                            file_text = df_email.iloc[0].to_string()
                        else:
                            file_text = email_file.read().decode('utf-8', errors='ignore')
                        st.session_state['email_draft'] = file_text
                        st.markdown(f'<span class="badge-ok">✓ File loaded: {email_file.name}</span>', unsafe_allow_html=True)
                        st.text_area("Preview", value=file_text[:300] + "..." if len(file_text) > 300 else file_text,
                                    height=120, disabled=True, label_visibility='collapsed', key='email_preview')
                    except Exception as e:
                        st.markdown(f'<span class="badge-danger">✗ Error reading file: {e}</span>', unsafe_allow_html=True)

            with input_tab3:
                sample_type = st.selectbox("Select sample:", list(ph_samples.keys()), key='ph_sample_sel')
                selected_ph_text = ph_samples[sample_type]
                st.markdown(f'''<div style="background:#0a1520;border:1px solid #1a3a4a;border-radius:6px;
                    padding:0.7rem;font-family:Share Tech Mono;font-size:0.72rem;color:#a0c4d4;
                    height:120px;overflow-y:auto;white-space:pre-wrap;">{selected_ph_text}</div>''',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⚡ Use This Sample", key='load_sample', use_container_width=True):
                    st.session_state['email_draft'] = ph_samples[sample_type]
                    st.rerun()

            email_text = st.session_state.get('email_draft', '')
            if email_text:
                st.markdown(f'<span class="badge-info">✓ Ready to analyze — {len(email_text)} characters</span>', unsafe_allow_html=True)

            st.markdown('<div class="sec-hdr">SETTINGS</div>', unsafe_allow_html=True)
            if ph_loaded:
                mode = st.radio("Detection Mode:", ["🚀 Full (All Models)", "⚡ Quick (Rules Only)"])
            else:
                mode = "⚡ Quick (Rules Only)"
                st.markdown('<span class="badge-warn">⚡ Rules-only mode (models not loaded)</span>', unsafe_allow_html=True)

            analyze_email = st.button("⚡ ANALYZE EMAIL", key='btn_email',
                                       disabled=len(email_text.strip()) < 10)

        with col_r:
            st.markdown('<div class="sec-hdr">RESULTS</div>', unsafe_allow_html=True)

            if analyze_email and email_text.strip():
                with st.spinner("Analyzing email..."):
                    if "Full" in mode and ph_loaded and ph_detector:
                        try:
                            result = ph_detector.predict(email_text)
                        except:
                            result = ph_rules_only(email_text) if ph_rules_only else None
                    elif ph_rules_only:
                        result = ph_rules_only(email_text)
                    else:
                        result = None

                if result:
                    is_phishing = result['is_phishing']
                    confidence  = result['confidence']
                    risk_level  = result['risk_level']
                    details     = result['details']

                    if is_phishing and confidence >= 80:
                        card_c, color, icon = 'verdict-phishing', '#ff3355', '🔴'
                    elif is_phishing:
                        card_c, color, icon = 'verdict-medium', '#ffcc00', '🟡'
                    else:
                        card_c, color, icon = 'verdict-legit', '#00ff88', '🟢'

                    verdict_text = "PHISHING" if is_phishing else "LEGITIMATE"
                    conf_display = confidence if is_phishing else 100 - confidence

                    st.markdown(f"""
                    <div class="{card_c}">
                        <p class="verdict-label" style="color:{color}">{icon} {verdict_text}</p>
                        <p class="verdict-conf">Confidence: {conf_display:.1f}% | Risk: {risk_level}</p>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    bar_class = 'bar-fake' if is_phishing else 'bar-real'
                    st.markdown(f"""
                    <div style="margin-bottom:.3rem">
                        <div style="display:flex;justify-content:space-between;
                                    font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:2px">
                            <span>PHISHING SCORE</span>
                            <span style="color:{'#ff3355' if is_phishing else '#00ff88'}">{confidence:.1f}%</span>
                        </div>
                        <div class="bar-wrap"><div class="{bar_class}" style="width:{confidence}%"></div></div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if 'bert_score' in details:
                        st.markdown('<div class="sec-hdr">MODEL BREAKDOWN</div>', unsafe_allow_html=True)
                        for name, score in [
                            ('BERT-LSTM',     details['bert_score']),
                            ('XGBoost',       details['xgb_score']),
                            ('Random Forest', details['rf_score']),
                            ('Rules v9',      details['rule_score']),
                        ]:
                            clr = '#ff3355' if score > 50 else '#00ff88'
                            st.markdown(f"""
                            <div class="model-row">
                                <span class="model-name">{name}</span>
                                <span style="font-family:Rajdhani;font-weight:600;font-size:.95rem;color:{clr}">{score:.1f}%</span>
                            </div>""", unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown('<div class="sec-hdr">FLAGS DETECTED</div>', unsafe_allow_html=True)
                    flag_col1, flag_col2 = st.columns(2)

                    with flag_col1:
                        st.markdown('<p style="font-family:Share Tech Mono;font-size:.7rem;color:#ff3355">RED FLAGS</p>', unsafe_allow_html=True)
                        if details.get('red_reasons'):
                            for r in details['red_reasons']:
                                st.markdown(f'<div class="flag-red">⚠️ {r}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-ok">✓ None detected</span>', unsafe_allow_html=True)

                    with flag_col2:
                        st.markdown('<p style="font-family:Share Tech Mono;font-size:.7rem;color:#00ff88">GREEN FLAGS</p>', unsafe_allow_html=True)
                        if details.get('green_reasons'):
                            for g in details['green_reasons']:
                                st.markdown(f'<div class="flag-green">✓ {g}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown('<span class="badge-warn">⚠️ None detected</span>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown('<div class="sec-hdr">RECOMMENDATION</div>', unsafe_allow_html=True)
                    if risk_level == "HIGH":
                        st.error("🚫 HIGH RISK — Do NOT click links, reply, or share personal info. Report to IT and mark as spam.")
                    elif risk_level == "MEDIUM":
                        st.warning("⚠️ MEDIUM RISK — Verify sender through official channels before taking any action.")
                    elif risk_level == "LOW":
                        st.info("ℹ️ LOW RISK — Review carefully. Verify if you expected this email.")
                    else:
                        st.success("✅ SAFE — No significant phishing indicators found.")
                else:
                    st.error("Phishing detection module not available.")

            elif not email_text.strip():
                st.markdown("""
                <div style="text-align:center;padding:2rem;border:1px dashed #1a3a4a;border-radius:8px;">
                    <p style="font-size:2rem;margin:0">📧</p>
                    <p style="font-family:Share Tech Mono;font-size:.78rem;color:#4a7a8a">Paste an email to analyze</p>
                </div>""", unsafe_allow_html=True)

    with ph_perf:
        st.markdown('<div class="sec-hdr">PHISHING DETECTION — MODEL PERFORMANCE (Conference / IEEE)</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:0.5rem">Static evaluation results — BERT-LSTM, XGBoost v2, Random Forest v2 trained on phishing email datasets.</p>', unsafe_allow_html=True)
        fig_ph = plot_phishing_graphs()
        st.pyplot(fig_ph, use_container_width=True)
        plt.close()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1f2d;border:1px solid #1a3a4a;border-radius:8px;padding:1rem;">
            <p class="sec-hdr" style="margin:0 0 0.5rem 0">KEY METRICS SUMMARY — BERT-LSTM (Best Model)</p>
            {_metrics_badges('99.28%', '99.61%', '99.55%', '99.24%')}
            <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:0.4rem;">
                <span class="badge-info">📊 XGBoost v2: 94.17% accuracy</span>
                <span class="badge-info">📊 Random Forest v2: 92.05% accuracy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — FAKE NEWS DETECTION
# ══════════════════════════════════════════════════════════════
with tab_news:
    fn_det, fn_perf = st.tabs(["📰  DETECTION", "📊  PERFORMANCE GRAPHS"])
    with fn_det:
        col_l, col_r = st.columns([1,1], gap="large")

        with col_l:
            st.markdown('<div class="sec-hdr">NEWS ARTICLE INPUT</div>', unsafe_allow_html=True)

            # ── ONLY CHANGE IN ENTIRE FILE — Fixed samples that give clear REAL/FAKE ──
            fn_samples = {
                "Fake — Conspiracy Theory": "BREAKING BOMBSHELL: Scientists CONFIRM 5G towers are secretly mind control devices installed by globalist elites. The mainstream media is BANNED from reporting this shocking truth. WAKE UP people! MUST SHARE before they delete this! The cover up goes all the way to the top!!! They dont want you to know this HIDDEN TRUTH! Exposed suppressed secret leaked warning exclusive bombshell shocking urgent!!!",

                "Fake — Health Misinformation": "SHOCKING EXPOSED: Doctors are HIDING the cure for cancer! Big Pharma has SUPPRESSED this natural remedy for decades. Leaked documents CONFIRM that drinking this tea cures ALL cancers in 3 days. They dont want you to know this HIDDEN TRUTH! Share before banned! BREAKING WARNING: Vaccines confirmed to contain microchips! Mainstream media covering up this bombshell secret exclusively suppressed by globalist elites wake up!!!",

                "Real — Financial News": "The Federal Reserve raised interest rates by 25 basis points on Wednesday, the tenth consecutive increase since March 2022. The decision was unanimous among the twelve voting members of the Federal Open Market Committee according to the official statement. Reuters and Associated Press reported that economists at Goldman Sachs and JPMorgan had widely anticipated the move following recent inflation data published by the Bureau of Labor Statistics showing a 0.4 percent monthly increase in consumer prices. The Federal Reserve chairman stated in a press conference that the committee remains committed to bringing inflation down to its 2 percent target.",

                "Real — Science News": "Researchers at the Massachusetts Institute of Technology published a peer-reviewed study in the journal Nature on Thursday showing that a new artificial intelligence model can predict protein folding structures with 95 percent accuracy. The research, conducted over three years with funding from the National Institute of Health, involved analysis of over 200,000 protein structures according to lead researcher Dr. Sarah Chen. The findings were independently verified by scientists at Stanford University and the University of Cambridge. The study represents a significant advancement according to researchers at the National Science Foundation.",
            }
            # ── END OF CHANGE ──────────────────────────────────────

            fn_tab1, fn_tab2, fn_tab3 = st.tabs(["✏️ Type / Paste", "📂 Upload File", "📋 Load Sample"])

            with fn_tab1:
                typed_news = st.text_area(
                    "Type or paste article",
                    height=220,
                    placeholder="Paste the news article or headline here...\n\nExample:\nBREAKING: Scientists discover cure for cancer hidden by Big Pharma...",
                    label_visibility='collapsed',
                    key='news_typed'
                )
                if typed_news.strip():
                    st.session_state['fn_draft'] = typed_news

            with fn_tab2:
                st.markdown('<p style="font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a">Upload .txt or .csv file</p>', unsafe_allow_html=True)
                news_file = st.file_uploader("Upload news file",
                    type=['txt','csv'],
                    label_visibility='collapsed',
                    key='news_file')
                if news_file is not None:
                    try:
                        if news_file.name.endswith('.csv'):
                            import pandas as pd
                            df_news = pd.read_csv(news_file)
                            for col in ['text','content','article','body','news']:
                                if col in df_news.columns:
                                    file_text = str(df_news[col].iloc[0])
                                    break
                            else:
                                file_text = df_news.iloc[0].to_string()
                        else:
                            file_text = news_file.read().decode('utf-8', errors='ignore')
                        st.session_state['fn_draft'] = file_text
                        st.markdown(f'<span class="badge-ok">✓ File loaded: {news_file.name}</span>', unsafe_allow_html=True)
                        st.text_area("Preview", value=file_text[:300] + "..." if len(file_text) > 300 else file_text,
                                    height=120, disabled=True, label_visibility='collapsed', key='news_preview')
                    except Exception as e:
                        st.markdown(f'<span class="badge-danger">✗ Error reading file: {e}</span>', unsafe_allow_html=True)

            with fn_tab3:
                fn_sample_type = st.selectbox("Select sample:", list(fn_samples.keys()), key='fn_sample_sel')
                selected_fn_text = fn_samples[fn_sample_type]
                st.markdown(f'''<div style="background:#0a1520;border:1px solid #1a3a4a;border-radius:6px;
                    padding:0.7rem;font-family:Share Tech Mono;font-size:0.72rem;color:#a0c4d4;
                    height:120px;overflow-y:auto;white-space:pre-wrap;">{selected_fn_text}</div>''',
                    unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⚡ Use This Sample", key='load_fn_sample', use_container_width=True):
                    st.session_state['fn_draft'] = fn_samples[fn_sample_type]
                    st.rerun()

            news_text = st.session_state.get('fn_draft', '')
            if news_text:
                st.markdown(f'<span class="badge-info">✓ Ready to analyze — {len(news_text)} characters</span>', unsafe_allow_html=True)

            analyze_news = st.button("⚡ ANALYZE ARTICLE", key='btn_news',
                                      disabled=len(news_text.strip()) < 20)

        with col_r:
            st.markdown('<div class="sec-hdr">RESULTS</div>', unsafe_allow_html=True)

            if analyze_news and news_text.strip():
                if fn_detect:
                    with st.spinner("Analyzing article..."):
                        import os
                        orig_dir = os.getcwd()
                        os.chdir(str(BASE_DIR / 'fakenews'))
                        result = fn_detect(news_text)
                        os.chdir(orig_dir)

                    verdict    = result['verdict']
                    confidence = result['confidence']
                    is_fake    = result['is_fake']
                    warnings   = result.get('warnings', [])

                    if verdict == 'FAKE':
                        card_c, color, icon = 'verdict-fake', '#ff3355', '🔴'
                    elif verdict == 'REAL':
                        card_c, color, icon = 'verdict-real', '#00ff88', '🟢'
                    else:
                        card_c, color, icon = 'verdict-medium', '#ffcc00', '🟡'

                    st.markdown(f"""
                    <div class="{card_c}">
                        <p class="verdict-label" style="color:{color}">{icon} {verdict}</p>
                        <p class="verdict-conf">Confidence: {confidence:.1f}%</p>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    bar_c = 'bar-fake' if is_fake else ('bar-real' if is_fake is False else 'bar-warn')
                    st.markdown(f"""
                    <div style="margin-bottom:.5rem">
                        <div style="display:flex;justify-content:space-between;
                                    font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:2px">
                            <span>CONFIDENCE SCORE</span>
                            <span style="color:{color}">{confidence:.1f}%</span>
                        </div>
                        <div class="bar-wrap"><div class="{bar_c}" style="width:{confidence}%"></div></div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown('<div class="sec-hdr">MODEL BREAKDOWN</div>', unsafe_allow_html=True)
                    db_fake = result.get('distilbert_fake', 0)
                    db_real = result.get('distilbert_real', 0)
                    tf_fake = result.get('tfidf_fake', 0)
                    tf_real = result.get('tfidf_real', 0)

                    for name, fake_score, real_score in [
                        ('DistilBERT (70%)', db_fake, db_real),
                        ('TF-IDF + LR (30%)', tf_fake, tf_real),
                    ]:
                        dominant = 'FAKE' if fake_score > real_score else 'REAL'
                        clr = '#ff3355' if dominant == 'FAKE' else '#00ff88'
                        st.markdown(f"""
                        <div class="model-row">
                            <span class="model-name">{name}</span>
                            <span style="font-family:Rajdhani;font-weight:600;font-size:.95rem;color:{clr}">
                                {dominant} ({max(fake_score,real_score):.1f}%)
                            </span>
                        </div>""", unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    if warnings:
                        st.markdown('<div class="sec-hdr">WARNING SIGNALS</div>', unsafe_allow_html=True)
                        for w in warnings:
                            st.markdown(f'<div class="flag-red">⚠️ {w}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-ok">✓ No warning signals detected</span>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown('<div class="sec-hdr">RECOMMENDATION</div>', unsafe_allow_html=True)
                    if verdict == 'FAKE':
                        st.error("🚫 FAKE NEWS — Do not share this article. Verify through trusted sources like Reuters, BBC, or AP News.")
                    elif verdict == 'REAL':
                        st.success("✅ LIKELY REAL — Article shows characteristics of legitimate news. Always verify independently.")
                    else:
                        st.warning("⚠️ UNCERTAIN — Could not determine clearly. Cross-check with multiple trusted news sources.")

                else:
                    st.error("Fake news detection module not available. Check fakenews/ folder.")

            elif not news_text.strip():
                st.markdown("""
                <div style="text-align:center;padding:2rem;border:1px dashed #1a3a4a;border-radius:8px;">
                    <p style="font-size:2rem;margin:0">📰</p>
                    <p style="font-family:Share Tech Mono;font-size:.78rem;color:#4a7a8a">Paste a news article to analyze</p>
                </div>""", unsafe_allow_html=True)

    with fn_perf:
        st.markdown('<div class="sec-hdr">FAKE NEWS DETECTION — MODEL PERFORMANCE (Conference / IEEE)</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-family:Share Tech Mono;font-size:.72rem;color:#4a7a8a;margin-bottom:0.5rem">Static evaluation results — DistilBERT + TF-IDF+LR trained on 45,000 articles (ISOT, FakeReal, WELFake datasets).</p>', unsafe_allow_html=True)
        fig_fn = plot_fakenews_graphs()
        st.pyplot(fig_fn, use_container_width=True)
        plt.close()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0d1f2d;border:1px solid #1a3a4a;border-radius:8px;padding:1rem;">
            <p class="sec-hdr" style="margin:0 0 0.5rem 0">KEY METRICS SUMMARY — Both Models</p>
            {_metrics_badges('98.0%', '99.0%', '99.0%', '99.0%')}
            <div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:0.4rem;">
                <span class="badge-info">📦 Ensemble: 70% DistilBERT + 30% TF-IDF</span>
                <span class="badge-info">📊 TF-IDF+LR: Acc 96.5% | F1 97%</span>
                <span class="badge-info">📊 Training: 45,000 articles</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:2rem;padding:1rem;border-top:1px solid #1a3a4a">
    <p style="font-family:Share Tech Mono;font-size:.68rem;color:#4a7a8a;margin:0">
        CYBERSHIELD v1.0 | Deepfake: EfficientNet-B0 + FaceForensics++ |
        Phishing: BERT-LSTM + XGBoost + RF + Rules |
        Fake News: DistilBERT + TF-IDF | IEEE Research 2026
    </p>
</div>
""", unsafe_allow_html=True)
