import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time
import os
import gdown

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title = "Bearing Fault Detector",
    page_icon  = "🔧",
    layout     = "centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a0e1a 100%);
    }
    .title-box {
        text-align: center;
        padding: 2rem 1rem;
        border-bottom: 1px solid #1e3a5f;
        margin-bottom: 2rem;
    }
    .title-box h1 {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.2rem;
        color: #00d4ff;
        letter-spacing: 3px;
        margin: 0;
        text-shadow: 0 0 20px rgba(0,212,255,0.5);
    }
    .title-box p {
        color: #4a7fa5;
        font-size: 1rem;
        letter-spacing: 2px;
        margin-top: 0.5rem;
    }
    .result-box {
        background: linear-gradient(135deg, #0d1f35, #0a1628);
        border: 1px solid #1e3a5f;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .fault-label {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2rem;
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0,212,255,0.3);
    }
    .healthy  { color: #00ff88 !important; border-left-color: #00ff88 !important; }
    .critical { color: #ff4444 !important; border-left-color: #ff4444 !important; }
    .moderate { color: #ffaa00 !important; border-left-color: #ffaa00 !important; }
    .mild     { color: #00d4ff !important; border-left-color: #00d4ff !important; }
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #1e3a5f;
        font-size: 1rem;
    }
    .metric-label { color: #4a7fa5; letter-spacing: 1px; }
    .metric-value { color: #e0e6f0; font-weight: 600; }
    .warning-box {
        background: linear-gradient(135deg, #1a0e0e, #1a1208);
        border: 1px solid #5f2a1e;
        border-left: 4px solid #ff4444;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
        font-family: 'Share Tech Mono', monospace;
        color: #ff4444;
        font-size: 0.9rem;
        letter-spacing: 1px;
        line-height: 1.8;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0096cc);
        color: #0a0e1a;
        font-family: 'Share Tech Mono', monospace;
        font-weight: 700;
        letter-spacing: 2px;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 4px;
        width: 100%;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffcc, #00d4ff);
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,212,255,0.4);
    }
    .stRadio label { color: #e0e6f0 !important; font-size: 1rem; letter-spacing: 1px; }
    .upload-section {
        border: 2px dashed #1e3a5f;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    div[data-testid="stFileUploader"] {
        background: #0d1f35;
        border-radius: 8px;
        padding: 1rem;
    }
    .scanline {
        font-family: 'Share Tech Mono', monospace;
        color: #4a7fa5;
        font-size: 0.75rem;
        letter-spacing: 1px;
        text-align: center;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  SETTINGS
# ══════════════════════════════════════════════════════════════

# Google Drive file IDs
SPECTRUM_ID    = "1R8xFvAAzRR_QT1wfGlnP8QRkrHVFue6Q"
SPECTROGRAM_ID = "1QVqIyrEUOsMO8rGzo6K4f2aO1YhvjzyU"

# Models are downloaded here at first start
MODEL_DIR    = "/tmp"
SPEC_PATH    = os.path.join(MODEL_DIR, "best_spectrum_model.h5")
SPECTRO_PATH = os.path.join(MODEL_DIR, "best_spectrogram_model.h5")
MIN_MODEL_BYTES = 1_000_000   # a real model is ~27 MB; smaller = bad download

CLASS_LABELS = ['Ball', 'Inner_Race', 'Normal', 'Outer_Race']

FAULT_CAUSES = {
    "Normal"     : "N/A — No fault detected",
    "Ball"       : "Rolling element fatigue, contamination, improper lubrication",
    "Inner_Race" : "Inner raceway fatigue, shaft misalignment, improper installation",
    "Outer_Race" : "Outer raceway fatigue, overloading, contamination ingress"
}
FAULT_ACTION = {
    "Normal"     : "No action needed. Bearing is operating normally.",
    "Ball"       : "Schedule inspection within 1 week. Check lubrication.",
    "Inner_Race" : "Immediate inspection required. Check shaft alignment.",
    "Outer_Race" : "Immediate inspection required. Check radial load capacity."
}
FAULT_ADVICE = {
    "Normal"     : "Continue regular maintenance schedule.",
    "Ball"       : "Consider bearing replacement if vibration level increases.",
    "Inner_Race" : "Stop machine if confidence is high to prevent shaft damage.",
    "Outer_Race" : "Check radial load against bearing rated capacity."
}

OOD_MAX_CONF = 0.50
OOD_GAP_MIN  = 0.20
IMG_MIN_SIZE = 50
IMG_MAX_SIZE = 5000

# ══════════════════════════════════════════════════════════════
#  FUNCTIONS
# ══════════════════════════════════════════════════════════════

def get_severity(pred_class, confidence):
    if pred_class == "Normal":
        return "✅ HEALTHY", "healthy"
    elif confidence >= 90:
        return "🚨 CRITICAL", "critical"
    elif confidence >= 75:
        return "⚠️ MODERATE", "moderate"
    else:
        return "🔍 MILD", "mild"

def validate_image_size(img):
    w, h = img.size
    if w < IMG_MIN_SIZE or h < IMG_MIN_SIZE:
        return False, f"Image too small ({w}x{h}px). Minimum: {IMG_MIN_SIZE}x{IMG_MIN_SIZE}px."
    if w > IMG_MAX_SIZE or h > IMG_MAX_SIZE:
        return False, f"Image too large ({w}x{h}px). Maximum: {IMG_MAX_SIZE}x{IMG_MAX_SIZE}px."
    return True, "OK"

def validate_image_content(img, img_type):
    img_rgb = img.convert('RGB').resize((224, 224))
    arr     = np.array(img_rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total   = 224 * 224

    white_ratio = np.sum((r > 220) & (g > 220) & (b > 220)) / total
    dark_ratio  = np.sum((r < 50)  & (g < 50)  & (b < 50))  / total
    color_ratio = np.sum(np.std(arr, axis=2) > 30) / total
    blue_ratio  = np.sum((b > 150) & (r < 100) & (g < 100)) / total
    red_ratio   = np.sum((r > 150) & (g < 100) & (b < 100)) / total

    if "Spectrum" in img_type:
        if white_ratio < 0.40:
            return False, (
                f"Image does not look like a Spectrum (FFT) image.<br>"
                f"White background detected: {white_ratio*100:.1f}% — minimum 40% required.<br>"
                f"Please upload a proper FFT spectrum image generated from bearing vibration data."
            )
        if blue_ratio < 0.005:
            return False, (
                f"Image does not contain a blue spectrum line.<br>"
                f"Blue content detected: {blue_ratio*100:.3f}% — minimum 0.5% required.<br>"
                f"Please upload a proper FFT spectrum image generated from bearing vibration data."
            )

    elif "Spectrogram" in img_type:
        if dark_ratio < 0.20:
            return False, (
                f"Image does not look like a Spectrogram (STFT) image.<br>"
                f"Dark background detected: {dark_ratio*100:.1f}% — minimum 20% required.<br>"
                f"Please upload a proper STFT spectrogram image generated from bearing vibration data."
            )
        if color_ratio < 0.10:
            return False, (
                f"Image does not contain enough colorful content for a spectrogram.<br>"
                f"Please upload a proper STFT spectrogram image generated from bearing vibration data."
            )
        if red_ratio < 0.001:
            return False, (
                f"Image does not contain jet colormap warm colors.<br>"
                f"Please upload a proper STFT spectrogram image generated from bearing vibration data."
            )

    return True, "OK"

def detect_ood(scores):
    max_score    = float(np.max(scores))
    second_score = float(np.sort(scores)[-2])
    if max_score < OOD_MAX_CONF and (max_score - second_score) < OOD_GAP_MIN:
        return True
    return False

def download_model(file_id, output_path, label):
    """Download a model from Google Drive if not already present."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > MIN_MODEL_BYTES:
        return
    with st.spinner(f"Downloading {label} model from Google Drive..."):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=True)

    # Verify the download is a real model, not an HTML error page
    if not os.path.exists(output_path) or os.path.getsize(output_path) < MIN_MODEL_BYTES:
        if os.path.exists(output_path):
            os.remove(output_path)
        st.error(
            f"❌ Failed to download the {label} model. "
            f"Check that the Google Drive file is shared as 'Anyone with the link'."
        )
        st.stop()

@st.cache_resource
def load_models():
    download_model(SPECTRUM_ID,    SPEC_PATH,    "Spectrum")
    download_model(SPECTROGRAM_ID, SPECTRO_PATH, "Spectrogram")

    spectrum_model    = tf.keras.models.load_model(SPEC_PATH,    compile=False)
    spectrogram_model = tf.keras.models.load_model(SPECTRO_PATH, compile=False)
    return spectrum_model, spectrogram_model

def predict(model, img):
    img        = img.resize((224, 224)).convert('RGB')
    arr        = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred       = model.predict(arr, verbose=0)
    pred_class = CLASS_LABELS[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)
    return pred_class, confidence, pred[0]

def show_chart(scores):
    colors = ['#00d4ff', '#7c3aed', '#00ff88', '#ffaa00']
    fig = go.Figure(go.Bar(
        x=list(scores.values()), y=list(scores.keys()),
        orientation='h', marker_color=colors,
        text=[f"{v:.1f}%" for v in scores.values()],
        textposition='outside'
    ))
    fig.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1f35',
        font=dict(color='#e0e6f0', family='Share Tech Mono'),
        xaxis=dict(range=[0, 115], gridcolor='#1e3a5f'),
        yaxis=dict(gridcolor='#1e3a5f'),
        height=250, margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════

st.markdown("""
<div class="title-box">
    <h1>⚙ BEARING FAULT DETECTOR</h1>
    <p>POWERED BY CNN + MOBILENETV2 | CWRU DATASET</p>
</div>
""", unsafe_allow_html=True)

# Load models (auto-download from Google Drive on first start)
spectrum_model, spectrogram_model = load_models()

st.markdown(
    '<div class="scanline">✓ MODELS LOADED — SYSTEM READY</div>',
    unsafe_allow_html=True
)
st.markdown("---")

st.markdown("### 📡 SELECT IMAGE TYPE")
img_type = st.radio(
    "Select image type",
    options=["Spectrum (FFT)", "Spectrogram (STFT)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("### 📁 UPLOAD BEARING IMAGE")
uploaded = st.file_uploader(
    "Upload spectrum or spectrogram image",
    type=['png', 'jpg', 'jpeg']
)

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Step 1 — size check
    valid_size, size_msg = validate_image_size(img)
    if not valid_size:
        st.markdown(
            f'<div class="warning-box">⚠️ IMAGE SIZE ERROR<br><br>{size_msg}</div>',
            unsafe_allow_html=True
        )
        st.stop()

    # Step 2 — visual content check
    valid_content, content_msg = validate_image_content(img, img_type)
    if not valid_content:
        st.markdown(
            f'<div class="warning-box">⚠️ INVALID IMAGE TYPE<br><br>{content_msg}</div>',
            unsafe_allow_html=True
        )
        st.stop()

    st.markdown("---")

    if st.button("🔍 ANALYZE BEARING"):
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            if "Spectrum" in img_type:
                model      = spectrum_model
                model_name = "Spectrum Model (FFT)"
            else:
                model      = spectrogram_model
                model_name = "Spectrogram Model (STFT)"
            pred_class, confidence, scores = predict(model, img)
            is_ood = detect_ood(scores)

        # Step 3 — confidence-based OOD check
        if is_ood:
            st.markdown("""
            <div class="warning-box">
                ⚠️ UNCERTAIN PREDICTION DETECTED<br><br>
                The model confidence is too low and evenly distributed
                across all classes.<br><br>
                This usually means the uploaded image is not a valid
                bearing vibration signal image, or the image quality
                is insufficient for reliable diagnosis.<br><br>
                Please ensure the image is a proper FFT spectrum or
                STFT spectrogram generated from bearing vibration data.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

        severity_label, color = get_severity(pred_class, confidence)

        st.markdown("---")
        st.markdown("### 📊 DIAGNOSIS RESULT")
        st.markdown(f"""
        <div class="result-box {color}">
            <div class="fault-label">{pred_class.replace('_', ' ').upper()}</div>
            <div style="margin-top:1rem">
                <div class="metric-row">
                    <span class="metric-label">MODEL USED</span>
                    <span class="metric-value">{model_name}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">CONFIDENCE</span>
                    <span class="metric-value">{confidence:.1f}%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">SEVERITY</span>
                    <span class="metric-value">{severity_label}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">ACTION</span>
                    <span class="metric-value">{FAULT_ACTION[pred_class]}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">POSSIBLE CAUSES</span>
                    <span class="metric-value">{FAULT_CAUSES[pred_class]}</span>
                </div>
                <div class="metric-row" style="border:none">
                    <span class="metric-label">ADVICE</span>
                    <span class="metric-value">{FAULT_ADVICE[pred_class]}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 CONFIDENCE BREAKDOWN")
        show_chart({
            label: float(score * 100)
            for label, score in zip(CLASS_LABELS, scores)
        })

        st.markdown(f"""
        <div class="scanline" style="margin-top:2rem">
            ANALYSIS COMPLETE | {pred_class.upper()} DETECTED |
            {confidence:.1f}% CONFIDENCE
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-section">
        <p style="color:#4a7fa5;
                  font-family:'Share Tech Mono',monospace;
                  letter-spacing:2px">
            ↑ UPLOAD AN IMAGE TO BEGIN ANALYSIS
        </p>
    </div>
    """, unsafe_allow_html=True)
