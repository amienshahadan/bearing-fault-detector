import io
import os
import re
import time
from fractions import Fraction

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from scipy import signal as sps
from scipy import io as sio
from matplotlib.figure import Figure
from matplotlib import colormaps

# ══════════════════════════════════════════════════════════════
#  SETTINGS  (must match the FYP 1 training pipeline)
# ══════════════════════════════════════════════════════════════

# Google Drive file IDs of the trained models
SPECTRUM_ID    = "1R8xFvAAzRR_QT1wfGlnP8QRkrHVFue6Q"
SPECTROGRAM_ID = "1QVqIyrEUOsMO8rGzo6K4f2aO1YhvjzyU"

MODEL_DIR       = "/tmp"
SPEC_PATH       = os.path.join(MODEL_DIR, "best_spectrum_model.h5")
SPECTRO_PATH    = os.path.join(MODEL_DIR, "best_spectrogram_model.h5")
MIN_MODEL_BYTES = 1_000_000

FS_MODEL     = 12000      # CWRU sampling rate used in training
CHUNK        = 1024       # samples per image, same as training
NPERSEG      = 512        # STFT window, same as training
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

OOD_MAX_CONF   = 0.50
OOD_GAP_MIN    = 0.20
IMG_MIN_SIZE   = 50
IMG_MAX_SIZE   = 5000
MIN_AGREEMENT  = 0.60     # below this share of agreeing chunks = unreliable

_JET = (colormaps["jet"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.float32)

# ══════════════════════════════════════════════════════════════
#  IMAGE GENERATION  (same code path as the FYP 1 notebook)
# ══════════════════════════════════════════════════════════════

def _render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                pad_inches=0, facecolor="white")
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def make_spectrum_image(chunk):
    fft_result = np.fft.fft(chunk)
    freqs      = np.fft.fftfreq(len(chunk), d=1 / FS_MODEL)
    pos_freqs  = freqs[:len(freqs) // 2]
    magnitude  = np.abs(fft_result[:len(fft_result) // 2])
    magnitude  = magnitude / (np.max(magnitude) + 1e-8)

    fig = Figure(figsize=(2.24, 2.24), dpi=100)
    ax  = fig.add_subplot(111)
    ax.plot(pos_freqs, magnitude, linewidth=0.8, color='blue')
    ax.set_xlim(0, FS_MODEL / 2)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return _render(fig)

def make_spectrogram_image(chunk):
    f, t_spec, Zxx = sps.stft(chunk, fs=FS_MODEL, nperseg=NPERSEG)
    magnitude      = np.abs(Zxx)
    magnitude      = magnitude / (np.max(magnitude) + 1e-8)

    fig = Figure(figsize=(2.24, 2.24), dpi=100)
    ax  = fig.add_subplot(111)
    ax.pcolormesh(t_spec, f, magnitude, shading='gouraud', cmap='jet')
    ax.set_ylim(0, 3000)
    ax.axis('off')
    return _render(fig)

def to_model_input(img):
    """Same as training: RGB, 224x224 with NEAREST resize, /255."""
    img = img.convert('RGB').resize((224, 224), Image.Resampling.NEAREST)
    return np.asarray(img, dtype=np.float32) / 255.0

# ══════════════════════════════════════════════════════════════
#  RAW SIGNAL HANDLING
# ══════════════════════════════════════════════════════════════

def _detect_sep(text):
    """Pick the delimiter from the first data line."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in (",", ";", "\t"):
            if sep in line:
                return sep
        return r"\s+"
    return r"\s+"

def read_table(raw_bytes):
    """Read a csv/txt file into a numeric DataFrame (header optional)."""
    text = raw_bytes.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(text), sep=_detect_sep(text), engine="python",
                     header=None, dtype=str, comment="#")

    # if the first row contains text, treat it as column names
    first = pd.to_numeric(df.iloc[0], errors="coerce")
    if first.isna().any():
        df.columns = [str(c).strip() or f"col{i}" for i, c in enumerate(df.iloc[0])]
        df = df.iloc[1:]
    else:
        df.columns = [f"col{i}" for i in range(df.shape[1])]

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return df.reset_index(drop=True)

def _walk_mat(obj, path, out, depth=0):
    """Collect numeric arrays from (nested) MATLAB structs and cell arrays."""
    if depth > 6 or len(out) >= 60:
        return
    if isinstance(obj, sio.matlab.mat_struct):
        fields = list(obj._fieldnames)
        # a channel record: struct with Name + Data (e.g. Paderborn 'vibration_1')
        if "Name" in fields and "Data" in fields:
            data = np.asarray(obj.Data)
            if np.issubdtype(data.dtype, np.number) and data.size >= CHUNK:
                out[f"{path} › {obj.Name}"] = data
            return
        for f in fields:
            _walk_mat(getattr(obj, f), f"{path}.{f}", out, depth + 1)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            for i, item in enumerate(obj.flat):
                _walk_mat(item, f"{path}[{i}]", out, depth + 1)
        elif np.issubdtype(obj.dtype, np.number) and obj.size >= CHUNK:
            out[path] = obj

def read_mat(raw_bytes):
    """Return {label: array} for numeric signals long enough, incl. nested structs."""
    mat = sio.loadmat(io.BytesIO(raw_bytes), struct_as_record=False,
                      squeeze_me=True)
    out = {}
    for k, v in mat.items():
        if k.startswith("__"):
            continue
        _walk_mat(v, k, out)
    return out

def validate_signal(x):
    x = np.asarray(x, dtype=np.float64).flatten()
    n_bad = int(np.sum(~np.isfinite(x)))
    x = x[np.isfinite(x)]
    if x.size < CHUNK:
        return None, f"Signal too short: {x.size} samples (need at least {CHUNK})."
    if np.std(x) < 1e-12:
        return None, "Signal is flat (zero variance). Wrong column or empty channel?"
    msg = f"{n_bad} invalid values (NaN/inf) were removed." if n_bad else "OK"
    return x, msg

def to_model_rate(x, fs_in):
    if abs(fs_in - FS_MODEL) < 1e-6:
        return x, False
    frac = Fraction(FS_MODEL / fs_in).limit_denominator(1000)
    return sps.resample_poly(x, frac.numerator, frac.denominator), True

def chunk_signal(x, max_chunks):
    n_total = len(x) // CHUNK
    if n_total <= max_chunks:
        idx = np.arange(n_total)
    else:
        idx = np.unique(np.linspace(0, n_total - 1, max_chunks).astype(int))
    return [x[i * CHUNK:(i + 1) * CHUNK] for i in idx], n_total

def predict_signal(model, chunks, maker):
    imgs  = [maker(c) for c in chunks]
    batch = np.stack([to_model_input(i) for i in imgs])
    probs = model.predict(batch, batch_size=32, verbose=0)
    return probs, imgs[0]

def aggregate(probs):
    """Majority vote over chunks; ties broken by mean probability."""
    probs  = np.atleast_2d(probs)
    votes  = np.bincount(np.argmax(probs, axis=1), minlength=len(CLASS_LABELS))
    mean   = probs.mean(axis=0)
    top    = np.flatnonzero(votes == votes.max())
    winner = int(top[np.argmax(mean[top])])
    return winner, votes, mean

# ══════════════════════════════════════════════════════════════
#  IMAGE UPLOAD VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_image_size(img):
    w, h = img.size
    if w < IMG_MIN_SIZE or h < IMG_MIN_SIZE:
        return False, f"Image too small ({w}x{h}px). Minimum: {IMG_MIN_SIZE}x{IMG_MIN_SIZE}px."
    if w > IMG_MAX_SIZE or h > IMG_MAX_SIZE:
        return False, f"Image too large ({w}x{h}px). Maximum: {IMG_MAX_SIZE}x{IMG_MAX_SIZE}px."
    return True, "OK"

def jet_match_ratio(arr):
    """Share of pixels lying on the matplotlib 'jet' colour curve."""
    px = arr.reshape(-1, 3).astype(np.float32)[::6]
    d  = np.sqrt(((px[:, None, :] - _JET[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return float((d < 100).mean())

def validate_image_content(img, img_type):
    arr   = np.array(img.convert('RGB').resize((224, 224)), dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    total = 224 * 224

    white_ratio = np.sum((r > 220) & (g > 220) & (b > 220)) / total

    if "Spectrum" in img_type:
        # Training spectrum images: white background + thin blue line only
        white_blue = np.sum((np.abs(r - g) < 30) & (b > 180)) / total
        blue_line  = np.sum((b > 150) & (r < 100) & (g < 100)) / total
        if white_ratio < 0.40 or white_blue < 0.97 or blue_line < 0.003:
            return False, (
                "Image does not look like a Spectrum (FFT) image.<br>"
                "Expected: white background with a thin blue spectrum line only.<br>"
                "Tip: upload the raw signal in the <b>Raw Signal</b> tab and "
                "let the app generate the image."
            )
    else:
        # Training spectrogram images: jet colormap fills the whole image
        jet     = jet_match_ratio(arr)
        pix_std = float(arr.reshape(-1, 3).std(axis=0).mean())
        if jet < 0.75 or white_ratio > 0.30 or pix_std < 1:
            return False, (
                f"Image does not look like a Spectrogram (STFT) image.<br>"
                f"Jet-colormap match: {jet*100:.0f}% — at least 75% required, "
                f"and the image must contain visible structure.<br>"
                f"Tip: upload the raw signal in the <b>Raw Signal</b> tab and "
                f"let the app generate the image."
            )
    return True, "OK"

# ══════════════════════════════════════════════════════════════
#  DIAGNOSIS HELPERS
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

def detect_ood(mean_scores):
    s = np.sort(np.asarray(mean_scores))
    return bool(s[-1] < OOD_MAX_CONF and (s[-1] - s[-2]) < OOD_GAP_MIN)

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def download_model(file_id, output_path, label):
    if os.path.exists(output_path) and os.path.getsize(output_path) > MIN_MODEL_BYTES:
        return
    import gdown
    with st.spinner(f"Downloading {label} model from Google Drive..."):
        gdown.download(f"https://drive.google.com/uc?id={file_id}",
                       output_path, quiet=True)
    if not os.path.exists(output_path) or os.path.getsize(output_path) < MIN_MODEL_BYTES:
        if os.path.exists(output_path):
            os.remove(output_path)
        st.error(f"❌ Failed to download the {label} model. Check that the "
                 f"Google Drive file is shared as 'Anyone with the link'.")
        st.stop()

@st.cache_resource
def load_models():
    import tensorflow as tf
    download_model(SPECTRUM_ID,    SPEC_PATH,    "Spectrum")
    download_model(SPECTROGRAM_ID, SPECTRO_PATH, "Spectrogram")
    return (tf.keras.models.load_model(SPEC_PATH,    compile=False),
            tf.keras.models.load_model(SPECTRO_PATH, compile=False))

# ══════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1528 50%, #0a0e1a 100%); }
    .title-box { text-align:center; padding:2rem 1rem; border-bottom:1px solid #1e3a5f; margin-bottom:2rem; }
    .title-box h1 { font-family:'Share Tech Mono',monospace; font-size:2.2rem; color:#00d4ff;
                    letter-spacing:3px; margin:0; text-shadow:0 0 20px rgba(0,212,255,0.5); }
    .title-box p { color:#4a7fa5; font-size:1rem; letter-spacing:2px; margin-top:0.5rem; }
    .result-box { background:linear-gradient(135deg,#0d1f35,#0a1628); border:1px solid #1e3a5f;
                  border-left:4px solid #00d4ff; border-radius:8px; padding:1.5rem; margin:1rem 0; }
    .fault-label { font-family:'Share Tech Mono',monospace; font-size:2rem; color:#00d4ff;
                   text-shadow:0 0 10px rgba(0,212,255,0.3); }
    .healthy  { color:#00ff88 !important; border-left-color:#00ff88 !important; }
    .critical { color:#ff4444 !important; border-left-color:#ff4444 !important; }
    .moderate { color:#ffaa00 !important; border-left-color:#ffaa00 !important; }
    .mild     { color:#00d4ff !important; border-left-color:#00d4ff !important; }
    .metric-row { display:flex; justify-content:space-between; gap:1rem; padding:0.5rem 0;
                  border-bottom:1px solid #1e3a5f; font-size:1rem; }
    .metric-label { color:#4a7fa5; letter-spacing:1px; white-space:nowrap; }
    .metric-value { color:#e0e6f0; font-weight:600; text-align:right; }
    .warning-box { background:linear-gradient(135deg,#1a0e0e,#1a1208); border:1px solid #5f2a1e;
                   border-left:4px solid #ff4444; border-radius:8px; padding:1.2rem 1.5rem; margin:1rem 0;
                   font-family:'Share Tech Mono',monospace; color:#ff4444; font-size:0.9rem;
                   letter-spacing:1px; line-height:1.8; }
    .note-box { background:#0d1f35; border:1px solid #1e3a5f; border-left:4px solid #ffaa00;
                border-radius:8px; padding:0.9rem 1.2rem; margin:0.5rem 0 1rem 0;
                font-family:'Share Tech Mono',monospace; color:#c9a24a; font-size:0.8rem;
                letter-spacing:1px; line-height:1.7; }
    .stButton > button { background:linear-gradient(135deg,#00d4ff,#0096cc); color:#0a0e1a;
                         font-family:'Share Tech Mono',monospace; font-weight:700; letter-spacing:2px;
                         border:none; padding:0.75rem 2rem; border-radius:4px; width:100%; font-size:1rem; }
    .stButton > button:hover { background:linear-gradient(135deg,#00ffcc,#00d4ff);
                               box-shadow:0 4px 20px rgba(0,212,255,0.4); }
    .stRadio label { color:#e0e6f0 !important; font-size:1rem; letter-spacing:1px; }
    div[data-testid="stFileUploader"] { background:#0d1f35; border-radius:8px; padding:1rem; }
    .scanline { font-family:'Share Tech Mono',monospace; color:#4a7fa5; font-size:0.75rem;
                letter-spacing:1px; text-align:center; padding:0.5rem; }
</style>
"""

def warn(title, body):
    st.markdown(f'<div class="warning-box">⚠️ {title}<br><br>{body}</div>',
                unsafe_allow_html=True)

def show_chart(scores):
    colors = ['#00d4ff', '#7c3aed', '#00ff88', '#ffaa00']
    fig = go.Figure(go.Bar(
        x=list(scores.values()), y=list(scores.keys()), orientation='h',
        marker_color=colors, text=[f"{v:.1f}%" for v in scores.values()],
        textposition='outside'))
    fig.update_layout(
        paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1f35',
        font=dict(color='#e0e6f0', family='Share Tech Mono'),
        xaxis=dict(range=[0, 115], gridcolor='#1e3a5f'),
        yaxis=dict(gridcolor='#1e3a5f'),
        height=250, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

def render_result(model_name, probs):
    """Show one diagnosis. probs has shape (n_chunks, 4)."""
    probs = np.atleast_2d(probs)
    n     = probs.shape[0]
    winner, votes, mean = aggregate(probs)
    pred_class = CLASS_LABELS[winner]
    confidence = float(mean[winner] * 100)
    agreement  = float(votes[winner] / n)

    st.markdown(f"### 📊 DIAGNOSIS — {model_name}")

    if detect_ood(mean):
        warn("UNCERTAIN PREDICTION",
             "The model confidence is too low and evenly spread across all "
             "classes. The input may not be a valid bearing vibration signal "
             "or is too different from the training data.")
        show_chart({l: float(s * 100) for l, s in zip(CLASS_LABELS, mean)})
        return pred_class

    severity_label, color = get_severity(pred_class, confidence)
    vote_text = f"{votes[winner]} of {n} chunks" if n > 1 else "single image"

    st.markdown(f"""
    <div class="result-box {color}">
        <div class="fault-label">{pred_class.replace('_', ' ').upper()}</div>
        <div style="margin-top:1rem">
            <div class="metric-row"><span class="metric-label">MODEL USED</span>
                <span class="metric-value">{model_name}</span></div>
            <div class="metric-row"><span class="metric-label">CONFIDENCE (MEAN)</span>
                <span class="metric-value">{confidence:.1f}%</span></div>
            <div class="metric-row"><span class="metric-label">CHUNK AGREEMENT</span>
                <span class="metric-value">{vote_text} ({agreement*100:.0f}%)</span></div>
            <div class="metric-row"><span class="metric-label">SEVERITY</span>
                <span class="metric-value">{severity_label}</span></div>
            <div class="metric-row"><span class="metric-label">ACTION</span>
                <span class="metric-value">{FAULT_ACTION[pred_class]}</span></div>
            <div class="metric-row"><span class="metric-label">POSSIBLE CAUSES</span>
                <span class="metric-value">{FAULT_CAUSES[pred_class]}</span></div>
            <div class="metric-row" style="border:none"><span class="metric-label">ADVICE</span>
                <span class="metric-value">{FAULT_ADVICE[pred_class]}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if n > 1 and agreement < MIN_AGREEMENT:
        warn("CHUNKS DISAGREE",
             f"Only {agreement*100:.0f}% of chunks agree on the winning class. "
             f"Treat this result as unreliable.")

    if n > 1:
        st.caption("Votes per class: " + ", ".join(
            f"{l} = {int(v)}" for l, v in zip(CLASS_LABELS, votes)))
    show_chart({l: float(s * 100) for l, s in zip(CLASS_LABELS, mean)})
    return pred_class

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════

def tab_raw_signal(spectrum_model, spectrogram_model):
    st.markdown(
        '<div class="note-box">Models were trained only on CWRU data '
        '(SKF 6205 bearing, drive-end sensor, 12 kHz). Results on other '
        'machines or sensors are NOT validated. The app converts your signal '
        'into spectrum / spectrogram images using the same pipeline as '
        'training.</div>', unsafe_allow_html=True)

    up = st.file_uploader("Upload vibration signal", type=["csv", "txt", "mat"],
                          key="raw_upload")
    if not up:
        st.markdown('<div class="scanline">↑ UPLOAD A .CSV / .TXT / .MAT SIGNAL FILE</div>',
                    unsafe_allow_html=True)
        return

    # ── read file and pick the signal column / variable ──────
    name       = up.name.lower()
    default_fs = float(FS_MODEL)
    try:
        if name.endswith(".mat"):
            cand = read_mat(up.getvalue())
            if not cand:
                warn("NO SIGNAL FOUND",
                     f"No numeric signal with at least {CHUNK} samples in this .mat file.")
                return
            keys    = list(cand.keys())
            default = next((i for i, k in enumerate(keys) if "DE_time" in k),
                      next((i for i, k in enumerate(keys) if "vibration" in k.lower()), 0))
            var     = st.selectbox("Signal to analyse", keys, index=default)
            raw     = cand[var]
            if "vibration" in var.lower():
                default_fs = 64000.0
                st.caption("Paderborn 'vibration' channel is sampled at 64 kHz — "
                           "sampling rate below was set to 64000 Hz for you. "
                           "Change it if your file is different.")
        else:
            df = read_table(up.getvalue())
            if df.shape[1] == 0:
                warn("NO NUMERIC DATA", "Could not find numeric columns in this file.")
                return
            col = st.selectbox("Column to analyse", list(df.columns)) \
                  if df.shape[1] > 1 else df.columns[0]
            raw = df[col].to_numpy()
    except Exception as e:
        warn("FILE READ ERROR", f"Could not read this file: {e}")
        return

    x, msg = validate_signal(raw)
    if x is None:
        warn("INVALID SIGNAL", msg)
        return
    if msg != "OK":
        st.caption(msg)

    c1, c2 = st.columns(2)
    with c1:
        fs_in = st.number_input("Sampling rate of your signal (Hz)",
                                min_value=100.0, max_value=200000.0,
                                value=default_fs, step=100.0)
    with c2:
        max_chunks = st.slider("Max chunks to analyse", 5, 200, 50, step=5,
                               help=f"Each chunk = {CHUNK} samples at {FS_MODEL} Hz. "
                                    f"More chunks = more reliable but slower.")

    rep = st.radio("Representation",
                   ["Spectrogram (STFT)", "Spectrum (FFT)", "Both"],
                   horizontal=True)

    x12, resampled = to_model_rate(x, fs_in)
    n_total = len(x12) // CHUNK
    info = (f"{len(x):,} samples → {len(x12):,} samples at {FS_MODEL} Hz "
            f"({n_total} full chunks of {CHUNK})")
    if resampled:
        info += f" — resampled from {fs_in:,.0f} Hz"
    st.caption(info)

    if fs_in < FS_MODEL:
        warn("LOW SAMPLING RATE",
             f"Your rate ({fs_in:,.0f} Hz) is below {FS_MODEL:,} Hz. Frequencies "
             f"above {fs_in/2:,.0f} Hz are missing, so the images will differ from training.")
    if n_total < 1:
        warn("SIGNAL TOO SHORT",
             f"After conversion to {FS_MODEL} Hz the signal has less than {CHUNK} samples.")
        return

    # quick look at the signal so the user can verify the right column
    show = x12[:3 * CHUNK]
    fig = go.Figure(go.Scatter(y=show, mode="lines", line=dict(color="#00d4ff", width=1)))
    fig.update_layout(paper_bgcolor='#0a0e1a', plot_bgcolor='#0d1f35',
                      font=dict(color='#e0e6f0', family='Share Tech Mono'),
                      height=200, margin=dict(l=20, r=20, t=20, b=20),
                      xaxis=dict(title="sample", gridcolor='#1e3a5f'),
                      yaxis=dict(gridcolor='#1e3a5f'), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    if not st.button("🔍 ANALYZE SIGNAL"):
        return

    chunks, n_total = chunk_signal(x12, max_chunks)
    runs = []
    if rep in ("Spectrogram (STFT)", "Both"):
        runs.append(("Spectrogram Model (STFT)", spectrogram_model, make_spectrogram_image))
    if rep in ("Spectrum (FFT)", "Both"):
        runs.append(("Spectrum Model (FFT)", spectrum_model, make_spectrum_image))

    results = []
    for title, model, maker in runs:
        with st.spinner(f"Generating images and running {title} on {len(chunks)} chunks..."):
            probs, sample = predict_signal(model, chunks, maker)
        st.markdown("---")
        st.image(sample, caption=f"Generated image for chunk 1 ({title})", width=224)
        results.append(render_result(title, probs))

    if len(results) == 2 and results[0] != results[1]:
        warn("MODELS DISAGREE",
             f"Spectrogram model says {results[0]}, Spectrum model says {results[1]}.")

def tab_image_upload(spectrum_model, spectrogram_model):
    st.markdown(
        '<div class="note-box">Only images in the exact training style are accepted: '
        'FFT spectrum (white background, blue line) or STFT spectrogram (jet colormap, '
        'no axes). For anything else, use the Raw Signal tab.</div>',
        unsafe_allow_html=True)

    img_type = st.radio("Image type", ["Spectrum (FFT)", "Spectrogram (STFT)"],
                        horizontal=True, key="img_type")
    up = st.file_uploader("Upload spectrum or spectrogram image",
                          type=['png', 'jpg', 'jpeg'], key="img_upload")
    if not up:
        return

    img = Image.open(up)
    st.image(img, caption="Uploaded Image", width=224)

    ok, msg = validate_image_size(img)
    if not ok:
        warn("IMAGE SIZE ERROR", msg)
        return
    ok, msg = validate_image_content(img, img_type)
    if not ok:
        warn("INVALID IMAGE TYPE", msg)
        return

    if st.button("🔍 ANALYZE IMAGE"):
        model = spectrum_model if "Spectrum" in img_type else spectrogram_model
        name  = "Spectrum Model (FFT)" if "Spectrum" in img_type else "Spectrogram Model (STFT)"
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            probs = model.predict(to_model_input(img)[None, ...], verbose=0)
        render_result(name, probs)

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(page_title="Bearing Fault Detector",
                       page_icon="🔧", layout="centered")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown("""
    <div class="title-box">
        <h1>⚙ BEARING FAULT DETECTOR</h1>
        <p>POWERED BY CNN + MOBILENETV2 | CWRU DATASET</p>
    </div>
    """, unsafe_allow_html=True)

    spectrum_model, spectrogram_model = load_models()
    st.markdown('<div class="scanline">✓ MODELS LOADED — SYSTEM READY</div>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Raw Signal", "🖼 Image Upload"])
    with tab1:
        tab_raw_signal(spectrum_model, spectrogram_model)
    with tab2:
        tab_image_upload(spectrum_model, spectrogram_model)

if __name__ == "__main__":
    main()
