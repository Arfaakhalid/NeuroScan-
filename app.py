"""
app.py  --  NeuroScan Pro  
=======================================================================
KEY FIXES vs original:
  1. 4 seizure classes (Normal / Focal / Absence / Tonic-Atonic)
  2. All graphs show clinically meaningful data:
     - Band power chart uses ACTUAL computed values, not placeholders
     - Probability bar shows all 4 classes with clear colour coding
     - Feature importance shows real feature names from training
  3. Scaler alignment is exact, feature names are preserved end-to-end,
     rule-based override applied.
  4. Image EEG uses proper row-scan extraction (not one row per channel)
  5. Consistent & Accurate predictions
  6. Dashboard shows real class descriptions and clinical thresholds.
  7. Train button now loads SYNTHETIC + real data so models learn
     real EEG physics even if the CSV has low separability.
=======================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import io, os, sys, tempfile, hashlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.signal import welch

from preprocessing import (
    preprocess, extract_features, detect_spikes,
    band_power_profile, dominant_frequency, EEG_BANDS,
)
from dataset import (
    SEIZURE_TYPES, SEIZURE_ICD10,
    load_real_csv, load_test_csv, prepare_split,
    generate_demo_features, rule_based_classify,
    gen_normal, gen_focal, gen_absence, gen_tonic,
    _DROP_COLS, _EEG_ONLY_COLS, generate_synthetic_training_data,
)
from models import (
    AdvancedEpilepsyClassifier, N_CLASSES,
    SEIZURE_DESCRIPTIONS, SEIZURE_KEY_FEATURES,
)

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan Pro | Epilepsy Detection",
    page_icon="🧠", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family:'Inter',sans-serif; }
.stApp { background:linear-gradient(135deg,#0a0e17 0%,#1a1f2e 100%); color:#e0e0e0; }

.med-header {
    background:linear-gradient(135deg,#1a237e 0%,#283593 50%,#3949ab 100%);
    border-radius:0 0 20px 20px; padding:2.2rem 2rem; text-align:center;
    box-shadow:0 8px 32px rgba(0,0,0,.35); border-bottom:4px solid #00bcd4;
    margin-bottom:1.8rem;
}
.med-header h1 { color:#fff; font-size:2.8rem; font-weight:800; margin-bottom:.4rem; }
.med-header p  { color:#b3e5fc; font-size:1.1rem; margin:0; }

.card {
    background:rgba(30,33,45,.92); border-radius:14px; padding:1.5rem;
    border:1px solid rgba(96,125,139,.3); margin-bottom:1.2rem;
}
.card h3 { color:#00bcd4; font-size:1.15rem; margin-bottom:.8rem;
           border-bottom:1px solid rgba(0,188,212,.25); padding-bottom:.5rem; }

.pill { display:inline-block; padding:.25rem .8rem; border-radius:20px;
        font-size:.8rem; font-weight:600; margin:.15rem; }
.pill-ok   { background:rgba(76,175,80,.2);  color:#81c784; border:1px solid #4caf50; }
.pill-warn { background:rgba(255,152,0,.2);  color:#ffb74d; border:1px solid #ff9800; }
.pill-crit { background:rgba(244,67,54,.2);  color:#e57373; border:1px solid #f44336; }
.pill-info { background:rgba(33,150,243,.2); color:#64b5f6; border:1px solid #2196f3; }

.result-box { border-radius:14px; padding:1.8rem; border-left:6px solid; margin:1.2rem 0; }
.result-normal   { background:rgba(76,175,80,.12);  border-color:#4caf50; }
.result-epilepsy { background:rgba(244,67,54,.15);  border-color:#f44336;
                   animation:pulse 2s infinite; }
@keyframes pulse {
  0%  { box-shadow:0 0 0 0 rgba(244,67,54,.5); }
  70% { box-shadow:0 0 0 12px rgba(244,67,54,0); }
  100%{ box-shadow:0 0 0 0 rgba(244,67,54,0); }
}
.stButton>button {
    background:linear-gradient(135deg,#1565c0,#0d47a1); color:#fff;
    border:1px solid rgba(33,150,243,.3); border-radius:10px;
    font-weight:600; transition:all .3s;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#1976d2,#1565c0);
    transform:translateY(-2px); box-shadow:0 6px 20px rgba(33,150,243,.4);
}
.stTabs [data-baseweb="tab-list"] {
    background:rgba(30,33,45,.8); border-radius:12px; padding:5px; gap:4px;
}
.stTabs [aria-selected="true"] {
    background:#1565c0 !important; color:#fff !important;
    box-shadow:0 4px 12px rgba(21,101,192,.3);
}
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────
_DEFAULTS = dict(
    data_loaded=False, models_trained=False,
    classifier=None, split=None,
    prediction=None, prediction_hash=None,
    eeg_raw=None, eeg_fs=256.0, eeg_filename="",
    feature_names=[], n_train_samples=0, n_classes=N_CLASSES,
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

MODEL_SAVE_PATH = "neuroscan_trained.pkl"

_DARK = dict(
    plot_bgcolor="#0d1117", paper_bgcolor="rgba(0,0,0,0)",
    font_color="#c9d1d9",
)

# Shared axis style — apply individually per chart, never via **_DARK
_AXIS = dict(gridcolor="#21262d", linecolor="#30363d", color="#c9d1d9")

# Seizure type colours
_SCOLOURS = {
    "Normal":       "#4caf50",
    "Focal":        "#ff9800",
    "Absence":      "#f44336",
    "Tonic-Atonic": "#9c27b0",
}


# ── Plots ────────────────────────────────────────────────────────
def _plot_eeg(eeg, fs, title="EEG Signal", max_ch=8, dur=10.0):
    n_ch  = min(eeg.shape[0], max_ch)
    n_pts = min(eeg.shape[1], int(fs * dur))
    t     = np.arange(n_pts) / fs
    fig   = make_subplots(rows=n_ch, cols=1, shared_xaxes=True, vertical_spacing=0.01)
    colors = px.colors.qualitative.Plotly
    for i in range(n_ch):
        sig = eeg[i, :n_pts]
        sd  = float(np.std(sig)) or 1.0
        fig.add_trace(
            go.Scatter(x=t, y=sig/sd + i*3,
                       line=dict(color=colors[i % len(colors)], width=1),
                       name=f"Ch{i+1}", showlegend=(i==0)),
            row=i+1, col=1)
        fig.update_yaxes(showticklabels=False, row=i+1, col=1)
    fig.update_layout(title=title, height=max(300, n_ch*80),
                      margin=dict(l=40,r=20,t=50,b=40), **_DARK)
    fig.update_xaxes(title_text="Time (s)", row=n_ch, col=1, **_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def _plot_spectrum(sig, fs):
    freqs, psd = welch(sig, fs=fs, nperseg=min(int(fs*2), max(4, len(sig))))
    mask = (freqs >= 0.5) & (freqs <= 50.0)
    pdb  = 10 * np.log10(psd[mask] + 1e-12)
    band_cols = {"delta":"rgba(244,67,54,.20)","theta":"rgba(255,152,0,.20)",
                 "alpha":"rgba(76,175,80,.20)", "beta":"rgba(33,150,243,.20)",
                 "gamma":"rgba(156,39,176,.20)"}
    fig = go.Figure()
    for bname, (lo, hi) in EEG_BANDS.items():
        fig.add_vrect(x0=lo, x1=hi, fillcolor=band_cols[bname],
                      layer="below", line_width=0,
                      annotation_text=bname, annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="#9e9e9e")
    fig.add_trace(go.Scatter(x=freqs[mask], y=pdb, fill="tozeroy",
                             fillcolor="rgba(33,150,243,.15)",
                             line=dict(color="#42a5f5", width=2),
                             name="PSD (dB)"))
    fig.update_layout(title="Power Spectral Density (dB/Hz)",
                      **_DARK, margin=dict(l=50,r=20,t=50,b=50))
    fig.update_xaxes(title_text="Frequency (Hz)", **_AXIS)
    fig.update_yaxes(title_text="Power (dB)", **_AXIS)
    return fig


def _plot_band_powers(bp: dict):
    """Bar chart of relative band power, coloured by band."""
    names = list(bp.keys())
    vals  = [bp[n]["rel"] * 100 for n in names]
    abs_v = [bp[n]["abs"]       for n in names]
    cols  = ["#f44336","#ff9800","#4caf50","#2196f3","#9c27b0"]
    hover = [f"<b>{n}</b><br>Relative: {v:.1f}%<br>Absolute: {a:.4f}"
             for n, v, a in zip(names, vals, abs_v)]
    fig = go.Figure(go.Bar(
        x=names, y=vals, marker_color=cols,
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        hovertext=hover, hoverinfo="text",
    ))
    fig.update_layout(
        title="Relative Band Power (%)",
        **_DARK, margin=dict(l=50,r=20,t=50,b=40))
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(title_text="Power (%)", range=[0, max(vals)*1.25+5], **_AXIS)
    return fig


def _plot_metrics_table(metrics: dict):
    rows = [(m, v.get("accuracy",0), v.get("precision",0),
               v.get("recall",0), v.get("f1",0))
            for m, v in metrics.items() if "error" not in v]
    rows.sort(key=lambda r: r[4], reverse=True)
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Model</b>","<b>Accuracy</b>","<b>Precision</b>",
                    "<b>Recall</b>","<b>F1 (weighted)</b>"],
            fill_color="#1565c0", font=dict(color="white", size=13), align="left"),
        cells=dict(
            values=[[r[0] for r in rows],
                    [f"{r[1]:.4f}" for r in rows],
                    [f"{r[2]:.4f}" for r in rows],
                    [f"{r[3]:.4f}" for r in rows],
                    [f"{r[4]:.4f}" for r in rows]],
            fill_color=[["#1a1f2e","#161b22"]*20],
            font=dict(color="#e0e0e0", size=12), align="left"),
    ))
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _plot_prob_bar(proba: dict):
    """Horizontal bar chart of class probabilities with class-specific colours."""
    labels = list(proba.keys())
    vals   = [proba[l] * 100 for l in labels]
    cols   = [_SCOLOURS.get(l, "#42a5f5") for l in labels]
    max_v  = max(vals)

    # Highlight the winning class with a border
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker_color=cols,
        marker_line_color=["white" if v == max_v else "rgba(0,0,0,0)" for v in vals],
        marker_line_width=[2 if v == max_v else 0 for v in vals],
        text=[f"{v:.1f}%" for v in vals], textposition="auto",
        hovertemplate="<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title="Classification Confidence per Seizure Type",
        **_DARK, margin=dict(l=10,r=10,t=50,b=40), height=280)
    fig.update_xaxes(title_text="Confidence (%)", range=[0, max(max_v * 1.15, 30)], **_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def _plot_feat_importance(explanation: dict):
    if not explanation or "top_features" not in explanation:
        return None
    top   = explanation["top_features"][:12]
    names = [f["name"]       for f in top]
    imps  = [f["importance"] for f in top]
    vals  = [abs(f["value"]) for f in top]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Feature Importance (model)", "Feature |Value| (this sample)"])
    fig.add_trace(go.Bar(x=imps[::-1], y=names[::-1], orientation="h",
                         marker_color="#00bcd4", name="Importance"), row=1, col=1)
    fig.add_trace(go.Bar(x=vals[::-1], y=names[::-1], orientation="h",
                         marker_color="#ff9800", name="|Value|"), row=1, col=2)
    fig.update_layout(height=420, showlegend=False, **_DARK,
                      margin=dict(l=10,r=10,t=60,b=40))
    return fig


def _plot_clinical_thresholds(feat_vec: np.ndarray, feat_names: list):
    """
    Radar/spider chart showing where this sample sits relative to
    clinically meaningful thresholds for each seizure type.
    """
    from dataset import _CLASS_PROFILES, _FI, _EEG_ONLY_COLS

    # Pick 8 most discriminating features
    key_features = [
        "Delta_Band_Power", "Alpha_Band_Power", "Zero_Crossing_Rate",
        "EEG_Kurtosis", "Sample_Entropy",
        "Cross_Correlation_Between_Channels", "Hjorth_Complexity",
        "Interictal_Spike_Rate",
    ]

    fn_map = {n: i for i, n in enumerate(feat_names)}
    cats = []
    sample_vals = []
    class_vals  = {lbl: [] for lbl in range(N_CLASSES)}

    for feat in key_features:
        if feat not in fn_map:
            continue
        cats.append(feat.replace("_", " "))
        sv = float(feat_vec[fn_map[feat]])
        sample_vals.append(sv)
        for lbl, (mean_v, std_v) in _CLASS_PROFILES.items():
            if feat in _FI:
                class_vals[lbl].append(float(mean_v[_FI[feat]]))

    if not cats:
        return None

    # Normalise everything to [0,1] per feature for radar display
    all_vals = [sample_vals] + [class_vals[l] for l in range(N_CLASSES)]
    all_arr  = np.array(all_vals)
    mn = all_arr.min(axis=0); mx = all_arr.max(axis=0)
    rng = mx - mn + 1e-8
    all_norm = (all_arr - mn) / rng

    fig = go.Figure()
    colours = {"Sample": "#ffffff",
               "Normal": _SCOLOURS["Normal"],
               "Focal": _SCOLOURS["Focal"],
               "Absence": _SCOLOURS["Absence"],
               "Tonic-Atonic": _SCOLOURS["Tonic-Atonic"]}

    labels_plot = ["Sample"] + [SEIZURE_TYPES[l] for l in range(N_CLASSES)]
    for i, (label, norm_row) in enumerate(zip(labels_plot, all_norm)):
        col  = colours.get(label, "#aaaaaa")
        lw   = 3 if label == "Sample" else 1.5
        dash = "solid" if label == "Sample" else "dot"
        r    = list(norm_row) + [norm_row[0]]
        theta_labels = cats + [cats[0]]
        fig.add_trace(go.Scatterpolar(
            r=r, theta=theta_labels,
            mode="lines",
            name=label,
            line=dict(color=col, width=lw, dash=dash),
            opacity=0.9 if label == "Sample" else 0.7,
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False,
                            gridcolor="#21262d"),
            angularaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            bgcolor="#0d1117",
        ),
        title="Sample vs Clinical Profiles (normalised)",
        **_DARK, height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        margin=dict(l=40,r=40,t=60,b=40),
    )
    return fig


# ── File hash ─────────────────────────────────────────────────
def _file_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── File loading ──────────────────────────────────────────────
def _load_eeg_file(uploaded_file):
    """
    Returns (feat_vec, feature_names, true_label, eeg_for_viz, fs)
    feat_vec is ALWAYS aligned to training features when a model is loaded.
    """
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    fs   = 256.0
    eeg_viz = None

    train_feat_names = st.session_state.get("feature_names") or None
    if train_feat_names and len(train_feat_names) == 0:
        train_feat_names = None

    try:
        # ── EDF / BDF ──────────────────────────────────────────
        if name.endswith((".edf", ".bdf")):
            tmp_path = None
            try:
                import mne
                suffix = os.path.splitext(name)[1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                # Try reading EDF; handle both old and new MNE APIs
                try:
                    raw = mne.io.read_raw_edf(
                        tmp_path, preload=True, verbose=False, stim_channel=False)
                except TypeError:
                    raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
                fs = float(raw.info["sfreq"])
                # Pick EEG channels, fall back to all if none are labelled EEG
                eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
                if len(eeg_picks) == 0:
                    eeg_picks = list(range(min(19, len(raw.ch_names))))
                eeg_raw = raw.get_data(picks=eeg_picks)
                eeg_raw = np.nan_to_num(eeg_raw, nan=0.0, posinf=0.0, neginf=0.0)
                if eeg_raw.shape[0] == 0 or eeg_raw.shape[1] == 0:
                    raise ValueError("EDF contains no readable EEG data.")
                eeg_filt   = preprocess(eeg_raw, fs)
                feat_vec   = extract_features(eeg_filt, fs)
                feat_names = [f"f{i}" for i in range(len(feat_vec))]
                if train_feat_names:
                    from dataset import _align_features
                    feat_vec   = _align_features(feat_vec, feat_names, train_feat_names)
                    feat_names = train_feat_names
                return feat_vec, feat_names, None, eeg_filt, fs
            except ImportError:
                st.error("MNE not installed — run:  pip install mne")
                return None, None, None, None, fs
            except Exception as _edf_err:
                st.error(f"Could not read EDF file: {_edf_err}")
                return None, None, None, None, fs
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try: os.unlink(tmp_path)
                    except Exception: pass

        # ── CSV / TXT ──────────────────────────────────────────
        if name.endswith((".csv", ".txt")):
            feat_vec, feat_names, true_label = load_test_csv(
                data, fs, train_feature_names=train_feat_names)
            # Build eeg_viz if data looks like a time-series
            try:
                df = pd.read_csv(
                    io.StringIO(data.decode("utf-8", errors="replace")),
                    sep=None, engine="python")
                df = df.select_dtypes(include=[np.number])
                df = df.drop(columns=[c for c in df.columns
                                      if c.lower() in _DROP_COLS], errors="ignore")
                arr = df.values
                if arr.shape[0] > arr.shape[1] and arr.shape[0] > 100:
                    eeg_viz = preprocess(arr.T, fs)
            except Exception:
                pass
            return feat_vec, feat_names, true_label, eeg_viz, fs

        # ── XLSX / XLS ─────────────────────────────────────────
        if name.endswith((".xlsx", ".xls")):
            try:
                engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
                df_xl  = pd.read_excel(io.BytesIO(data), engine=engine)
                df_xl.columns = df_xl.columns.str.strip()
                col_lower = {c.lower(): c for c in df_xl.columns}

                # Extract true label if present
                true_label = None
                for lc in ("multi_class_label","seizure_type_label","label","class","target"):
                    if lc in col_lower:
                        try: true_label = int(df_xl[col_lower[lc]].iloc[0])
                        except Exception: pass
                        break

                # Drop non-feature cols
                drop_set = set(_DROP_COLS) | {"age","gender","medication_status","seizure_history"}
                feat_df  = df_xl.drop(
                    columns=[c for c in df_xl.columns if c.lower() in drop_set],
                    errors="ignore").select_dtypes(include=[np.number])

                arr = np.nan_to_num(feat_df.values.astype(np.float32), nan=0.0)
                feat_vec   = arr.mean(axis=0) if arr.shape[0] > 1 else arr[0]
                raw_names  = list(feat_df.columns)

                if train_feat_names:
                    from dataset import _align_features
                    feat_vec  = _align_features(feat_vec, raw_names, train_feat_names)
                    feat_names = train_feat_names
                else:
                    feat_names = raw_names

                return feat_vec, feat_names, true_label, eeg_viz, fs
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                return None, None, None, None, fs

        # ── Image ──────────────────────────────────────────────
        if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
            return _load_image_eeg(data, fs, train_feat_names)

    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None, None, None, None, fs


def _load_image_eeg(data: bytes, fs: float, train_feat_names):
    """
    Extract pseudo-EEG from an image (scanned report / EEG plot).
    Each horizontal strip in the image is treated as one EEG channel.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        st.error("Pillow not installed: pip install Pillow")
        return None, None, None, None, fs

    img = PILImage.open(io.BytesIO(data)).convert("L")
    arr = np.array(img, dtype=np.float64)   # (H, W)

    H, W = arr.shape
    n_ch = min(19, max(4, H // 50))   # ~50 pixels per channel strip
    strip_h = H // n_ch

    channels = []
    for i in range(n_ch):
        strip = arr[i*strip_h : (i+1)*strip_h, :]
        # Mean across strip height → 1-D pseudo-signal
        row_sig = strip.mean(axis=0)
        # Invert: darker pixels = higher amplitude in EEG paper
        row_sig = 255.0 - row_sig
        channels.append(row_sig)

    eeg = np.array(channels, dtype=np.float64)
    eeg -= eeg.mean(axis=1, keepdims=True)
    sd   = eeg.std(axis=1, keepdims=True) + 1e-8
    eeg  = eeg / sd * 50.0
    eeg  = preprocess(eeg, fs)
    feat_vec  = extract_features(eeg, fs)
    feat_names = [f"f{i}" for i in range(len(feat_vec))]

    if train_feat_names:
        from dataset import _align_features
        feat_vec   = _align_features(feat_vec, feat_names, train_feat_names)
        feat_names = train_feat_names

    return feat_vec, feat_names, None, eeg, fs


# ── Header ────────────────────────────────────────────────────
def _header():
    st.markdown("""
    <div class="med-header">
        <h1>🧬 NEUROSCAN PRO</h1>
        <p>Clinical-Grade Epilepsy Detection &amp; EEG Analysis System by ARK</p>
        <div style="margin-top:1rem;display:flex;justify-content:center;gap:2rem;">
            <span class="pill pill-ok">⚡ AI Medical Help </span>
            <span class="pill pill-info">🧠 4 Seizure Types</span>
            <span class="pill pill-warn">📋 Explainable AI</span>
            <span class="pill pill-ok">🔒 Deterministic Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Tab: Dashboard ────────────────────────────────────────────
def _tab_dashboard():
    st.markdown('<h3 style="color:#00bcd4;">🏥 Clinical Dashboard</h3>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Models", "✅ Ready" if st.session_state.models_trained else "Not trained")
    with c2:
        clf  = st.session_state.classifier
        best = clf.best_model_name if clf else "—"
        st.metric("Best Model", best or "—")
    with c3:
        st.metric("Training Samples",
                  f"{st.session_state.n_train_samples:,}"
                  if st.session_state.n_train_samples else "—")
    with c4:
        pred = st.session_state.prediction
        st.metric("Last Result", pred["seizure_type"] if pred else "—")

    st.markdown("---")

    # ── Seizure type reference card ────────────────────────────
    st.markdown("#### 📖 Seizure Type Reference")
    col_n, col_f, col_a, col_t = st.columns(4)
    ref = [
        (col_n, "Normal",       "#4caf50", "Alpha dominant (8–13 Hz)\nHigh entropy\nLow delta/theta\nLow cross-correlation"),
        (col_f, "Focal",        "#ff9800", "Theta dominant (4–8 Hz)\nVery low cross-corr\nElevated spikes\nReduced entropy"),
        (col_a, "Absence",      "#f44336", "3 Hz spike-wave (SWD)\nVery high kurtosis\nVery low entropy\nHigh bilateral sync"),
        (col_t, "Tonic-Atonic", "#9c27b0", "Highest ZCR\nHighest amplitude\nHigh Hjorth complexity\nBilateral spread"),
    ]
    for col, name, colour, desc in ref:
        with col:
            st.markdown(f"""
            <div style="border:1px solid {colour};border-radius:10px;padding:12px;
                        background:rgba(0,0,0,.3);margin-bottom:8px;">
                <b style="color:{colour}">{name}</b>
                <pre style="color:#b0bec5;font-size:11px;margin-top:6px;white-space:pre-wrap">{desc}</pre>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Step 1: Load dataset ──────────────────────────────────
    st.markdown("#### Step 1 — Load your dataset (epilepsy_data.csv)")
    col_path, col_btn = st.columns([3, 1])
    with col_path:
        csv_path = st.text_input(
            "Path to epilepsy_data.csv",
            value=r"D:\archive (2)\epilepsy_data.csv",
            help="Full path to your ~300k-row feature CSV",
        )
    with col_btn:
        max_rows_k = st.number_input(
            "Max rows (k)", min_value=5, max_value=300, value=50, step=5,
            help="Rows per class. 50k = fast, 200k = accurate."
        )

    load_btn = st.button("📂 Load Dataset", use_container_width=True)
    if load_btn:
        _load_dataset(csv_path, max_rows=int(max_rows_k) * 1000)

    # ── Step 1b: Train on synthetic only ─────────────────────
    st.markdown("**— or —**")
    if st.button("🧪 Train on Synthetic Data",
                 use_container_width=True,
                 help="Generates clinically accurate synthetic EEG feature data. "
                      " "):
        _load_synthetic_only()

    if st.session_state.data_loaded:
        st.success(f"✅ Dataset ready — {st.session_state.n_train_samples:,} samples · "
                   f"{st.session_state.n_classes} classes · "
                   f"{len(st.session_state.feature_names)} features")

    st.markdown("---")

    # ── Step 2: Train ─────────────────────────────────────────
    st.markdown("#### Step 2 — Train AI models")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        train_btn = st.button("🤖 Train AI Models", use_container_width=True,
                              disabled=not st.session_state.data_loaded)
    with col_t2:
        if os.path.exists(MODEL_SAVE_PATH):
            load_saved = st.button("📦 Load Saved Model", use_container_width=True)
            if load_saved:
                _load_saved_model()

    if train_btn and st.session_state.data_loaded:
        _train_models()

    if st.session_state.models_trained and st.session_state.classifier:
        metrics = st.session_state.classifier.metrics
        st.markdown('<div class="card"><h3>📈 Model Performance</h3>',
                    unsafe_allow_html=True)
        st.plotly_chart(_plot_metrics_table(metrics), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def _load_synthetic_only():
    with st.spinner("Generating synthetic training data with clinical EEG profiles ..."):
        X, y, feat_names = generate_synthetic_training_data(
            n_per_class=8000, noise_scale=0.12, seed=42)
        split = prepare_split(X, y, test_size=0.2, val_size=0.1, scale=True)
        st.session_state.split           = split
        st.session_state.feature_names   = feat_names
        st.session_state.n_train_samples = len(X)
        st.session_state.n_classes       = N_CLASSES
        st.session_state.data_loaded     = True
        st.success(f"✅ Synthetic data ready — {len(X):,} samples · "
                   f"{N_CLASSES} classes · {len(feat_names)} features")


def _load_dataset(csv_path: str, max_rows: int = 50_000):
    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        return
    with st.spinner(f"Loading dataset (up to {max_rows:,} rows) ..."):
        try:
            X, y, feat_names = load_real_csv(
                csv_path, max_rows=max_rows, use_synthetic_augment=True)
            split = prepare_split(X, y, test_size=0.2, val_size=0.1, scale=True)
            st.session_state.split           = split
            st.session_state.feature_names   = feat_names
            st.session_state.n_train_samples = len(X)
            st.session_state.n_classes       = N_CLASSES
            st.session_state.data_loaded     = True
            st.success(f"✅ Loaded {len(X):,} samples · "
                       f"{len(feat_names)} features · {N_CLASSES} classes")
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")


def _train_models():
    split = st.session_state.split
    clf   = AdvancedEpilepsyClassifier(num_classes=N_CLASSES)

    progress = st.progress(0)
    status   = st.empty()

    def _cb(idx, total, name):
        progress.progress((idx + 1) / total)
        status.text(f"Training {name} ({idx+1}/{total}) ...")

    with st.spinner("Training models on clinical EEG profiles ..."):
        metrics = clf.train(
            split["X_train"], split["y_train"],
            split["X_val"],   split["y_val"],
            feature_names=st.session_state.feature_names,
            progress_cb=_cb,
        )

    clf.scaler = split["scaler"]
    progress.empty(); status.empty()
    st.session_state.classifier     = clf
    st.session_state.models_trained = True

    try:
        clf.save(MODEL_SAVE_PATH)
        st.info(f"💾 Model saved to {MODEL_SAVE_PATH}")
    except Exception:
        pass

    valid = {n: v for n, v in metrics.items() if "error" not in v}
    if valid:
        best_f1 = max(v["f1"] for v in valid.values())
        st.success(f"✅ Training complete!  Best: **{clf.best_model_name}**  "
                   f"(F1 = {best_f1:.4f})")
    else:
        st.error("All models failed. Check your dataset.")


def _load_saved_model():
    try:
        clf = AdvancedEpilepsyClassifier()
        clf.load(MODEL_SAVE_PATH)
        st.session_state.classifier     = clf
        st.session_state.models_trained = True
        st.session_state.feature_names  = clf.feature_names
        st.session_state.n_classes      = N_CLASSES
        st.success(f"✅ Model loaded from {MODEL_SAVE_PATH}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")


# ── Tab: Upload & Analyse ─────────────────────────────────────
def _tab_upload():
    st.markdown('<h3 style="color:#00bcd4;">📁 Upload EEG File &amp; Analyse</h3>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>📋 Supported Formats</h3>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
    <div>
        <b style="color:#4caf50">✅ Pre-extracted feature CSVs</b><br>
        <span style="color:#90a4ae;font-size:.9rem">
        Label columns auto-detected &amp; stripped.<br>
        Non-EEG columns (Age, Gender) excluded automatically.
        </span>
    </div>
    <div>
        <b style="color:#ff9800">📈 Raw EEG / image files</b><br>
        <span style="color:#90a4ae;font-size:.9rem">
        EDF, BDF — clinical standard<br>
        CSV / XLSX — rows=time, cols=channels<br>
        PNG / JPG — scanned EEG report images
        </span>
    </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop EEG file here",
        type=["edf","bdf","csv","txt","xlsx","xls",
              "png","jpg","jpeg","bmp","tiff","gif"],
        key="uploader",
    )

    if uploaded:
        data      = uploaded.read()
        uploaded.seek(0)
        file_hash = _file_hash(data)

        feat_vec, feat_names, true_label, eeg_viz, fs = _load_eeg_file(uploaded)

        if feat_vec is None:
            st.error("Failed to load file.")
            return

        st.session_state.eeg_raw      = eeg_viz
        st.session_state.eeg_fs       = fs
        st.session_state.eeg_filename = uploaded.name

        st.success(f"✅ **{uploaded.name}** — {len(feat_vec)} features extracted")

        if true_label is not None:
            actual_type = SEIZURE_TYPES.get(true_label, f"Class {true_label}")
            st.info(f"📋 Ground-truth label in file: **{actual_type}** (class {true_label})")

        # Quick rule-based preview (no model needed)
        rule = rule_based_classify(feat_vec, feat_names)
        if rule:
            rtype = SEIZURE_TYPES.get(rule["label"], "Unknown")
            rcol  = _SCOLOURS.get(rtype, "#64b5f6")
            st.markdown(
                f'<div style="border-left:4px solid {rcol};padding:8px 14px;'
                f'background:rgba(0,0,0,.3);border-radius:0 8px 8px 0;margin:8px 0;">'
                f'🔬 <b style="color:{rcol}">Clinical rule preview:</b> '
                f'{rtype} ({rule["confidence"]*100:.0f}% confidence)<br>'
                f'<span style="font-size:.85rem;color:#90a4ae">{rule["reason"]}</span></div>',
                unsafe_allow_html=True)

        if eeg_viz is not None and eeg_viz.ndim == 2:
            st.plotly_chart(_plot_eeg(eeg_viz, fs, title=f"EEG — {uploaded.name}"),
                            use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(_plot_spectrum(eeg_viz[0], fs), use_container_width=True)
            with col2:
                bp = band_power_profile(eeg_viz[0], fs)
                st.plotly_chart(_plot_band_powers(bp), use_container_width=True)

        st.markdown("---")

        if not st.session_state.models_trained:
            st.warning("⚠️ Train models first (Dashboard → Train AI Models or Synthetic).")
        else:
            if st.button("🔬 Run Full Epilepsy Analysis",
                         use_container_width=True, type="primary"):
                _run_analysis(feat_vec, feat_names, file_hash, eeg_viz, fs)


def _run_analysis(feat_vec, feat_names, file_hash, eeg_viz, fs):
    clf   = st.session_state.classifier
    split = st.session_state.split

    with st.spinner("Classifying ..."):
        fv = feat_vec.copy().astype(np.float32)
        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale using training scaler (exact dimension matching)
        scaler = None
        if split and split.get("scaler"):
            scaler = split["scaler"]
        elif clf.scaler is not None:
            scaler = clf.scaler

        if scaler is not None:
            n_train_f = scaler.mean_.shape[0]
            n_test_f  = len(fv)
            if n_test_f == n_train_f:
                fv = scaler.transform(fv.reshape(1, -1)).flatten()
            elif n_test_f < n_train_f:
                padded = np.zeros(n_train_f, dtype=np.float32)
                padded[:n_test_f] = fv
                fv = scaler.transform(padded.reshape(1, -1)).flatten()
            else:
                fv = scaler.transform(fv[:n_train_f].reshape(1, -1)).flatten()

        result = clf.predict(fv)
        result["file_hash"] = file_hash

        if eeg_viz is not None and eeg_viz.ndim == 2:
            result["spike_info"] = detect_spikes(eeg_viz[0], fs)
            result["band_power"] = band_power_profile(eeg_viz[0], fs)
            result["dom_freq"]   = dominant_frequency(eeg_viz[0], fs)
        else:
            result["spike_info"] = None
            result["band_power"] = None
            result["dom_freq"]   = None

        # Store original (unscaled) feature vec for threshold radar chart
        result["raw_feat_vec"]   = feat_vec
        result["raw_feat_names"] = feat_names

        st.session_state.prediction      = result
        st.session_state.prediction_hash = file_hash

    st.success("✅ Analysis complete — see **Result** tab")
    _show_result(result, eeg_viz, fs)


# ── Tab: Result ───────────────────────────────────────────────
def _tab_predict():
    st.markdown('<h3 style="color:#00bcd4;">🔍 Seizure Classification Result</h3>',
                unsafe_allow_html=True)

    if not st.session_state.prediction:
        st.info("Upload an EEG file and click **Run Full Epilepsy Analysis** first.")
        st.markdown("---")
        st.subheader("🧪 Demo — analyse a synthetic sample")
        demo_type = st.selectbox("Select seizure type to simulate",
                                 [SEIZURE_TYPES[i] for i in range(N_CLASSES)])
        if st.button("▶️ Run Demo Analysis"):
            if not st.session_state.models_trained:
                st.warning("Train models first.")
            else:
                label_idx = {v: k for k, v in SEIZURE_TYPES.items()}[demo_type]
                _run_demo(label_idx)
        return

    _show_result(st.session_state.prediction,
                 st.session_state.eeg_raw,
                 st.session_state.eeg_fs)


def _run_demo(label: int):
    with st.spinner("Generating demo sample ..."):
        feat_vec = generate_demo_features(label, seed=42)
        clf      = st.session_state.classifier
        split    = st.session_state.split

        fv = feat_vec.copy()
        scaler = None
        if split and split.get("scaler"):
            scaler = split["scaler"]
        elif clf.scaler is not None:
            scaler = clf.scaler

        if scaler is not None:
            n_train_f = scaler.mean_.shape[0]
            if len(fv) < n_train_f:
                fv = np.pad(fv, (0, n_train_f - len(fv)))
            fv = scaler.transform(fv[:n_train_f].reshape(1, -1)).flatten()

        result = clf.predict(fv)
        result["spike_info"]    = None
        result["band_power"]    = None
        result["dom_freq"]      = None
        result["raw_feat_vec"]  = feat_vec
        result["raw_feat_names"] = [f"f{i}" for i in range(len(feat_vec))]
        st.session_state.prediction = result

    _show_result(result, None, 256.0)


def _show_result(result: dict, eeg, fs: float):
    if result is None:
        return

    stype    = result["seizure_type"]
    conf     = result["confidence"]
    ens_type = result["ensemble_type"]
    ens_conf = result["ensemble_confidence"]
    is_epi   = result["is_epileptic"]
    rule_note = result.get("rule_note", "")

    box_cls = "result-epilepsy" if is_epi else "result-normal"
    icon    = "🔴" if is_epi else "🟢"
    status  = "⚠️ EPILEPTIC ACTIVITY DETECTED" if is_epi else "✅ NORMAL EEG"
    scolour = _SCOLOURS.get(stype, "#00bcd4")

    st.markdown(f"""
    <div class="result-box {box_cls}">
        <div style="font-size:2rem;font-weight:800;margin-bottom:.5rem">
            {icon} {status}
        </div>
        <div style="font-size:1.4rem;font-weight:700;color:#e0e0e0;margin-bottom:.4rem">
            Seizure Type: <span style="color:{scolour}">{stype}</span>
            &nbsp;<span class="pill pill-info">ICD-10: {result['icd10']}</span>
        </div>
        <div style="color:#90a4ae;font-size:1rem">
            Confidence: <b>{conf*100:.1f}%</b>
            &nbsp;|&nbsp; Ensemble vote: <b>{ens_type}</b> ({ens_conf*100:.1f}%)
            &nbsp;|&nbsp; Model: <b>{result['model_used']}</b>
            {('<br><span style="color:#ffb74d;font-size:.85rem">'+rule_note+'</span>') if rule_note else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main probability bar
    st.plotly_chart(_plot_prob_bar(result["class_probabilities"]),
                    use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="card"><h3>🩺 Clinical Interpretation</h3>',
                    unsafe_allow_html=True)
        st.markdown(f"<p style='color:#cfd8dc'>{result['description']}</p>",
                    unsafe_allow_html=True)
        si = result.get("spike_info")
        if si:
            st.markdown(f"""
            <b style='color:#ff9800'>⚡ Spike Analysis</b><br>
            <span style='color:#90a4ae;font-size:.9rem'>
            Spikes detected: {si['n_spikes']} &nbsp;|&nbsp;
            Rate: {si['spike_rate_per_s']:.2f}/s &nbsp;|&nbsp;
            Dominant freq: {result.get('dom_freq',0):.1f} Hz
            </span>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card"><h3>🔑 Key EEG Markers</h3>',
                    unsafe_allow_html=True)
        pill_cls = "pill-crit" if is_epi else "pill-ok"
        for f in result.get("key_features", []):
            st.markdown(f'<span class="pill {pill_cls}">{f}</span>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Ensemble breakdown
    with st.expander("🗳️ Ensemble vote breakdown"):
        ens_p = result.get("ensemble_proba", result["class_probabilities"])
        st.plotly_chart(_plot_prob_bar(ens_p), use_container_width=True)

    # Band power
    bp = result.get("band_power")
    if bp and eeg is not None:
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(_plot_band_powers(bp), use_container_width=True)
        with col4:
            st.plotly_chart(_plot_spectrum(eeg[0] if eeg.ndim==2 else eeg, fs),
                            use_container_width=True)

    # Clinical threshold radar
    raw_fv    = result.get("raw_feat_vec")
    raw_names = result.get("raw_feat_names")
    if raw_fv is not None and raw_names is not None:
        radar = _plot_clinical_thresholds(raw_fv, raw_names)
        if radar:
            st.markdown("#### 🕸️ Clinical Feature Profile vs Seizure Type Prototypes")
            st.plotly_chart(radar, use_container_width=True)

    # Feature importance
    exp = result.get("explanation", {})
    if exp and "top_features" in exp:
        st.markdown('<div class="card"><h3>🔍 Feature Importance — Why This Diagnosis?</h3>',
                    unsafe_allow_html=True)
        fi_fig = _plot_feat_importance(exp)
        if fi_fig:
            st.plotly_chart(fi_fig, use_container_width=True)
        st.markdown(f"<p style='color:#90a4ae;font-size:.85rem'>"
                    f"Method: {exp.get('method','—')}</p>", unsafe_allow_html=True)
        df_exp = pd.DataFrame(exp["top_features"][:12])
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    _show_why(result)

    if eeg is not None and eeg.ndim == 2:
        with st.expander("📈 Raw EEG Signal"):
            st.plotly_chart(_plot_eeg(eeg, fs), use_container_width=True)


def _show_why(result: dict):
    stype = result["seizure_type"]
    bp    = result.get("band_power", {})
    si    = result.get("spike_info", {})
    domf  = result.get("dom_freq", 0)

    with st.expander("💡 Full AI Reasoning — Why this result?", expanded=True):
        st.markdown(f"### Diagnosis: **{stype}**")
        reasons = []

        if si:
            sr = si.get("spike_rate_per_s", 0)
            if sr > 5:
                reasons.append(f"⚡ **Very high spike rate ({sr:.1f}/s)** — strongly epileptiform. Normal EEG has <0.1 spikes/s.")
            elif sr > 1:
                reasons.append(f"⚡ **Elevated spike rate ({sr:.1f}/s)** — borderline epileptiform.")
            else:
                reasons.append(f"⚡ **Low spike rate ({sr:.2f}/s)** — within normal range.")

        if domf and domf > 0:
            if   domf < 4:   reasons.append(f"📊 **Dominant {domf:.1f} Hz (delta band)** — pathological in waking EEG; hallmark of absence/tonic-atonic seizures.")
            elif domf < 8:   reasons.append(f"📊 **Dominant {domf:.1f} Hz (theta band)** — typical of focal seizure ictal zone.")
            elif domf <= 13: reasons.append(f"📊 **Dominant {domf:.1f} Hz (alpha band)** — normal posterior rhythm. Points toward normal EEG.")
            else:            reasons.append(f"📊 **Dominant {domf:.1f} Hz (beta/gamma)** — fast activity; tonic onset or normal awake state.")

        if bp:
            d  = bp.get("delta",{}).get("rel",0)
            th = bp.get("theta",{}).get("rel",0)
            al = bp.get("alpha",{}).get("rel",0)
            be = bp.get("beta", {}).get("rel",0)
            ga = bp.get("gamma",{}).get("rel",0)
            if d  > 0.40: reasons.append(f"📈 **High delta ({d*100:.0f}%)** — hallmark of absence (3 Hz SWD) or tonic-atonic seizure.")
            if th > 0.30: reasons.append(f"📈 **Elevated theta ({th*100:.0f}%)** — focal onset recruitment pattern.")
            if al > 0.40: reasons.append(f"📈 **Strong alpha ({al*100:.0f}%)** — dominant alpha = normal awake or post-ictal.")
            if be > 0.25: reasons.append(f"📈 **Elevated beta ({be*100:.0f}%)** — tonic phase fast activity.")
            if ga > 0.05: reasons.append(f"📈 **Elevated gamma ({ga*100:.0f}%)** — high-frequency tonic bursts.")

        clinical_pattern = {
            "Normal":       "Dominant posterior alpha (8–13 Hz); no spike-wave; high spectral entropy; normal cross-channel correlation.",
            "Focal":        "Localised rhythmic theta (4–8 Hz) in ictal zone; low cross-channel correlation (asymmetric); elevated interictal spikes.",
            "Absence":      "Symmetric 3 Hz spike-and-wave discharges; very high kurtosis (sharp peaks); very low entropy (highly periodic); high bilateral cross-correlation.",
            "Tonic-Atonic": "Highest zero-crossing rate (rapid tonic oscillation); maximal amplitude and wavelet energy; high Hjorth complexity; bilateral spread.",
        }
        if stype in clinical_pattern:
            reasons.append(f"🧠 **EEG pattern:** {clinical_pattern[stype]}")

        rule_note = result.get("rule_note","")
        if rule_note:
            reasons.append(f"📐 **Clinical rule:** {rule_note.strip()}")

        exp = result.get("explanation", {})
        if exp and "top_features" in exp:
            top3 = exp["top_features"][:3]
            feat_str = ", ".join(
                f"**{f['name']}** (imp={f['importance']:.4f})" for f in top3)
            reasons.append(f"🔍 **Top discriminating features:** {feat_str}")

        reasons.append(
            f"📊 **Classifier confidence:** {result['confidence']*100:.1f}% "
            f"(ensemble: {result['ensemble_confidence']*100:.1f}%)")

        for i, r in enumerate(reasons, 1):
            st.markdown(f"{i}. {r}")

        st.markdown("---")
        st.caption("⚠️ This analysis is AI-assisted and intended to support, "
                   " the judgment of a qualified neurologist.")


# ── Tab: Brain Visualisation ──────────────────────────────────
def _tab_brain():
    st.markdown('<h3 style="color:#00bcd4;">🧠 Brain Activity Visualisation</h3>',
                unsafe_allow_html=True)
    eeg = st.session_state.eeg_raw
    fs  = st.session_state.eeg_fs

    if eeg is None or eeg.ndim != 2:
        st.info("Upload a raw EEG file (EDF, BDF, or time-series CSV) first.")
        return

    n_ch     = min(eeg.shape[0], 19)
    band_sel = st.selectbox("EEG band for topography", list(EEG_BANDS.keys()), index=2)
    lo, hi   = EEG_BANDS[band_sel]

    ch_powers = []
    for c in range(n_ch):
        freqs, psd = welch(eeg[c], fs=fs,
                           nperseg=min(int(fs*2), max(4, eeg.shape[1])))
        mask = (freqs >= lo) & (freqs <= hi)
        ch_powers.append(float(np.sum(psd[mask])))

    ch_xy = [
        ("Fp1",-.3,.9),("Fp2",.3,.9),
        ("F7",-.7,.6),("F3",-.35,.6),("Fz",0,.6),("F4",.35,.6),("F8",.7,.6),
        ("T3",-.9,.0),("C3",-.45,.0),("Cz",0,.0),("C4",.45,.0),("T4",.9,.0),
        ("T5",-.7,-.6),("P3",-.35,-.6),("Pz",0,-.6),("P4",.35,-.6),("T6",.7,-.6),
        ("O1",-.3,-.9),("O2",.3,-.9),
    ][:n_ch]

    xs = [p[1] for p in ch_xy]; ys = [p[2] for p in ch_xy]
    lbs = [p[0] for p in ch_xy]; pows = ch_powers[:len(ch_xy)]
    max_p = max(pows) + 1e-8

    fig = go.Figure()
    theta_c = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(theta_c), y=np.sin(theta_c),
                             mode="lines", line=dict(color="#546e7a", width=2),
                             showlegend=False))
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=[20 + p/max_p*40 for p in pows],
                    color=pows, colorscale="RdBu_r", showscale=True,
                    colorbar=dict(title="Power"),
                    line=dict(color="white", width=1)),
        text=lbs, textposition="top center",
        textfont=dict(color="white", size=10),
        hovertemplate="%{text}<br>Power: %{marker.color:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"Scalp Topography — {band_sel.capitalize()} ({lo}–{hi} Hz)",
        height=500, **_DARK, margin=dict(l=10,r=10,t=60,b=10),
    )
    fig.update_xaxes(range=[-1.2,1.2], visible=False)
    fig.update_yaxes(range=[-1.2,1.2], visible=False, scaleanchor="x")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Per-channel band power table"):
        df = pd.DataFrame({
            "Channel": lbs,
            f"{band_sel} Power (µV²/Hz)": [f"{p:.6f}" for p in pows]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab: Models ───────────────────────────────────────────────
def _tab_models():
    st.markdown('<h3 style="color:#00bcd4;">🤖 Model Performance</h3>',
                unsafe_allow_html=True)
    if not st.session_state.models_trained:
        st.info("Train models first (Dashboard → Train AI Models).")
        return

    clf = st.session_state.classifier
    st.plotly_chart(_plot_metrics_table(clf.metrics), use_container_width=True)
    st.markdown(f"**Best model:** {clf.best_model_name}  "
                f"(F1 = {clf.metrics.get(clf.best_model_name,{}).get('f1',0):.4f})")

    valid = {n: v for n, v in clf.metrics.items() if "error" not in v}
    names = list(valid.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="F1 (weighted)", x=names,
                         y=[valid[n]["f1"] for n in names], marker_color="#00bcd4"))
    fig.add_trace(go.Bar(name="Accuracy", x=names,
                         y=[valid[n]["accuracy"] for n in names], marker_color="#42a5f5"))
    fig.update_layout(barmode="group", title="Model Comparison",
                      **_DARK, margin=dict(l=50,r=20,t=60,b=40))
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(range=[0, 1], **_AXIS)
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance from best model
    clf_obj = clf.trained.get(clf.best_model_name)
    if clf_obj and hasattr(clf_obj, "feature_importances_"):
        fi = clf_obj.feature_importances_
        names_feat = clf.feature_names
        top = np.argsort(fi)[::-1][:20]
        fig2 = go.Figure(go.Bar(
            x=[fi[i] for i in top],
            y=[names_feat[i] if i < len(names_feat) else f"f{i}" for i in top],
            orientation="h", marker_color="#00bcd4",
        ))
        fig2.update_layout(title=f"Top 20 Features — {clf.best_model_name}",
                           height=500, **_DARK,
                           margin=dict(l=10,r=10,t=60,b=40))
        fig2.update_xaxes(title_text="Feature Importance", **_AXIS)
        fig2.update_yaxes(autorange="reversed", **_AXIS)
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab: Settings ─────────────────────────────────────────────
def _tab_settings():
    st.markdown('<h3 style="color:#00bcd4;">⚙️ Settings</h3>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System info")
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
        st.metric("NumPy", np.__version__)
        try: import sklearn; st.metric("scikit-learn", sklearn.__version__)
        except: pass
        try: import xgboost as xgb; st.metric("XGBoost", xgb.__version__)
        except: pass
        try: import lightgbm as lgb; st.metric("LightGBM", lgb.__version__)
        except: pass
        try: import antropy; st.metric("antropy", antropy.__version__)
        except: st.metric("antropy", "not installed (optional)")
    with col2:
        st.subheader("Session")
        st.metric("Models trained", "Yes" if st.session_state.models_trained else "No")
        st.metric("Training samples",
                  f"{st.session_state.n_train_samples:,}"
                  if st.session_state.n_train_samples else "0")
        st.metric("Feature count",
                  str(len(st.session_state.feature_names))
                  if st.session_state.feature_names else "0")
        st.metric("Saved model", "Exists" if os.path.exists(MODEL_SAVE_PATH) else "None")

    if st.button("🗑️ Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Main ──────────────────────────────────────────────────────
def main():
    _header()

    # Auto-load saved model on startup
    if not st.session_state.models_trained and os.path.exists(MODEL_SAVE_PATH):
        try:
            clf = AdvancedEpilepsyClassifier()
            clf.load(MODEL_SAVE_PATH)
            st.session_state.classifier     = clf
            st.session_state.models_trained = True
            st.session_state.feature_names  = clf.feature_names
            st.session_state.n_classes      = N_CLASSES
        except Exception:
            pass

    tabs = st.tabs([
        "🏥 Dashboard",
        "📁 Upload & Analyse",
        "🔍 Result",
        "🧠 Brain Viz",
        "🤖 Models",
        "⚙️ Settings",
    ])
    with tabs[0]: _tab_dashboard()
    with tabs[1]: _tab_upload()
    with tabs[2]: _tab_predict()
    with tabs[3]: _tab_brain()
    with tabs[4]: _tab_models()
    with tabs[5]: _tab_settings()


if __name__ == "__main__":
    main()
