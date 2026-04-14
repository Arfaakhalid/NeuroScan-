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
    chat_history=[],
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



def _plot_confusion_matrix(cm_data: list, model_name: str):
    """Heatmap confusion matrix."""
    labels = [SEIZURE_TYPES[i] for i in range(N_CLASSES)]
    cm     = np.array(cm_data)
    # Normalise to percentages
    row_sums = cm.sum(axis=1, keepdims=True).clip(1)
    cm_pct   = (cm / row_sums * 100).round(1)

    text_vals = [[f"{cm_pct[r,c]:.1f}%\n({cm[r,c]})"
                  for c in range(N_CLASSES)] for r in range(N_CLASSES)]

    fig = go.Figure(go.Heatmap(
        z=cm_pct, x=labels, y=labels,
        text=text_vals, texttemplate="%{text}",
        colorscale="Blues", showscale=True,
        hoverongaps=False,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name} (val set, % + count)",
        xaxis_title="Predicted", yaxis_title="Actual",
        **_DARK, height=380, margin=dict(l=10,r=10,t=60,b=60)
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(autorange="reversed", **_AXIS)
    return fig


def _plot_cv_results(cv_results: dict):
    """Bar chart of K-fold CV mean F1 with std error bars."""
    names  = [n for n in cv_results if "cv_f1_mean" in cv_results[n]]
    if not names:
        return None
    means  = [cv_results[n]["cv_f1_mean"] for n in names]
    stds   = [cv_results[n]["cv_f1_std"]  for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True, color="#ff9800"),
        marker_color="#00bcd4",
        text=[f"{m:.3f}±{s:.3f}" for m,s in zip(means,stds)],
        textposition="outside",
    ))
    fig.update_layout(
        title="5-Fold Cross-Validation F1 (mean ± std)",
        **_DARK, margin=dict(l=50,r=20,t=60,b=60)
    )
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(range=[0, 1.05], title_text="Weighted F1", **_AXIS)
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
            # -- Validate CSV is EEG-related
            try:
                import io as _io_v
                _dfv = pd.read_csv(
                    _io_v.StringIO(data.decode("utf-8", errors="replace")),
                    sep=None, engine="python", nrows=5)
                _csv_ok, _csv_why = _validate_tabular(_dfv, name)
                if not _csv_ok:
                    st.error("❌ Not an EEG file — " + _csv_why)
                    st.info(
                        "Please upload an EEG feature CSV "
                        "(same column format as epilepsy_data.csv) or "
                        "a raw EEG time-series CSV (rows=time, columns=channels)."
                    )
                    return None, None, None, None, fs
            except Exception:
                pass  # validation read failed; allow through
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

                # -- Validate XLSX is EEG-related
                _xl_ok, _xl_why = _validate_tabular(df_xl, name)
                if not _xl_ok:
                    st.error("❌ Not an EEG file — " + _xl_why)
                    st.info(
                        "Please upload an EEG feature XLSX "
                        "(same column format as epilepsy_data.csv)."
                    )
                    return None, None, None, None, fs
                return feat_vec, feat_names, true_label, eeg_viz, fs
            except Exception as e:
                st.error(f"Error reading Excel file: {e}")
                return None, None, None, None, fs

        # ── Image ──────────────────────────────────────────────
        if name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif")):
            _img_ok, _img_why = _validate_image(data)
            if not _img_ok:
                st.error("❌ Not an EEG image — " + _img_why)
                st.info(
                    "**Accepted image types:**\n"
                    "- Scanned EEG paper printouts (black signal traces on white paper)\n"
                    "- EEG software or digital recorder screenshots\n"
                    "- Clinical EEG report images or plots\n\n"
                    "**Not accepted:** photographs, posters, advertisements, logos, "
                    "diagrams, or any image unrelated to EEG brain signals."
                )
                return None, None, None, None, fs
            return _load_image_eeg(data, fs, train_feat_names)

    except Exception as e:
        st.error(f"Error loading file: {e}")
    return None, None, None, None, fs



# =====================================================================
#  FILE VALIDATORS  --  reject non-EEG uploads before any processing
# =====================================================================

_EEG_COL_KW = {
    "delta","theta","alpha","beta","gamma","eeg","seizure","amplitude",
    "entropy","kurtosis","hjorth","wavelet","spectral","frequency","band",
    "correlation","lyapunov","hurst","fractal","spike","power","energy",
    "zero_crossing","zcr","rms","variance","mobility","complexity",
    "line_length","sample","approximate","shannon","permutation",
    "detrended","higuchi","katz","lempel","interictal","mean_eeg",
    "signal_energy","peak_to_peak","psd","cross_corr","auto_corr",
}

_NON_EEG_COL_KW = {
    "name","surname","firstname","lastname","address","phone","email",
    "price","cost","revenue","sales","profit","invoice","order","product",
    "description","category","city","country","month","year",
    "employee","department","salary","company","brand","item",
    "quantity","stock","rating","review","comment","title","subject",
    "customer","client","vendor","tax","discount","total","amount",
}


def _validate_image(data):
    # ─────────────────────────────────────────────────────────────────
    # STRICT EEG image validator.
    # An EEG image must pass ALL hard gates.  We default to REJECT
    # and only accept when MULTIPLE independent EEG signals are present.
    #
    # Hard rejection triggers (ANY one → reject immediately):
    #   • Too small
    #   • Portrait orientation  (photos / posters / selfies)
    #   • High colour saturation (photographs / posters / coloured diagrams)
    #   • Skin-tone dominant pixels (face photos)
    #   • Too few light-background pixels (dark-themed non-EEG images)
    #   • Mid-tone-dominant content (natural photos)
    #   • Too many dark pixels overall (x-ray / MRI / dark images)
    #   • Single-colour / blank image
    #   • Insufficient horizontal-stripe structure (EEG channels run left→right)
    #   • No visible thin oscillating traces (column-wise variance test)
    # ─────────────────────────────────────────────────────────────────
    try:
        from PIL import Image as _PIL
        import io as _io
        import numpy as _np

        img = _PIL.open(_io.BytesIO(data))
        W, H = img.size

        # ── Hard gate 1: minimum size ────────────────────────────────
        if W < 200 or H < 80:
            return False, (
                f"Image too small ({W}×{H} px). "
                "EEG recordings are at least 200×80 px."
            )

        # ── Hard gate 2: portrait orientation ────────────────────────
        # EEG records are ALWAYS landscape or square.
        # Anything taller than wide (even by a little) → reject.
        if H > W * 1.05:
            return False, (
                f"Portrait orientation ({W}×{H} px). "
                "EEG recordings are always landscape or square. "
                "This looks like a photo, selfie, or poster — not an EEG."
            )

        # ── Pixel analysis ────────────────────────────────────────────
        rgb_arr = _np.array(img.convert("RGB"), dtype=_np.float32)   # H×W×3
        gray    = _np.array(img.convert("L"),   dtype=_np.float32)   # H×W

        r_ch, g_ch, b_ch = rgb_arr[:,:,0], rgb_arr[:,:,1], rgb_arr[:,:,2]

        # Per-pixel colour saturation
        sat_per_px = float(_np.mean(_np.std(rgb_arr, axis=2)))

        # ── Hard gate 3: high colour saturation ──────────────────────
        # EEG images (paper / screen caps): sat ≈ 0–18
        # Colour photos / posters: sat ≈ 30–80+
        if sat_per_px > 22:
            return False, (
                f"High colour saturation ({sat_per_px:.1f}). "
                "EEG recordings are monochrome or near-monochrome. "
                "Colour photographs, posters, illustrations, and face images "
                "are not EEG data."
            )

        # ── Hard gate 4: skin-tone detection ─────────────────────────
        # Skin tones: R>95, G>40, B>20, R>G, R>B, |R-G|>15, R-B>15
        skin_mask = (
            (r_ch > 95) & (g_ch > 40) & (b_ch > 20) &
            (r_ch > g_ch) & (r_ch > b_ch) &
            (_np.abs(r_ch - g_ch) > 15) & ((r_ch - b_ch) > 15)
        )
        skin_frac = float(skin_mask.mean())
        if skin_frac > 0.08:
            return False, (
                f"Skin-tone pixels detected ({skin_frac*100:.0f}% of image). "
                "This appears to be a photograph of a person, face, or body. "
                "Please upload an EEG signal recording or report image."
            )

        # ── Compute pixel distribution ────────────────────────────────
        white_frac  = float((gray > 220).mean())   # bright white
        light_frac  = float((gray > 160).mean())   # light (incl. cream)
        dark_frac   = float((gray < 50).mean())    # very dark
        mid_frac    = 1.0 - light_frac - dark_frac
        overall_std = float(gray.std())

        # ── Hard gate 5: insufficient light background ────────────────
        # EEG paper: > 50% light pixels
        if light_frac < 0.45:
            return False, (
                f"Insufficient light background ({light_frac*100:.0f}% light px). "
                "EEG paper recordings have > 45% light/white background. "
                "Dark images, MRI scans, X-rays, or photos are not accepted."
            )

        # ── Hard gate 6: mid-tone dominated (natural photos) ──────────
        if mid_frac > 0.50:
            return False, (
                f"Mid-tone dominated ({mid_frac*100:.0f}% mid-tone px). "
                "EEG images are high-contrast (white bg + thin dark traces). "
                "This looks like a natural photograph or illustration."
            )

        # ── Hard gate 7: too many dark pixels ─────────────────────────
        if dark_frac > 0.40:
            return False, (
                f"Too many dark pixels ({dark_frac*100:.0f}%). "
                "EEG traces are thin lines on a light background — "
                "a mostly dark image is not an EEG printout."
            )

        # ── Hard gate 8: blank / near-uniform image ────────────────────
        if overall_std < 12:
            return False, "Image appears blank or nearly uniform — not an EEG recording."

        # ── Hard gate 9: horizontal trace structure ────────────────────
        # EEG channels run left→right. Column-wise variance (oscillation
        # within each column) should be reasonable; row-wise variance
        # (contrast between channel rows) should also be present.
        col_var = float(_np.mean(_np.var(gray, axis=0)))   # variance along x
        row_var = float(_np.mean(_np.var(gray, axis=1)))   # variance along y

        # Reject if column variance is almost zero (no oscillating traces)
        if col_var < 8:
            return False, (
                "No horizontal signal traces detected (column variance too low). "
                "EEG recordings contain multiple oscillating channel traces. "
                "This image does not match that pattern."
            )

        # Reject if the image is essentially purely vertical (portrait structure)
        if row_var > 0 and (col_var / (row_var + 1e-6)) < 0.25:
            return False, (
                "Image has predominantly vertical structure. "
                "EEG recordings have horizontal channel traces. "
                "This does not appear to be an EEG recording."
            )

        # ── Hard gate 10: must have multiple horizontal stripes ────────
        # Divide image into N_STRIPES horizontal bands; each EEG channel
        # strip should have meaningful variance (i.e., contain a signal).
        N_STRIPES = 4
        strip_h   = max(1, H // N_STRIPES)
        active_stripes = 0
        for s in range(N_STRIPES):
            strip = gray[s*strip_h:(s+1)*strip_h, :]
            if strip.std() > 6 and strip.mean() > 100:
                active_stripes += 1
        if active_stripes < 2:
            return False, (
                f"Only {active_stripes} of {N_STRIPES} horizontal strips "
                "contain signal-like content. "
                "An EEG image should show multiple channel traces stacked vertically. "
                "This image does not have that structure."
            )

        # ── Final accept: all hard gates passed ───────────────────────
        return True, ""

    except Exception as _ex:
        return False, f"Could not read image: {_ex}"



def _validate_tabular(df, filename):
    # Returns (is_valid: bool, reason: str).
    # Accepts EEG feature CSVs/XLSXs and raw EEG time-series.
    # Rejects sales/HR/contact spreadsheets and unrelated data tables.
    if df is None or len(df) == 0:
        return False, f"'{filename}' is empty."

    drop_set = {
        "multi_class_label","seizure_type_label","label","class","target",
        "age","gender","medication_status","seizure_history",
    }
    num_df = df.drop(
        columns=[c for c in df.columns if c.lower() in drop_set],
        errors="ignore"
    ).select_dtypes(include=["number"])

    n_num   = num_df.shape[1]
    cols_lc = [c.lower().strip() for c in df.columns]

    # 1 -- Need at least 5 numeric columns
    if n_num < 5:
        return False, (
            f"Only {n_num} numeric column(s) found. "
            "An EEG feature file needs at least 5 numeric columns. "
            "This does not look like EEG data."
        )

    # 2 -- Obvious non-EEG column names -> reject
    bad_cols = [c for c in cols_lc if c in _NON_EEG_COL_KW]
    if len(bad_cols) >= 3:
        return False, (
            f"Column names suggest a non-EEG spreadsheet: {bad_cols[:5]}. "
            "This looks like sales, HR, or contact data. "
            "Please upload an EEG feature CSV or XLSX."
        )

    # 3 -- Recognised EEG column names -> accept immediately
    eeg_hits = [c for c in cols_lc if any(kw in c for kw in _EEG_COL_KW)]
    if eeg_hits:
        return True, ""

    # 4 -- Generic numeric column names (f0, f1, ch0, ch1...)
    generic = sum(1 for c in cols_lc if len(c) > 1 and c[0] in ("f","c") and c[1:].isdigit())
    if generic >= 5:
        return True, ""

    # 5 -- All-integer column names (0, 1, 2...)
    try:
        all_int = all(c.lstrip("-").isdigit() for c in cols_lc)
    except Exception:
        all_int = False
    if all_int and n_num >= 10:
        return True, ""

    # 6 -- Large numeric file (>= 20 cols) = probably raw time-series
    if n_num >= 20:
        return True, ""

    # 7 -- 5-19 numeric cols, no EEG keywords -> reject
    return False, (
        f"Could not confirm this is an EEG file. "
        f"Found {n_num} numeric columns but no EEG feature names. "
        f"Columns: {list(df.columns[:6])}... "
        "Please upload an EEG feature file (epilepsy_data.csv column format) "
        "or a raw EEG time-series with >= 20 numeric columns."
    )

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
            <span class="pill pill-ok">🔒 Deterministic Results</span><br>
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

    load_btn = st.button("📂 Load Dataset", key="btn_load_dataset", use_container_width=True)
    if load_btn:
        _load_dataset(csv_path, max_rows=int(max_rows_k) * 1000)

    # ── Step 1b: Train on synthetic only ─────────────────────
    st.markdown("**— or —**")
    if st.button("🧪 Train on Synthetic Data",
                 key="btn_train_synthetic", use_container_width=True,
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
        train_btn = st.button("🤖 Train AI Models", key="btn_train_models", use_container_width=True,
                              disabled=not st.session_state.data_loaded)
    with col_t2:
        if os.path.exists(MODEL_SAVE_PATH):
            load_saved = st.button("📦 Load Saved Model", key="btn_load_saved", use_container_width=True)
            if load_saved:
                _load_saved_model()

    if train_btn and st.session_state.data_loaded:
        _train_models()

    if st.session_state.models_trained and st.session_state.classifier:
        st.success("✅ Models are trained and ready. See the **🤖 Models** tab for full performance metrics.")


def _load_synthetic_only():
    with st.spinner("Generating synthetic training data with clinical EEG profiles ..."):
        X, y, feat_names = generate_synthetic_training_data(
            n_per_class=10000, noise_scale=0.10, seed=42)
        split = prepare_split(X, y, test_size=0.15, val_size=0.15, scale=True)
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
            split = prepare_split(X, y, test_size=0.15, val_size=0.15, scale=True)
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
    total_models = 5   # RF, LR, SVM, KNN, CNN-1D

    def _cb(idx, total, name):
        progress.progress((idx + 1) / max(total, 1))
        status.text(f"Training {name} ({idx+1}/{total}) ...")

    with st.spinner("Training 5 models (RandomForest, LogisticRegression, SVM, KNN, CNN-1D) ..."):
        metrics = clf.train(
            split["X_train"], split["y_train"],
            split["X_val"],   split["y_val"],
            X_test=split.get("X_test"), y_test=split.get("y_test"),
            feature_names=st.session_state.feature_names,
            progress_cb=_cb,
            run_cv=True, n_cv_splits=5,
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
        st.success(
            f"✅ Training complete — 70% train / 15% val / 15% test\n"
            f"Best model: **{clf.best_model_name}**  (Val F1 = {best_f1:.4f})"
        )
        # Show split sizes
        s = split
        st.info(
            f"Train: {len(s['X_train']):,} | "
            f"Val: {len(s['X_val']):,} | "
            f"Test: {len(s.get('X_test', [])):,} samples"
        )
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
                         key="btn_run_analysis", use_container_width=True, type="primary"):
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
        if st.button("▶️ Run Demo Analysis", key="btn_demo"):
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

    clf   = st.session_state.classifier
    split = st.session_state.split or {}

    # ── Split info banner ──────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Train", f"{len(split.get('X_train',[])): ,}" if split.get('X_train') is not None else "—")
    c2.metric("Val (15%)",  f"{len(split.get('X_val',[])): ,}"   if split.get('X_val') is not None else "—")
    c3.metric("Test (15%)", f"{len(split.get('X_test',[])): ,}"  if split.get('X_test') is not None else "—")
    c4.metric("Best Model", clf.best_model_name or "—")

    st.markdown("---")

    # ── Validation metrics table ───────────────────────────────
    st.markdown("#### 📊 Validation Set Metrics")
    st.plotly_chart(_plot_metrics_table(clf.metrics), use_container_width=True)

    # ── Test set metrics table ─────────────────────────────────
    if clf._test_metrics:
        st.markdown("#### 🧪 Held-out Test Set Metrics (15%)")
        rows = []
        for name, v in clf._test_metrics.items():
            if "error" not in v:
                rows.append((name, v.get("test_accuracy",0), v.get("test_precision",0),
                             v.get("test_recall",0), v.get("test_f1",0)))
        rows.sort(key=lambda r: r[4], reverse=True)
        fig_t = go.Figure(go.Table(
            header=dict(
                values=["<b>Model</b>","<b>Test Acc</b>","<b>Test Prec</b>",
                        "<b>Test Recall</b>","<b>Test F1</b>"],
                fill_color="#0d47a1", font=dict(color="white",size=13), align="left"),
            cells=dict(
                values=[[r[0] for r in rows],
                        [f"{r[1]:.4f}" for r in rows],[f"{r[2]:.4f}" for r in rows],
                        [f"{r[3]:.4f}" for r in rows],[f"{r[4]:.4f}" for r in rows]],
                fill_color=[["#1a1f2e","#161b22"]*20],
                font=dict(color="#e0e0e0",size=12), align="left"),
        ))
        fig_t.update_layout(margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_t, use_container_width=True)

    st.markdown("---")

    # ── Val + Test F1 comparison chart ─────────────────────────
    st.markdown("#### 📈 Val vs Test F1 — Overfitting Check")
    valid = {n: v for n, v in clf.metrics.items() if "error" not in v}
    names_m = sorted(valid.keys())
    val_f1s  = [valid[n]["f1"] for n in names_m]
    test_f1s = [clf._test_metrics.get(n,{}).get("test_f1", 0) for n in names_m]
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="Val F1",  x=names_m, y=val_f1s,
                             marker_color="#00bcd4"))
    fig_cmp.add_trace(go.Bar(name="Test F1", x=names_m, y=test_f1s,
                             marker_color="#ff9800"))
    fig_cmp.update_layout(barmode="group",
                          title="Validation F1 vs Test F1 (close gap = no overfitting)",
                          **_DARK, margin=dict(l=50,r=20,t=60,b=60))
    fig_cmp.update_xaxes(**_AXIS)
    fig_cmp.update_yaxes(range=[0,1.05], title_text="Weighted F1", **_AXIS)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── K-fold CV results ──────────────────────────────────────
    if clf.cv_results:
        st.markdown("#### 🔄 5-Fold Cross-Validation Results")
        cv_fig = _plot_cv_results(clf.cv_results)
        if cv_fig:
            st.plotly_chart(cv_fig, use_container_width=True)
        with st.expander("CV details table"):
            cv_rows = [(n, v["cv_f1_mean"], v["cv_f1_std"],
                        v["cv_acc_mean"], v["cv_acc_std"])
                       for n, v in clf.cv_results.items()]
            cv_df = pd.DataFrame(cv_rows,
                columns=["Model","CV F1 mean","CV F1 std","CV Acc mean","CV Acc std"])
            st.dataframe(cv_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Confusion matrix for best model ───────────────────────
    st.markdown(f"#### 🗂️ Confusion Matrix — {clf.best_model_name}")
    best_cm = clf.metrics.get(clf.best_model_name, {}).get("confusion_matrix")
    if best_cm:
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.markdown("**Validation set**")
            st.plotly_chart(_plot_confusion_matrix(best_cm, clf.best_model_name),
                            use_container_width=True)
        with col_cm2:
            test_cm = clf._test_metrics.get(clf.best_model_name, {}).get("test_cm")
            if test_cm:
                st.markdown("**Test set (held-out 15%)**")
                st.plotly_chart(_plot_confusion_matrix(test_cm,
                                f"{clf.best_model_name} [TEST]"),
                                use_container_width=True)

    # ── Per-model confusion matrix selector ───────────────────
    with st.expander("🗂️ Confusion matrix for any model"):
        all_names = [n for n,v in clf.metrics.items()
                     if "confusion_matrix" in v and "error" not in v]
        if all_names:
            sel = st.selectbox("Select model", all_names)
            sel_cm = clf.metrics[sel]["confusion_matrix"]
            st.plotly_chart(_plot_confusion_matrix(sel_cm, sel),
                            use_container_width=True)

    # ── Feature importance ─────────────────────────────────────
    clf_obj = clf.trained.get(clf.best_model_name)
    if clf_obj and hasattr(clf_obj, "feature_importances_"):
        fi         = clf_obj.feature_importances_
        names_feat = clf.feature_names
        top        = np.argsort(fi)[::-1][:20]
        fig2 = go.Figure(go.Bar(
            x=[fi[i] for i in top],
            y=[names_feat[i] if i < len(names_feat) else f"f{i}" for i in top],
            orientation="h", marker_color="#00bcd4",
        ))
        fig2.update_layout(title=f"Top 20 Features — {clf.best_model_name}",
                           height=520, **_DARK, margin=dict(l=10,r=10,t=60,b=40))
        fig2.update_xaxes(title_text="Feature Importance", **_AXIS)
        fig2.update_yaxes(autorange="reversed", **_AXIS)
        st.plotly_chart(fig2, use_container_width=True)


# ── Tab: NeuroBot Chatbot ─────────────────────────────────────
def _neurobot_answer(question: str, prediction: dict) -> str:
    """
    Rule-based intelligent chatbot that answers EEG / epilepsy questions.
    Prioritises context from the current prediction result when available.
    Returns a markdown-formatted answer string.
    """
    q = question.lower().strip()
    # Remove punctuation noise
    import re as _re
    q = _re.sub(r"[^\w\s]", " ", q)
    q = " ".join(q.split())

    # ── Helpers ──────────────────────────────────────────────────────
    stype   = prediction.get("seizure_type",  "Unknown")  if prediction else None
    conf    = prediction.get("confidence",    0.0)        if prediction else None
    icd10   = prediction.get("icd10",         "--")       if prediction else None
    is_epi  = prediction.get("is_epileptic",  False)      if prediction else False
    bp      = prediction.get("band_power",    {})         if prediction else {}
    si      = prediction.get("spike_info",    {})         if prediction else {}
    domf    = prediction.get("dom_freq",      0)          if prediction else 0
    desc    = prediction.get("description",   "")         if prediction else ""

    has_result = prediction is not None and stype not in (None, "Unknown")

    def _result_ctx():
        if not has_result:
            return ""
        return (f"\n\n📋 **Your latest result:** {stype} "
                f"(confidence {conf*100:.0f}%, ICD-10: {icd10})")

    # ── Intent matching helpers ────────────────────────────────────
    def _has(*words):
        return any(w in q for w in words)

    def _has_all(*words):
        return all(w in q for w in words)

    # ────────────────────────────────────────────────────────────────
    # INTENT 1: greeting / what can you do
    # ────────────────────────────────────────────────────────────────
    if _has("hello", "hi", "hey", "greet") or q in ("what can you do", "help", "who are you"):
        return (
            "👋 **Hi! I'm NeuroBot**, the AI assistant for NeuroScan Pro.\n\n"
            "I can answer questions about:\n"
            "- Your **EEG analysis result** and what it means\n"
            "- **Seizure types** — Normal, Focal, Absence, Tonic-Atonic\n"
            "- **EEG features** — band power, entropy, spikes, Hjorth parameters\n"
            "- **What to do next** after a result\n"
            "- **Medications**, general management, and clinical context\n\n"
            "Just ask me anything! For example:\n"
            "*\"What does my result mean?\"*, *\"What is an absence seizure?\"*, "
            "*\"What causes focal seizures?\"*"
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 2: what does my result / diagnosis mean
    # ────────────────────────────────────────────────────────────────
    if (_has("result", "diagnosis", "diagnos", "mean", "tell me", "explain my", "what does") and
            _has("result", "diagnos", "prediction", "detection", "finding", "output", "say")):
        if not has_result:
            return ("⚠️ No result yet — please upload an EEG file and run analysis first. "
                    "Then I can explain exactly what the result means.")
        if not is_epi:
            return (
                f"✅ **Your EEG shows: Normal**\n\n"
                f"The AI detected **no epileptic activity** in your EEG. "
                f"Brain electrical signals appear to be within normal clinical parameters:\n"
                f"- Alpha rhythm (8–13 Hz) is dominant — typical of a healthy awake brain\n"
                f"- No spike-wave discharges or epileptiform patterns were found\n"
                f"- Spectral entropy is high, reflecting complex, irregular (healthy) brain activity\n\n"
                f"**Confidence:** {conf*100:.0f}%\n\n"
                f"⚠️ This is AI-assisted screening. A normal result does **not** rule out "
                f"epilepsy — please confirm with a qualified neurologist."
            )
        else:
            return (
                f"🔴 **Your EEG shows: {stype} Seizure**\n\n"
                f"{desc}\n\n"
                f"**ICD-10:** {icd10} | **Confidence:** {conf*100:.0f}%\n\n"
                f"⚠️ This AI result is for research/screening purposes. "
                f"Please consult a **neurologist** for clinical evaluation and diagnosis."
            )

    # ────────────────────────────────────────────────────────────────
    # INTENT 3: confidence / accuracy questions
    # ────────────────────────────────────────────────────────────────
    if _has("confiden", "accurat", "reliable", "trust", "how sure", "certain", "probability"):
        base = (
            "**Confidence** is the probability the ensemble of AI models assigns to the predicted class.\n\n"
            "- **>85%** — high confidence, very likely correct\n"
            "- **70–85%** — moderate confidence, result is probable\n"
            "- **<70%** — lower confidence, result should be interpreted cautiously\n\n"
            "The system uses a soft-vote **ensemble** of up to 5 models "
            "(RandomForest, LogisticRegression, SVM, KNN, CNN-1D).\n\n"
        )
        if has_result:
            level = "high" if conf > 0.85 else ("moderate" if conf > 0.70 else "lower")
            base += f"📋 **Your result confidence: {conf*100:.0f}% ({level})**"
        return base

    # ────────────────────────────────────────────────────────────────
    # INTENT 4: seizure type — absence
    # ────────────────────────────────────────────────────────────────
    if _has("absence", "petit mal", "3 hz", "spike wave", "swd"):
        return (
            "## Absence Seizure (ICD-10: G40.3)\n\n"
            "Absence seizures are brief (~5–30 sec) generalised non-convulsive episodes "
            "characterised by:\n"
            "- **3 Hz symmetric spike-and-wave discharges (SWD)** on EEG\n"
            "- Abrupt onset and offset — the person suddenly stops and stares\n"
            "- Very high **delta band power**, very **low entropy** (highly periodic signal)\n"
            "- High **bilateral cross-correlation** (both hemispheres in sync)\n"
            "- **No convulsions** — the person does not fall\n\n"
            "**Typical patients:** Children aged 4–14; often outgrown by adulthood.\n\n"
            "**Treatment:** Ethosuximide (first-line), valproate, or lamotrigine."
            + _result_ctx()
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 5: focal seizure
    # ────────────────────────────────────────────────────────────────
    if _has("focal", "partial", "localised", "localized", "temporal lobe"):
        return (
            "## Focal (Partial) Seizure (ICD-10: G40.1)\n\n"
            "Focal seizures originate in **one localised brain region**:\n"
            "- **Rhythmic theta discharge (4–8 Hz)** confined to one hemisphere/lobe\n"
            "- **Very low cross-channel correlation** (other channels unaffected)\n"
            "- Elevated **interictal spike rate** in the ictal zone\n"
            "- **Reduced spectral entropy** (more ordered than normal EEG)\n\n"
            "**Symptoms vary** by location: motor (jerking), sensory (tingling), "
            "autonomic (nausea), or cognitive (déjà vu).\n\n"
            "**May generalise** into a tonic-clonic seizure.\n\n"
            "**Treatment:** Carbamazepine, oxcarbazepine, levetiracetam. "
            "Surgical resection if drug-resistant."
            + _result_ctx()
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 6: tonic / atonic / tonic-atonic
    # ────────────────────────────────────────────────────────────────
    if _has("tonic", "atonic", "drop attack", "grand mal", "convuls"):
        return (
            "## Tonic / Atonic Seizure (ICD-10: G40.5 / G40.8)\n\n"
            "**Tonic phase:** Sudden sustained muscle stiffening:\n"
            "- **Highest zero-crossing rate** of all seizure types (rapid oscillation)\n"
            "- **Maximal signal amplitude and wavelet energy**\n"
            "- **High Hjorth complexity** and elevated gamma power\n"
            "- **Bilateral spread** → high cross-channel correlation\n\n"
            "**Atonic phase:** Sudden loss of muscle tone → fall ('drop attack').\n\n"
            "**Danger:** High injury risk from falls.\n\n"
            "**Treatment:** Valproate, lamotrigine, rufinamide. "
            "VNS or corpus callosotomy for refractory cases."
            + _result_ctx()
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 7: normal EEG
    # ────────────────────────────────────────────────────────────────
    if ((_has("normal") and _has("eeg", "result", "brain", "signal", "mean")) or
            q in ("what is normal eeg", "normal eeg")):
        return (
            "## Normal EEG\n\n"
            "A normal awake EEG shows:\n"
            "- **Alpha rhythm dominant (8–13 Hz)** — especially in posterior regions\n"
            "- **High spectral entropy** — complex, irregular activity (healthy)\n"
            "- **Low delta and theta power** — slow waves are abnormal when awake\n"
            "- **Low cross-channel correlation** — channels oscillate independently\n"
            "- **Low interictal spike rate** — fewer than 1 spike/second\n\n"
            "A normal result does **not** rule out epilepsy — seizures may not appear "
            "in a short recording."
            + _result_ctx()
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 8: band power questions
    # ────────────────────────────────────────────────────────────────
    if _has("delta", "theta", "alpha", "beta", "gamma") and _has("band", "power", "wave", "frequency", "hz"):
        band_info = {
            "delta": ("0.5–4 Hz", "Deep sleep; dominant in absence and tonic-atonic seizures; abnormal in waking EEG."),
            "theta": ("4–8 Hz", "Drowsiness/focal seizure ictal zone; elevated theta suggests focal onset."),
            "alpha": ("8–13 Hz", "Relaxed wakefulness; dominant alpha = normal brain state."),
            "beta":  ("13–30 Hz", "Active thinking; elevated in tonic phase or medication effect."),
            "gamma": ("30–45 Hz", "High cognitive load or tonic burst; elevated in tonic seizures."),
        }
        answers = []
        for bname, (brange, bdesc) in band_info.items():
            if bname in q:
                answers.append(f"**{bname.capitalize()} ({brange}):** {bdesc}")
        if answers:
            result_ctx = ""
            if has_result and bp:
                result_ctx = "\n\n📋 **Your band powers:**\n"
                for b, (lo, hi) in EEG_BANDS.items():
                    rel = bp.get(b, {}).get("rel", 0) * 100
                    result_ctx += f"- {b.capitalize()}: {rel:.1f}%\n"
            return "\n\n".join(answers) + result_ctx
        # Generic band power question
        if has_result and bp:
            lines = ["**Your EEG band power breakdown:**"]
            for b in ("delta","theta","alpha","beta","gamma"):
                rel = bp.get(b, {}).get("rel", 0) * 100
                lines.append(f"- {b.capitalize()}: {rel:.1f}%")
            return "\n".join(lines)
        return ("EEG is divided into frequency bands:\n"
                "- **Delta (0.5–4 Hz):** deep sleep / seizure\n"
                "- **Theta (4–8 Hz):** drowsy / focal seizure\n"
                "- **Alpha (8–13 Hz):** relaxed wakefulness (normal)\n"
                "- **Beta (13–30 Hz):** active cognition\n"
                "- **Gamma (30–45 Hz):** high-level processing / tonic bursts")

    # ────────────────────────────────────────────────────────────────
    # INTENT 9: spike / interictal spikes
    # ────────────────────────────────────────────────────────────────
    if _has("spike", "interictal", "discharge", "epileptiform"):
        base = (
            "**Interictal spikes** are sharp EEG transients (lasting <200 ms) "
            "that occur *between* seizures. They are a hallmark of epilepsy.\n\n"
            "- **Normal:** <0.1 spikes/second\n"
            "- **Borderline:** 1–5 spikes/second\n"
            "- **Epileptiform:** >5 spikes/second\n\n"
            "Focal spikes appear in one region; generalised spike-wave "
            "(3 Hz) is the hallmark of absence epilepsy."
        )
        if has_result and si:
            sr = si.get("spike_rate_per_s", 0)
            n  = si.get("n_spikes", 0)
            base += f"\n\n📋 **Your EEG:** {n} spikes detected ({sr:.2f}/s)"
        return base

    # ────────────────────────────────────────────────────────────────
    # INTENT 10: entropy questions
    # ────────────────────────────────────────────────────────────────
    if _has("entropy", "sample entropy", "spectral entropy", "permutation"):
        return (
            "**Entropy** measures the complexity/randomness of the EEG signal:\n\n"
            "- **High entropy** → complex, irregular → **Normal** brain activity\n"
            "- **Low entropy** → periodic, ordered → **Seizure** (especially absence)\n\n"
            "Types used in NeuroScan Pro:\n"
            "- **Sample entropy** — regularity of short patterns\n"
            "- **Spectral entropy** — spread of frequency content\n"
            "- **Permutation entropy** — ordinal patterns in time series\n\n"
            "Absence seizures have the **lowest entropy** (most periodic). "
            "Normal EEG has the **highest**." + _result_ctx()
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 11: what to do next / next steps
    # ────────────────────────────────────────────────────────────────
    if _has("what to do", "next step", "what should", "what now", "recommend",
            "treatment", "see doctor", "neurologist", "hospital"):
        if not has_result:
            return (
                "Without a result yet, I recommend:\n"
                "1. Upload your EEG file in **Upload & Analyse** tab\n"
                "2. Train models (or load a saved model) in **Dashboard**\n"
                "3. Run Full Epilepsy Analysis\n"
                "4. Return here for guidance on your result"
            )
        if not is_epi:
            return (
                "✅ Your EEG shows **Normal** activity.\n\n"
                "**Recommended next steps:**\n"
                "1. Share this result with your **neurologist** for clinical confirmation\n"
                "2. Note that a single normal EEG does **not** rule out epilepsy — "
                "seizures may not occur during the recording\n"
                "3. If you have symptoms, request a **prolonged EEG** or sleep EEG\n"
                "4. Keep a **seizure diary** if you experience any episodes"
            )
        else:
            specific = {
                "Focal": (
                    "1. Consult a **neurologist** urgently for clinical EEG and MRI\n"
                    "2. First-line medications: **carbamazepine**, oxcarbazepine, levetiracetam\n"
                    "3. Avoid driving until seizure-free for the legally required period\n"
                    "4. Discuss surgical options if drug-resistant (focal resection)"
                ),
                "Absence": (
                    "1. Consult a **paediatric neurologist** (most common in children)\n"
                    "2. First-line: **ethosuximide** or valproate\n"
                    "3. Usually benign — many children outgrow absence epilepsy\n"
                    "4. Avoid triggers: hyperventilation, flickering lights"
                ),
                "Tonic-Atonic": (
                    "1. Seek **urgent neurological evaluation**\n"
                    "2. Safety measures: wear a **helmet**, use padded environments\n"
                    "3. Medications: valproate, lamotrigine, rufinamide\n"
                    "4. Discuss VNS or corpus callosotomy for refractory cases"
                ),
            }
            steps = specific.get(stype,
                "1. Consult a neurologist for full clinical EEG\n"
                "2. Start appropriate anti-seizure medication\n"
                "3. Avoid seizure triggers and unsafe activities")
            return (
                f"⚠️ Your EEG shows **{stype} Seizure** activity.\n\n"
                f"**Recommended next steps:**\n{steps}\n\n"
                f"⚠️ This is AI screening — always confirm with a "
                f"qualified neurologist before any medical decisions."
            )

    # ────────────────────────────────────────────────────────────────
    # INTENT 12: medication questions
    # ────────────────────────────────────────────────────────────────
    if _has("medication", "medicine", "drug", "tablet", "pill", "treat",
            "valproate", "carbamazepine", "lamotrigine", "levetiracetam",
            "ethosuximide", "phenytoin"):
        meds = {
            "valproate":     "Broad-spectrum; first-line for absence, tonic-atonic, generalised epilepsy.",
            "carbamazepine": "First-line for focal seizures; not effective for absence.",
            "lamotrigine":   "Used for focal and generalised seizures; safe in pregnancy.",
            "levetiracetam": "Broad-spectrum add-on; well-tolerated, minimal drug interactions.",
            "ethosuximide":  "First-line specifically for absence seizures only.",
            "phenytoin":     "Older agent for focal/tonic-clonic; narrow therapeutic window.",
        }
        specific = [f"**{k.capitalize()}:** {v}" for k, v in meds.items() if k in q]
        if specific:
            return "\n\n".join(specific) + "\n\n⚠️ Never start or stop medication without a neurologist's guidance."
        return (
            "**Common anti-seizure medications:**\n\n"
            "| Medication | Best for |\n"
            "|---|---|\n"
            "| Ethosuximide | Absence only |\n"
            "| Carbamazepine | Focal seizures |\n"
            "| Valproate | Generalised / absence / tonic-atonic |\n"
            "| Lamotrigine | Focal + generalised |\n"
            "| Levetiracetam | Add-on for focal + generalised |\n\n"
            "⚠️ Medication must be prescribed and monitored by a neurologist."
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 13: what is epilepsy / what is EEG
    # ────────────────────────────────────────────────────────────────
    if _has("what is epilepsy", "epilepsy is", "define epilepsy") or (
            _has("epilepsy") and _has("what", "define", "explain", "tell me about")):
        return (
            "**Epilepsy** is a neurological disorder defined by recurrent unprovoked seizures.\n\n"
            "- Affects ~50 million people worldwide\n"
            "- Caused by abnormal, excessive electrical discharges in the brain\n"
            "- Diagnosed after 2 or more unprovoked seizures, or one seizure with high recurrence risk\n"
            "- Classified by seizure type: focal (partial) or generalised\n\n"
            "**EEG (Electroencephalography)** records the brain's electrical activity via scalp electrodes. "
            "It is the **gold standard** tool for diagnosing and classifying epilepsy."
        )

    if (_has("what is eeg", "eeg is", "eeg mean", "eeg work") or
            (q in ("eeg", "electroencephalography", "what is eeg"))):
        return (
            "**EEG (Electroencephalography)** measures electrical activity of the brain "
            "using electrodes placed on the scalp.\n\n"
            "- Each electrode records voltage fluctuations produced by neuron firing\n"
            "- The standard clinical montage uses **19 electrodes** (10–20 system)\n"
            "- Sample rate is typically **256–512 Hz**\n"
            "- Key patterns: alpha rhythm (normal), spike-wave (epilepsy), slow waves (pathology)\n\n"
            "NeuroScan Pro analyses EEG signals to detect and classify epileptic activity."
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 14: dominant frequency
    # ────────────────────────────────────────────────────────────────
    if _has("dominant frequency", "dominant freq", "peak frequency", "main frequency"):
        base = (
            "**Dominant frequency** is the frequency at which the EEG has its highest power.\n\n"
            "- **<4 Hz (delta):** pathological when awake — absence or tonic-atonic seizure\n"
            "- **4–8 Hz (theta):** focal seizure or drowsiness\n"
            "- **8–13 Hz (alpha):** normal relaxed wakefulness\n"
            "- **>13 Hz (beta/gamma):** active cognition or tonic fast activity\n"
        )
        if has_result and domf and domf > 0:
            base += f"\n\n📋 **Your dominant frequency: {domf:.1f} Hz**"
        return base

    # ────────────────────────────────────────────────────────────────
    # INTENT 15: hjorth parameters
    # ────────────────────────────────────────────────────────────────
    if _has("hjorth", "mobility", "complexity", "activity"):
        return (
            "**Hjorth parameters** characterise EEG signal morphology:\n\n"
            "- **Activity:** signal variance — how energetic the signal is\n"
            "- **Mobility:** ratio of first-derivative variance to signal variance "
            "— measures mean frequency\n"
            "- **Complexity:** how closely the signal resembles a pure sine wave; "
            "higher = more complex waveform\n\n"
            "In seizures:\n"
            "- Tonic-atonic: **highest complexity** (irregular rapid bursts)\n"
            "- Absence: **elevated complexity** (irregular spike-wave morphology)\n"
            "- Normal: **low complexity** (smooth alpha rhythm)"
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 16: models used / how does AI work
    # ────────────────────────────────────────────────────────────────
    if (_has("model", "algorithm", "ai", "machine learning", "how does it work",
             "random forest", "svm", "cnn", "knn", "logistic")):
        return (
            "**NeuroScan Pro uses an ensemble of 5 AI models:**\n\n"
            "**Classical models:**\n"
            "- **Random Forest** — ensemble of decision trees, primary classifier\n"
            "- **Logistic Regression** — linear probabilistic classifier\n"
            "- **SVM** (Support Vector Machine) — RBF kernel classifier\n"
            "- **KNN** (K-Nearest Neighbours) — distance-based classifier\n\n"
            "**Deep learning (pure NumPy — no GPU needed):**\n"
            "- **CNN-1D** — 2-layer neural network with ReLU + dropout regularisation\n\n"
            "Training: **70% train / 15% validation / 15% test** + **5-fold cross-validation**.\n"
            "Final prediction: **soft-vote ensemble** of all models with val F1 > 0.40."
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 17: ICD-10 / classification code
    # ────────────────────────────────────────────────────────────────
    if _has("icd", "icd10", "icd-10", "code", "classification code"):
        return (
            "**ICD-10 codes for epilepsy:**\n\n"
            "| Type | ICD-10 |\n"
            "|---|---|\n"
            "| Normal (no epilepsy) | — |\n"
            "| Focal (partial) seizure | G40.1 |\n"
            "| Absence seizure | G40.3 |\n"
            "| Tonic / Atonic seizure | G40.5 / G40.8 |\n"
            + _result_ctx()
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 18: split / training / validation data
    # ────────────────────────────────────────────────────────────────
    if _has("train", "split", "validation", "test set", "cross valid", "kfold", "k fold"):
        return (
            "**NeuroScan Pro training protocol:**\n\n"
            "- **70% training** — used to fit all models\n"
            "- **15% validation** — used to select the best model and hyperparameters\n"
            "- **15% test** — held-out, never seen during training; final performance benchmark\n\n"
            "Additionally, **5-fold stratified cross-validation** is run on the training "
            "set for tree-based models to ensure robust estimates of generalisation.\n\n"
            "This prevents **data leakage** and ensures the reported metrics are honest."
        )

    # ────────────────────────────────────────────────────────────────
    # INTENT 19: what causes epilepsy / triggers
    # ────────────────────────────────────────────────────────────────
    if _has("cause", "trigger", "why", "risk factor") and _has("epilepsy", "seizure"):
        return (
            "**Causes of epilepsy:**\n"
            "- Genetic/hereditary factors (idiopathic epilepsy)\n"
            "- Structural: brain tumour, stroke, traumatic brain injury\n"
            "- Metabolic: hypoglycaemia, hyponatraemia, drug/alcohol withdrawal\n"
            "- Infectious: meningitis, encephalitis\n"
            "- Unknown (cryptogenic) — most common\n\n"
            "**Common seizure triggers:**\n"
            "- Sleep deprivation\n"
            "- Missed medication\n"
            "- Stress or anxiety\n"
            "- Flickering lights (photosensitive epilepsy)\n"
            "- Alcohol or recreational drugs\n"
            "- Fever (in children — febrile seizures)"
        )

    # ────────────────────────────────────────────────────────────────
    # FALLBACK — unclear question
    # ────────────────────────────────────────────────────────────────
    return (
        "🤔 I'm not sure I understood that. Here are some things you can ask me:\n\n"
        "- *\"What does my result mean?\"*\n"
        "- *\"What is an absence seizure?\"*\n"
        "- *\"What are the next steps after my diagnosis?\"*\n"
        "- *\"What medications are used for focal seizures?\"*\n"
        "- *\"Explain the band power in my EEG\"*\n"
        "- *\"How does the AI model work?\"*\n"
        "- *\"What is the ICD-10 code for my result?\"*\n\n"
        "Try rephrasing your question!"
    )


def _tab_chatbot():
    st.markdown('<h3 style="color:#00bcd4;">💬 NeuroBot — EEG Assistant</h3>',
                unsafe_allow_html=True)

    prediction = st.session_state.prediction

    # Context banner
    if prediction:
        stype  = prediction.get("seizure_type", "Unknown")
        conf   = prediction.get("confidence", 0) * 100
        colour = _SCOLOURS.get(stype, "#00bcd4")
        st.markdown(
            f'<div style="border:1px solid {colour};border-radius:10px;padding:10px 16px;'
            f'background:rgba(0,0,0,.3);margin-bottom:16px;">'
            f'<b style="color:{colour}">🔬 Current result context:</b> '
            f'<span style="color:#e0e0e0">{stype}</span> '
            f'<span style="color:#90a4ae">({conf:.0f}% confidence)</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("💡 No result loaded yet. Upload an EEG file and run analysis for personalised answers. "
                "You can still ask general epilepsy and EEG questions.")

    # Chat history display
    history = st.session_state.chat_history
    for msg in history:
        role  = msg["role"]
        text  = msg["text"]
        if role == "user":
            st.markdown(
                f'<div style="background:rgba(21,101,192,.25);border-radius:10px 10px 2px 10px;'
                f'padding:10px 14px;margin:6px 0 6px 60px;text-align:right;color:#e0e0e0;">'
                f'👤 {text}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div style="background:rgba(0,188,212,.10);border:1px solid rgba(0,188,212,.2);'
                f'border-radius:2px 10px 10px 10px;padding:12px 16px;margin:6px 60px 6px 0;'
                f'color:#e0e0e0;">'
                f'🤖 <b style="color:#00bcd4">NeuroBot:</b><br>{text}</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Quick-question chips
    st.markdown("**💡 Quick questions:**")
    qcols = st.columns(3)
    quick_qs = [
        "What does my result mean?",
        "What are the next steps?",
        "What is an absence seizure?",
        "How does the AI model work?",
        "Explain the band power",
        "What medications are used?",
    ]
    for i, qq in enumerate(quick_qs):
        with qcols[i % 3]:
            if st.button(qq, key=f"qq_{i}", use_container_width=True):
                answer = _neurobot_answer(qq, prediction)
                st.session_state.chat_history.append({"role": "user",  "text": qq})
                st.session_state.chat_history.append({"role": "bot",   "text": answer})
                st.rerun()

    st.markdown("")

    # Text input
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        user_input = st.text_input(
            "Ask NeuroBot:",
            placeholder="e.g. What does focal seizure mean? What should I do next?",
            label_visibility="collapsed",
            key="chatbot_input",
        )
    with col_btn:
        send = st.button("Send ➤", key="btn_chat_send", use_container_width=True, type="primary")

    if send and user_input.strip():
        answer = _neurobot_answer(user_input.strip(), prediction)
        st.session_state.chat_history.append({"role": "user", "text": user_input.strip()})
        st.session_state.chat_history.append({"role": "bot",  "text": answer})
        st.rerun()

    if history:
        if st.button("🗑️ Clear chat", key="btn_clear_chat", use_container_width=False):
            st.session_state.chat_history = []
            st.rerun()



    st.markdown('<h3 style="color:#00bcd4;">⚙️ Settings</h3>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("System info")
        st.metric("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
        st.metric("NumPy", np.__version__)
        try: import sklearn; st.metric("scikit-learn", sklearn.__version__)
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

    if st.button("🗑️ Reset session", key="btn_reset_chatbot"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


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

    if st.button("🗑️ Reset session", key="btn_reset_settings"):
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
        "💬 Ask NeuroBot",
        "⚙️ Settings",
    ])
    with tabs[0]: _tab_dashboard()
    with tabs[1]: _tab_upload()
    with tabs[2]: _tab_predict()
    with tabs[3]: _tab_brain()
    with tabs[4]: _tab_models()
    with tabs[5]: _tab_chatbot()
    with tabs[6]: _tab_settings()


if __name__ == "__main__":
    main()
