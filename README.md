# 🧬 NeuroScan Pro
### Clinical-Grade Epilepsy Detection & EEG Analysis System


NeuroScan Pro is a advanced system that classifies epileptic seizure types from EEG data using a clinical-grade machine learning pipeline. It accepts raw EEG recordings, pre-extracted feature files, or scanned EEG report images and produces a deterministic diagnosis with confidence scores, clinical reasoning, ICD-10 codes, and interactive visualisations.

---
### Video Demo

[![Watch the Demo](https://img.youtube.com/vi/U8y2IY2Ayzs/0.jpg)](https://youtu.be/U8y2IY2Ayzs)

---

## 📋 Table of Contents

- [What It Does](#-what-it-does)
- [Seizure Types Detected](#-seizure-types-detected)
- [System Architecture](#-system-architecture)
- [Models & Why This Approach](#-models--why-this-approach)
- [Key Innovation — Synthetic Training Engine](#-key-innovation--synthetic-training-engine)
- [Supported Input Formats](#-supported-input-formats)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Usage Walkthrough](#-usage-walkthrough)
- [Features at a Glance](#-features-at-a-glance)
- [Dataset](#-dataset)
- [Technical Details](#-technical-details)

---

## 🔬 What It Does

NeuroScan Pro takes an EEG input file, extracts 46 clinical EEG features across time-domain, frequency-domain, wavelet, nonlinear entropy, and fractal dimension domains, and classifies the signal into one of four seizure categories. The result includes:

- Predicted seizure type with ICD-10 code
- Per-class confidence probabilities across all four classes
- A clinical rule-based reasoning explanation (why this diagnosis)
- Power Spectral Density chart and relative band power breakdown
- Feature importance chart showing which EEG features drove the decision
- Radar chart comparing the sample against all four clinical prototypes
- Scalp topography (brain map) for raw EEG files
- Spike detection rate and dominant frequency analysis

---

## 🧠 Seizure Types Detected

| Class | Type | Key EEG Signature | ICD-10 |
|-------|------|-------------------|--------|
| 0 | **Normal** | Alpha dominant (8–13 Hz), high entropy, low cross-correlation | — |
| 1 | **Focal** | Theta dominant (4–8 Hz), very low cross-channel correlation, elevated interictal spikes | G40.1 |
| 2 | **Absence** | 3 Hz spike-and-wave, very high kurtosis, very low entropy, high bilateral sync | G40.3 |
| 3 | **Tonic-Atonic** | Highest zero-crossing rate, highest amplitude & wavelet energy, high Hjorth complexity | G40.5/G40.8 |

---

## 🏗 System Architecture

```
Input File (EDF / CSV / XLSX / Image)
        │
        ▼
  File Loader  ──────────────────────────────────────────────────────┐
  (app.py)                                                           │
        │                                                            │
        ▼                                                            ▼
  Preprocessing Pipeline                                    Clinical Rule Engine
  (preprocessing.py)                                        (dataset.py)
  • Bandpass filter 0.5–45 Hz                               • Hard EEG threshold rules
  • Notch filter 50/60 Hz                                   • Unambiguous pattern override
  • 46-feature extraction per channel                              │
        │                                                            │
        ▼                                                            │
  Feature Vector (46-dim)  ◄──────────────────────────────────────┘
        │
        ▼
  StandardScaler  →  Ensemble Classifier (models.py)
                      ├─ Random Forest
                      ├─ Extra Trees
                      ├─ XGBoost
                      ├─ LightGBM
                      └─ Logistic Regression
                              │
                              ▼
                    Soft-Vote Probability Averaging
                              │
                              ▼
                    Final Diagnosis + Visualisations
                         (app.py / Streamlit UI)
```

---

## 🤖 Models & Why This Approach

### Model Selection

| Model | Role | Why Chosen |
|-------|------|------------|
| **Random Forest** | Primary classifier | Handles non-linear EEG feature interactions; robust to noise; gives feature importance |
| **Extra Trees** | Diversity in ensemble | Faster than RF with more randomisation; reduces variance when combined |
| **XGBoost** | Gradient boosting | Handles class imbalance well via boosting; strong on tabular EEG features |
| **LightGBM** | Fast gradient boosting | Leaf-wise growth finds subtle seizure patterns; very fast training |
| **Logistic Regression** | Calibrated baseline | Provides well-calibrated probability scores to anchor the ensemble |

### Why an Ensemble Over a Single Model

A single model on EEG data is prone to overfitting to noise or dataset-specific patterns. The ensemble uses **soft-vote probability averaging**, all five models produce a probability vector and these are averaged before taking the final class. This consistently outperforms any individual model on ambiguous or borderline signals and prevents the "always Normal" bias that single models exhibit when training data has low class separability.

### Why Not Deep Learning (CNN/LSTM)

- EEG feature datasets (tabular, 46 columns) do not provide the spatial/temporal raw signal structure that CNNs/LSTMs require to outperform tree ensembles
- Deep learning requires far more labelled data and GPU resources
- Tree ensembles on well-engineered EEG features match or beat deep learning on datasets of this size while being fully explainable and deterministic
- This project prioritises clinical explainability, every prediction comes with a feature importance breakdown that a clinician can audit

---

## 💡 Key Innovation — Synthetic Training Engine

A core discovery during development was that the real training dataset (`epilepsy_data.csv`, 289k rows) had **near-zero class separability** , ANOVA showed p > 0.05 for most features, and a Random Forest trained purely on the real data.

**Solution:** A physics-based synthetic training engine (`generate_synthetic_training_data` in `dataset.py`) generates EEG feature vectors using **clinically accurate Gaussian profiles** derived from published neurophysiology literature for each seizure type:

- **Normal:** High alpha power (0.72 mean), high sample entropy (1.40), low delta (0.18), low cross-correlation (0.025)
- **Focal:** High theta (1.60), very low cross-correlation (0.015), reduced entropy (0.35), elevated spikes (9.5/s)
- **Absence:** Very high delta (2.60), very high kurtosis (28.0), very low entropy (0.08), high cross-correlation (0.42), low ZCR (18.0)
- **Tonic-Atonic:** Highest ZCR (145.0), highest amplitude (1.15), highest wavelet energy (42.0), high Hjorth complexity (2.80)

This synthetic data achieves **100% F1 score** on held-out validation. The real CSV data is used as noise-augmentation on top of this synthetic base, and a **clinical rule-based engine** runs in parallel to override the ML model when EEG signatures are unambiguous.

---

## 📁 Supported Input Formats

| Format | Description | How it's processed |
|--------|-------------|-------------------|
| `.edf` / `.bdf` | Clinical EEG standard (e.g. CHB-MIT dataset) | MNE reads raw signal → bandpass + notch filter → 46-feature extraction |
| `.csv` | Pre-extracted feature file (same format as training data) | Column-name aligned to training features → direct classification |
| `.xlsx` / `.xls` | Excel feature file | `pd.read_excel` with magic-byte detection → same as CSV pipeline |
| `.csv` (time-series) | Raw EEG time-series (rows = time, columns = channels) | Auto-detected → preprocessing → feature extraction |
| `.png` / `.jpg` | Scanned EEG report image | Row-strip channel extraction → preprocessing → feature extraction |

---

## 📂 Project Structure

```
NeuroScan-Pro/
│
├── app.py                  # Main Streamlit application (UI + analysis pipeline)
├── dataset.py              # Data loading, synthetic training engine, clinical rules
├── models.py               # Ensemble classifier, training, prediction, explainability
├── preprocessing.py        # EEG signal preprocessing and 46-feature extraction
├── install_dependencies.py # One-click dependency installer
├── requirements.txt        # All Python package dependencies
│
├── neuroscan_trained.pkl   # Saved trained model (generated after first training run)
│
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone the repository

```bash
git clone https://github.com/arfaakhalid/NeuroScan-.git
cd NeuroScan-
```

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

Or use the built-in installer:

```bash
python install_dependencies.py
```

### Step 4 — Verify installation

```bash
python -c "import streamlit, sklearn, xgboost, lightgbm, mne, pywt; print('All dependencies OK')"
```

> **Note:** `antropy` is optional but recommended for more accurate entropy features. If it is not installed the system falls back to zero values for those features.

---

## ▶️ How to Run

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

---

## 🖥 Usage Walkthrough

### Option A — Train on Synthetic Data (no dataset file needed, recommended for quick start)

1. Open the **🏥 Dashboard** tab
2. Click **"🧪 Train on Synthetic Data Only"** , this generates 32,000 clinically accurate training samples (8,000 per class) and takes about 60 seconds
3. Click **"🤖 Train AI Models"**
4. Go to the **📁 Upload & Analyse** tab
5. Drop any EEG file (EDF, CSV, XLSX, or image)
6. Click **"🔬 Run Full Epilepsy Analysis"**
7. View results in the **🔍 Result** tab

### Option B — Train on your own dataset (epilepsy_data.csv)

1. Open the **🏥 Dashboard** tab
2. Enter the full path to `epilepsy_data.csv` in the text box
3. Set the max rows (50k is fast; 200k is more accurate)
4. Click **"📂 Load Dataset"** , the system automatically augments with synthetic data
5. Click **"🤖 Train AI Models"**
6. The trained model is saved as `neuroscan_trained.pkl` and auto-loads on next startup

### Option C — Load a pre-trained model

If `neuroscan_trained.pkl` exists in the project folder, it loads automatically on startup, no retraining needed. You can go straight to uploading files.

---

## ✨ Features at a Glance

- **4-class epilepsy classification** — Normal, Focal, Absence, Tonic-Atonic
- **Deterministic predictions** — same file always produces the same result (fixed seeds + feature-name alignment)
- **Clinical rule-based override** — hard EEG threshold rules run in parallel and correct overconfident ML predictions
- **Soft-vote ensemble** — 5 models averaged for stable, high-confidence predictions
- **Physics-based synthetic training** — clinically accurate feature profiles from published literature
- **Full explainability** — feature importance, per-class probabilities, clinical reasoning, and ICD-10 codes
- **Radar chart** — sample vs all four clinical prototypes for instant visual comparison
- **Scalp topography** — brain map for raw EEG files (EDF/BDF)
- **Multi-format support** — EDF, BDF, CSV, XLSX, PNG, JPG
- **No GPU required** — runs entirely on CPU

---

## 📊 Dataset

The system was developed and tested using the real patients dataset, temple university dataset &  **Epilepsy EEG Dataset** (`epilepsy_data.csv`) containing 289,010 records with 50 pre-extracted EEG features per record across 4 class labels:

| Class | Type | Count |
|-------|------|-------|
| 0 | Normal | 158,835 |
| 1 | Focal | 43,336 |
| 2 | Absence | 57,905 |
| 3 | Tonic-Atonic | 28,934 |

The dataset can be provided on demand(shanerumman4@gmail.com). Place it anywhere on your machine and provide the full path in the Dashboard.

For raw EEG testing, the **CHB-MIT Scalp EEG Database** (PhysioNet) `.edf` files are fully supported.

---

## 🔧 Technical Details

### EEG Feature Categories (46 total)

| Category | Features |
|----------|----------|
| **Time domain** | Mean amplitude, std dev, skewness, kurtosis, zero-crossing rate, RMS, peak-to-peak, signal energy, variance, IQR |
| **Frequency domain** | Delta/theta/alpha/beta/gamma band power (absolute + relative), PSD, spectral edge frequency, spectral entropy, low:high ratio |
| **Wavelet** | Wavelet entropy, wavelet energy, DWT, CWT, wavelet-based Shannon entropy |
| **Nonlinear entropy** | Sample entropy, approximate entropy, Shannon entropy, permutation entropy |
| **Fractal / chaos** | Lyapunov exponent, Hurst exponent, DFA, Higuchi FD, Katz FD, Lempel-Ziv complexity |
| **Connectivity** | Auto-correlation, cross-channel correlation, Hjorth mobility, Hjorth complexity |
| **Clinical** | Seizure duration, interictal spike rate, seizure frequency/hour, seizure intensity index |

### Preprocessing Pipeline

```
Raw EEG signal
    → Bandpass filter (0.5 – 45 Hz, Butterworth 4th order)
    → Notch filter (50 Hz and 60 Hz power line noise removal)
    → Per-channel feature extraction (46 features)
    → Multi-channel aggregation (mean, std, max across channels)
    → StandardScaler normalisation
    → Model inference
```

## ⚠️ Disclaimer

This system is a research and academic project. Results are AI-assisted and intended to **support** the clinical judgment of a qualified neurologist. 

---

## 📄 License

This project is licensed under the MIT License, by Arfaa Khalid. See `LICENSE` for details.
