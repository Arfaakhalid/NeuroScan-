"""
preprocessing.py  —  NeuroScan Pro
Clinical EEG signal processing and feature extraction.

"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import signal as sp_signal, stats
from scipy.signal import butter, filtfilt, welch, find_peaks
import pywt
from sklearn.preprocessing import StandardScaler

try:
    import antropy as ant
    _ANTROPY = True
except ImportError:
    _ANTROPY = False


# ── EEG frequency bands ────────────────────────────────────────
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _safe(func, *args, default=0.0, **kwargs):
    try:
        v = func(*args, **kwargs)
        v = float(v)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _notch(sig, freq, fs, Q=30):
    nyq = fs / 2.0
    if freq >= nyq:
        return sig
    b, a = sp_signal.iirnotch(freq / nyq, Q)
    return filtfilt(b, a, sig)


# ──────────────────────────────────────────────────────────────
def preprocess(eeg: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Band-pass 0.5-45 Hz + notch 50/60 Hz per channel.
    eeg : (n_channels, n_samples) or (n_samples,)
    """
    squeezed = False
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]
        squeezed = True

    out = np.zeros_like(eeg, dtype=np.float64)
    for c in range(eeg.shape[0]):
        s = eeg[c].astype(np.float64)
        s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
        if len(s) >= 9:
            b, a = butter(4, 0.5 / (fs / 2), btype="high")
            s = filtfilt(b, a, s)
            hi_cut = min(45.0, fs / 2 - 1.0)
            b, a = butter(4, hi_cut / (fs / 2), btype="low")
            s = filtfilt(b, a, s)
            for fn in (50.0, 60.0):
                s = _notch(s, fn, fs)
        out[c] = s

    return out.squeeze() if squeezed else out


# ──────────────────────────────────────────────────────────────
def extract_channel_features(sig: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """~80 features from a single EEG channel."""
    feats = []
    n = max(len(sig), 1)

    # --- Time-domain (19 features) ---
    mn     = float(np.mean(sig))
    sd     = float(np.std(sig)) + 1e-12
    var    = float(np.var(sig))
    med    = float(np.median(sig))
    iqr_v  = float(stats.iqr(sig))
    mx     = float(np.max(sig))
    mi     = float(np.min(sig))
    ptp    = float(np.ptp(sig))
    rms    = float(np.sqrt(np.mean(sig ** 2)))
    sk     = _safe(stats.skew,     sig, default=0.0)
    ku     = _safe(stats.kurtosis, sig, default=0.0)
    mav    = float(np.mean(np.abs(sig)))
    energy = float(np.sum(sig ** 2))
    feats += [mn, sd, var, med, iqr_v, mx, mi, ptp, rms, sk, ku, mav, energy]

    zc_idx   = np.where(np.diff(np.sign(sig)))[0]
    zcr      = len(zc_idx) / n
    zcr_mean = float(np.mean(np.diff(zc_idx))) if len(zc_idx) > 1 else 0.0
    feats   += [zcr, zcr_mean]

    d1 = np.diff(sig); d2 = np.diff(d1)
    v0, v1, v2 = float(np.var(sig)), float(np.var(d1)), float(np.var(d2))
    activity   = v0
    mobility   = float(np.sqrt(v1 / v0))  if v0 > 1e-12 else 0.0
    complexity = (float(np.sqrt(v2 / v1) / mobility)
                  if (v1 > 1e-12 and mobility > 1e-12) else 0.0)
    line_len   = float(np.sum(np.abs(np.diff(sig))))
    feats     += [activity, mobility, complexity, line_len]

    peaks_idx, _ = find_peaks(np.abs(sig), height=sd * 3,
                               distance=max(1, int(fs * 0.05)))
    spike_rate = len(peaks_idx) / (n / fs)
    feats.append(spike_rate)

    # --- Frequency-domain (14 features) ---
    nperseg = max(4, min(int(fs * 2), n))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    total_pwr  = float(np.sum(psd)) + 1e-12

    band_abs = {}
    for bname, (lo, hi) in EEG_BANDS.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        bp  = float(np.sum(psd[idx])) if len(idx) else 0.0
        band_abs[bname] = bp
        feats += [bp, bp / total_pwr]

    feats += [
        band_abs["theta"] / (band_abs["alpha"] + 1e-12),
        band_abs["delta"] / (band_abs["alpha"] + 1e-12),
        band_abs["beta"]  / (band_abs["alpha"] + 1e-12),
    ]

    cum  = np.cumsum(psd); cumn = cum / (cum[-1] + 1e-12)
    sef95 = float(freqs[min(np.searchsorted(cumn, 0.95), len(freqs)-1)])
    sef50 = float(freqs[min(np.searchsorted(cumn, 0.50), len(freqs)-1)])
    mf    = float(np.sum(freqs * psd) / total_pwr)
    pnorm = psd / total_pwr
    sp_ent= float(-np.sum(pnorm * np.log2(pnorm + 1e-12)))
    feats += [sef95, sef50, mf, sp_ent]

    # --- Wavelet (24 features) ---
    try:
        coeffs = pywt.wavedec(sig, "db4", level=5)
        for c in coeffs:
            if len(c) == 0:
                feats += [0.0, 0.0, 0.0, 0.0]
            else:
                feats += [
                    float(np.mean(np.abs(c))),
                    float(np.std(c)),
                    float(np.sum(c ** 2)),
                    _safe(stats.kurtosis, c, default=0.0) if len(c) > 3 else 0.0,
                ]
    except Exception:
        feats += [0.0] * 24

    # --- Nonlinear (5 features) ---
    if _ANTROPY:
        feats.append(_safe(ant.sample_entropy,        sig, order=2, metric="chebyshev"))
        feats.append(_safe(ant.perm_entropy,          sig, order=3, normalize=True))
        feats.append(_safe(ant.detrended_fluctuation, sig))
        feats.append(_safe(ant.higuchi_fd,            sig, kmax=10))
        binary = (sig > float(np.median(sig))).astype(bool)
        feats.append(_safe(ant.lziv_complexity,       binary, normalize=True))
    else:
        feats += [0.0] * 5

    return np.array(feats, dtype=np.float32)


# ──────────────────────────────────────────────────────────────
def extract_features(eeg: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Multi-channel feature extraction.
    eeg : (n_channels, n_samples) or (n_samples,)
    Returns 1-D float32 vector.
    """
    if eeg.ndim == 1:
        eeg = eeg[np.newaxis, :]
    n_ch = eeg.shape[0]

    per_ch = np.stack([extract_channel_features(eeg[c], fs) for c in range(n_ch)], axis=0)
    mean_f = np.mean(per_ch, axis=0)
    std_f  = np.std(per_ch,  axis=0)
    max_f  = np.max(per_ch,  axis=0)

    if n_ch > 1:
        corr_vals = []
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                c = float(np.corrcoef(eeg[i], eeg[j])[0, 1])
                corr_vals.append(c if np.isfinite(c) else 0.0)
        coh = [float(np.mean(corr_vals)),
               float(np.std(corr_vals)),
               float(np.max(np.abs(corr_vals)))]
    else:
        coh = [0.0, 0.0, 0.0]

    feat_vec = np.concatenate([mean_f, std_f, max_f, coh]).astype(np.float32)
    return np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)


# ── Utility helpers ────────────────────────────────────────────
def detect_spikes(sig: np.ndarray, fs: float = 256.0,
                  sigma: float = 3.5) -> dict:
    sd = float(np.std(sig))
    peaks, _ = find_peaks(np.abs(sig), height=sd * sigma,
                           distance=max(1, int(fs * 0.05)))
    dur_s = max(len(sig) / fs, 1e-6)
    return {
        "spike_times_s":    (peaks / fs).tolist(),
        "spike_rate_per_s": float(len(peaks) / dur_s),
        "spike_amplitudes": np.abs(sig[peaks]).tolist() if len(peaks) else [],
        "n_spikes":         int(len(peaks)),
    }


def dominant_frequency(sig: np.ndarray, fs: float = 256.0) -> float:
    nperseg = max(4, min(int(fs * 2), len(sig)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    mask = (freqs >= 0.5) & (freqs <= 45.0)
    return float(freqs[mask][np.argmax(psd[mask])]) if np.any(mask) else 0.0


def band_power_profile(sig: np.ndarray, fs: float = 256.0) -> dict:
    nperseg = max(4, min(int(fs * 2), len(sig)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    total = float(np.sum(psd)) + 1e-12
    out = {}
    for name, (lo, hi) in EEG_BANDS.items():
        idx = np.where((freqs >= lo) & (freqs <= hi))[0]
        bp  = float(np.sum(psd[idx])) if len(idx) else 0.0
        out[name] = {"abs": bp, "rel": bp / total}
    return out


# ── Legacy shim ────────────────────────────────────────────────
class AdvancedEEGPreprocessor:
    def __init__(self, fs: float = 256.0, n_channels: int = 19):
        self.fs = fs
        self.n_channels = n_channels
        self.scaler = StandardScaler()

    def preprocess_signals(self, s): return preprocess(s, self.fs)
    def extract_advanced_features(self, s):
        fv = extract_features(s, self.fs)
        return fv, [f"f{i}" for i in range(len(fv))]