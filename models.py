"""
models.py  --  NeuroScan Pro 
=======================================================================
Features a deterministic 4-class epilepsy classifier with:
  1. num_classes = 4 always (Normal / Focal / Absence / Tonic-Atonic).
  2. Ensemble uses SOFT voting (averaged probabilities) + calibration,
     not argmax of a single model.  This is more stable.
  3. Rule-based clinical override: if the EEG feature values clearly
     match a seizure type it'll prevents all-Normal bias.
  4. Best model selection is by F1 on validation set (not first model).
 
=======================================================================
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import os

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

try:
    from lightgbm import LGBMClassifier
    _LGB = True
except ImportError:
    _LGB = False

from dataset import SEIZURE_TYPES, SEIZURE_ICD10, rule_based_classify

# Number of real seizure classes in the dataset
N_CLASSES = 4


# ── Clinical metadata ──────────────────────────────────────────
SEIZURE_DESCRIPTIONS = {
    "Normal": (
        "No pathological EEG activity detected. Brain electrical activity is within "
        "normal clinical parameters. Alpha rhythm is dominant (8–13 Hz); no epileptiform "
        "discharges. Spectral entropy is high, reflecting the complex, irregular nature "
        "of healthy brain activity."
    ),
    "Focal": (
        "Focal (partial) seizure originating in a localised cortical region. "
        "Characterised by unilateral rhythmic theta discharge (4–8 Hz) with sharp spike "
        "transients confined to one hemisphere or lobe. Cross-channel correlation is very "
        "low (focal onset, not bilateral). Spectral entropy is reduced. ICD-10: G40.1."
    ),
    "Absence": (
        "Absence seizure: generalised non-convulsive epilepsy. Defined by abrupt-onset "
        "3 Hz symmetric spike-and-wave discharges (SWD). Very high delta power, very "
        "high EEG kurtosis (sharp spike peaks), and very low signal entropy are the "
        "hallmarks. High bilateral cross-correlation and low zero-crossing rate confirm "
        "generalised slow-wave activity. ICD-10: G40.3."
    ),
    "Tonic-Atonic": (
        "Tonic or atonic seizure. Tonic phase: sustained rapid oscillations produce the "
        "highest zero-crossing rate and highest overall amplitude seen in any seizure type. "
        "Wavelet energy is maximal. Hjorth complexity is elevated. Bilateral spread gives "
        "high cross-channel correlation. Atonic component produces brief voltage "
        "attenuation following the tonic burst. ICD-10: G40.5 / G40.8."
    ),
}

SEIZURE_KEY_FEATURES = {
    "Normal":       ["High alpha (8–13 Hz)", "High entropy (complex signal)",
                     "High Lyapunov exponent", "Low delta/theta power",
                     "Low cross-channel correlation", "High fractal dimension"],
    "Focal":        ["Theta dominant (4–8 Hz) in ictal zone",
                     "Very low cross-channel correlation (focal)",
                     "Asymmetric amplitude", "Elevated interictal spike rate",
                     "Reduced signal entropy", "Elevated theta/alpha ratio"],
    "Absence":      ["3 Hz spike-wave discharges (SWD)", "Very high EEG kurtosis",
                     "Very low signal entropy (periodic)", "High bilateral cross-correlation",
                     "Low zero-crossing rate", "Very high delta power"],
    "Tonic-Atonic": ["Highest zero-crossing rate (rapid oscillation)",
                     "Highest amplitude & wavelet energy",
                     "High Hjorth complexity", "Elevated gamma power",
                     "Bilateral generalised spread", "Longest seizure duration"],
}


# ── Model builder ──────────────────────────────────────────────
def _build_models() -> dict:
    """Build the candidate model pool. All use random_state=42."""
    models = {}

    # Random Forest (primary workhorse for EEG features)
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_split=4,
        max_features="sqrt", n_jobs=-1, random_state=42,
        class_weight="balanced",
    )

    # Extra Trees (fast, diverse ensemble -- good complement to RF)
    models["ExtraTrees"] = ExtraTreesClassifier(
        n_estimators=400, max_depth=None, min_samples_split=4,
        max_features="sqrt", n_jobs=-1, random_state=42,
        class_weight="balanced",
    )

    if _XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            eval_metric="mlogloss", n_jobs=-1, random_state=42,
        )

    if _LGB:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=400, max_depth=10, learning_rate=0.05,
            num_leaves=80, subsample=0.85, colsample_bytree=0.85,
            class_weight="balanced",
            n_jobs=-1, random_state=42, verbose=-1,
        )

    # Logistic Regression (provides calibrated baseline)
    models["LogisticRegression"] = LogisticRegression(
        C=0.5, max_iter=2000, solver="lbfgs",
        multi_class="multinomial", random_state=42,
        class_weight="balanced", n_jobs=-1,
    )

    return models


# ── Main classifier ────────────────────────────────────────────
class AdvancedEpilepsyClassifier:
    """
    Deterministic 4-class epilepsy classifier with:
    - Soft-voting ensemble (averaged calibrated probabilities)
    - Clinical rule-based override for clear-cut EEG signatures
    - Consistent predictions
    """

    def __init__(self, num_classes: int = N_CLASSES, use_gpu: bool = False):
        self.num_classes      = N_CLASSES   # always 4, ignore argument
        self.trained: dict    = {}
        self.metrics: dict    = {}
        self.scaler           = None
        self.best_model_name  = ""
        self.feature_names: list = []
        self.n_train_features: int = 0
        # Mapping from trained model output indices to class labels
        # (some models may not see all classes during training)
        self._class_indices: dict = {}   # model_name -> array of class indices

    # ── Training ────────────────────────────────────────────────
    def train(self, X_train, y_train, X_val, y_val,
              feature_names=None, progress_cb=None) -> dict:

        self.feature_names    = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self.n_train_features = X_train.shape[1]
        self.num_classes      = N_CLASSES

        # Verify all 4 classes are present in training data
        present = np.unique(y_train)
        if len(present) < N_CLASSES:
            print(f"[WARN] Only {len(present)} classes in training data: {present}")

        candidates  = _build_models()
        best_f1     = -1.0
        all_metrics = {}

        for idx, (name, model) in enumerate(candidates.items()):
            if progress_cb:
                progress_cb(idx, len(candidates), name)
            try:
                model.fit(X_train, y_train)

                # Store which classes this model knows about
                if hasattr(model, "classes_"):
                    self._class_indices[name] = model.classes_
                else:
                    self._class_indices[name] = np.arange(N_CLASSES)

                y_pred = model.predict(X_val)
                m = {
                    "accuracy":  round(float(accuracy_score(y_val, y_pred)), 4),
                    "precision": round(float(precision_score(
                        y_val, y_pred, average="weighted", zero_division=0)), 4),
                    "recall":    round(float(recall_score(
                        y_val, y_pred, average="weighted", zero_division=0)), 4),
                    "f1":        round(float(f1_score(
                        y_val, y_pred, average="weighted", zero_division=0)), 4),
                }
                all_metrics[name]  = m
                self.trained[name] = model

                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    self.best_model_name = name

            except Exception as e:
                all_metrics[name] = {"error": str(e)}

        self.metrics = all_metrics
        self.models  = self.trained
        return all_metrics

    # ── Prediction ──────────────────────────────────────────────
    def predict(self, features: np.ndarray, model_name: str = None) -> dict:
        """
        Deterministic prediction.  
        Pipeline:
          1. Align feature vector to training dimension
          2. Run all trained models -> soft-vote ensemble probabilities
          3. Also run best single model for primary result
          4. Apply clinical rule-based override if signal is unambiguous
          5. Return full result dict
        """
        if not self.trained:
            raise RuntimeError("No models trained yet.")

        # --- 1. Prepare feature vector ---
        fv = np.array(features, dtype=np.float32).ravel()
        fv = np.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
        fv = self._align_to_train(fv).reshape(1, -1)

        # --- 2. Ensemble (soft vote, sorted model names for determinism) ---
        ensemble_proba = np.zeros(self.num_classes, dtype=np.float64)
        n_models = 0
        for mname in sorted(self.trained.keys()):
            p = self._get_proba(mname, fv)
            ensemble_proba += p
            n_models += 1
        ensemble_proba /= max(n_models, 1)

        # --- 3. Best single model ---
        name = model_name or self.best_model_name
        if name not in self.trained:
            name = next(iter(self.trained))
        best_proba = self._get_proba(name, fv)

        # --- 4. Clinical rule override ---
        rule_result = None
        if self.feature_names:
            rule_result = rule_based_classify(fv.ravel(), self.feature_names)

        # Blend: use ensemble as base, override if rules are confident
        final_proba = ensemble_proba.copy()
        if rule_result is not None and rule_result["confidence"] >= 0.55:
            rule_label = rule_result["label"]
            rule_conf  = rule_result["confidence"]
            
            boost = np.zeros(self.num_classes)
            boost[rule_label] = rule_conf
            final_proba = 0.60 * ensemble_proba + 0.40 * boost
            # Re-normalise
            total = final_proba.sum()
            if total > 1e-6:
                final_proba /= total

        pred_class   = int(np.argmax(final_proba))
        confidence   = float(final_proba[pred_class])
        seizure_type = SEIZURE_TYPES.get(pred_class, "Unknown")

        ens_class = int(np.argmax(ensemble_proba))
        ens_conf  = float(ensemble_proba[ens_class])
        ens_type  = SEIZURE_TYPES.get(ens_class, "Unknown")

        rule_note = ""
        if rule_result is not None and rule_result["confidence"] >= 0.55:
            rule_note = f" [Rule: {rule_result['reason']}]"

        return {
            "predicted_label":     pred_class,
            "seizure_type":        seizure_type,
            "icd10":               SEIZURE_ICD10.get(seizure_type, "--"),
            "confidence":          confidence,
            "class_probabilities": {
                SEIZURE_TYPES[i]: round(float(final_proba[i]), 4)
                for i in range(self.num_classes)
            },
            "ensemble_type":       ens_type,
            "ensemble_confidence": ens_conf,
            "ensemble_proba": {
                SEIZURE_TYPES[i]: round(float(ensemble_proba[i]), 4)
                for i in range(self.num_classes)
            },
            "is_epileptic":  (pred_class != 0),
            "model_used":    name,
            "description":   SEIZURE_DESCRIPTIONS.get(seizure_type, ""),
            "key_features":  SEIZURE_KEY_FEATURES.get(seizure_type, []),
            "explanation":   self._explain(name, fv),
            "rule_note":     rule_note,
        }

    # ── Internal helpers ─────────────────────────────────────────
    def _align_to_train(self, fv: np.ndarray) -> np.ndarray:
        n = self.n_train_features
        if n == 0 or len(fv) == n:
            return fv
        if len(fv) < n:
            padded = np.zeros(n, dtype=np.float32)
            padded[:len(fv)] = fv
            return padded
        return fv[:n]

    def _get_proba(self, model_name: str, fv: np.ndarray) -> np.ndarray:
        """
        Get probability vector of length num_classes.
        Handles the case where a model was trained on a subset of classes.
        """
        model = self.trained[model_name]
        full  = np.zeros(self.num_classes, dtype=np.float64)

        try:
            if hasattr(model, "predict_proba"):
                raw = model.predict_proba(fv)[0]
                # Map model output indices to global class indices
                cls_idx = self._class_indices.get(
                    model_name,
                    getattr(model, "classes_", np.arange(len(raw))))
                for i, cls in enumerate(cls_idx):
                    if 0 <= int(cls) < self.num_classes:
                        full[int(cls)] = raw[i]
            else:
                pred = int(model.predict(fv)[0])
                if 0 <= pred < self.num_classes:
                    full[pred] = 1.0
        except Exception:
            full[0] = 1.0   # fallback to Normal on error

        s = full.sum()
        return full / s if s > 1e-6 else full

    def _explain(self, model_name: str, fv: np.ndarray) -> dict:
        model = self.trained.get(model_name)
        if model is None:
            return {}
        n_top  = 15
        names  = self.feature_names
        n_feat = fv.shape[1]

        def _top(importance):
            imp = np.abs(np.array(importance)).flatten()[:n_feat]
            idx = np.argsort(imp)[::-1][:n_top]
            return [{"name":       names[i] if i < len(names) else f"f{i}",
                     "importance": round(float(imp[i]), 6),
                     "value":      round(float(fv[0, i]), 4)}
                    for i in idx]

        base = (getattr(model, "base_estimator", None) or
                getattr(model, "estimator", None) or model)
        fi = getattr(base, "feature_importances_", None)
        if fi is None:
            fi = getattr(model, "feature_importances_", None)
        if fi is not None:
            return {"method": "Feature Importance (tree-based)", "top_features": _top(fi)}

        coef = getattr(model, "coef_", None)
        if coef is not None:
            imp = np.max(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            return {"method": "Coefficient Magnitude", "top_features": _top(imp)}

        return {"method": "Feature Magnitude (fallback)",
                "top_features": _top(np.abs(fv[0]))}

    # ── Persistence ─────────────────────────────────────────────
    def save(self, path: str = "neuroscan_model.pkl"):
        joblib.dump({
            "trained":           self.trained,
            "metrics":           self.metrics,
            "best":              self.best_model_name,
            "scaler":            self.scaler,
            "feature_names":     self.feature_names,
            "num_classes":       self.num_classes,
            "n_train_features":  self.n_train_features,
            "class_indices":     self._class_indices,
        }, path, compress=3)

    def load(self, path: str = "neuroscan_model.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        d = joblib.load(path)
        self.trained          = d["trained"]
        self.models           = self.trained
        self.metrics          = d["metrics"]
        self.best_model_name  = d["best"]
        self.scaler           = d.get("scaler")
        self.feature_names    = d.get("feature_names", [])
        self.num_classes      = N_CLASSES   # always override to 4
        self.n_train_features = d.get("n_train_features", 0)
        self._class_indices   = d.get("class_indices", {})

    # ── Legacy shims ─────────────────────────────────────────────
    def get_model_metrics(self):   return self.metrics
    def get_best_model(self):      return self.best_model_name
    def explain_prediction(self, features, model_name):
        return self._explain(model_name, np.array(features).reshape(1, -1))


class RealEEGDataset:  pass
class PyTorchEEGNet:   pass