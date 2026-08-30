"""V3-local frozen feature schema and helpers.

This module contains only the small set of feature functions that V3 inherited
scientifically from the earlier exploratory work.  Keeping them here makes V3
executable without importing any pre-V3 model package.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from Tools.dsd_feature_extraction import config as C
from Tools.dsd_feature_extraction.features import robust_location_scale


DECISION_DELAY_S = 0.5
RANDOM_STATE = 42
TARGET_LABELS = ("NVC_CORE", "PREVOID_PROGRESSIVE")
LABEL_TO_TARGET = {"NVC_CORE": 1, "PREVOID_PROGRESSIVE": 0}

P_FEATURES = (
    "delta_p_current_norm",
    "delta_p_peak_so_far_norm",
    "pressure_slope_0p5s_norm",
    "pressure_slope_change_norm",
    "positive_dpdt_fraction_1s",
    "auc_growth_rate_norm",
)
EUS_FEATURES = (
    "eus_relative_tonic_occupancy",
    "eus_relative_envelope_slope",
    "eus_dpdt_coupling_2s",
)
SPECTRAL_FEATURES = ("relative_pressure_power_0p2_0p6",)
MODEL_FEATURES = {
    "P": P_FEATURES,
    "PE": P_FEATURES + EUS_FEATURES,
    "PEF": P_FEATURES + EUS_FEATURES + SPECTRAL_FEATURES,
}
SUBJECTS_338 = ("STxF26", "STxF27", "STxF29")
SUBJECTS_164 = ("STxF31", "STxF33", "STxF34", "STxF35", "STxF37")
SUBJECTS = SUBJECTS_338 + SUBJECTS_164
C0_FEATURES = tuple(C.PRESSURE_FEATURES)

FORBIDDEN_EXACT = {
    "subject", "animal", "animal_id", "cycle", "cycle_id", "event_id", "event_uid",
    "label", "label_name", "teacher_label", "true_label", "urine", "urine_result",
    "void_confirmed", "future_duration", "future_peak", "final_peak", "future_recovery",
    "manual_class", "quality_conclusion", "detector_fallback", "prevoid_nvc_outcome",
}
FORBIDDEN_FRAGMENTS = ("sparc164", "dataset164", "164_information")


def _as_bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).casefold() in {"true", "1", "yes"}


def assert_feature_schema_safe(feature_names: Sequence[str]) -> None:
    offenders = []
    for name in feature_names:
        lowered = str(name).casefold()
        if lowered in FORBIDDEN_EXACT or any(token in lowered for token in FORBIDDEN_FRAGMENTS):
            offenders.append(str(name))
    if offenders:
        raise RuntimeError(f"V3 feature schema contains leakage fields: {offenders}")


def _pressure_features(delta: np.ndarray, adaptive: Dict[str, np.ndarray], index: int,
                       event: pd.Series) -> Optional[dict]:
    fs = C.DP_FS_HZ
    one_second = int(round(fs))
    half_second = int(round(0.5 * fs))
    if index < one_second or index >= len(delta):
        return None
    confirm = float(adaptive["adaptive_confirm"][index])
    sigma_dpdt = float(adaptive["sigma_dpdt"][index])
    if not np.isfinite(confirm) or confirm <= 0 or not np.isfinite(sigma_dpdt) or sigma_dpdt <= 0:
        return None
    start_index = max(0, min(index, int(event["start_index"])))
    trough_index = max(0, min(index, int(event["local_trough_index"])))
    history = np.asarray(delta[trough_index:index + 1], dtype=np.float64)
    recent = np.asarray(delta[index - one_second:index + 1], dtype=np.float64)
    if history.size < 2 or recent.size != one_second + 1:
        return None
    if not np.isfinite(history).all() or not np.isfinite(recent).all():
        return None
    dpdt = np.diff(recent) * fs
    slope_1s = float(np.mean(dpdt) / sigma_dpdt)
    slope_0p5s = float(np.mean(dpdt[-half_second:]) / sigma_dpdt)
    auc_delta = np.asarray(delta[start_index:index + 1], dtype=np.float64)
    auc_start = np.asarray(adaptive["adaptive_start"][start_index:index + 1], dtype=np.float64)
    if auc_delta.size < 2 or not np.isfinite(auc_delta).all() or not np.isfinite(auc_start).all():
        return None
    elapsed_s = (auc_delta.size - 1) / fs
    auc = float(np.trapezoid(np.maximum(auc_delta - auc_start, 0.0), dx=1.0 / fs))
    return {
        "delta_p_current_norm": float(delta[index] / confirm),
        "delta_p_peak_so_far_norm": float((np.max(history) - history[0]) / confirm),
        "pressure_slope_0p5s_norm": slope_0p5s,
        "pressure_slope_change_norm": float(slope_0p5s - slope_1s),
        "positive_dpdt_fraction_1s": float(np.mean(dpdt > 0)),
        "auc_growth_rate_norm": float(auc / max(elapsed_s * confirm, np.finfo(float).eps)),
    }


def _eus_features(delta: np.ndarray, eus_env: np.ndarray, eus_valid: np.ndarray,
                  index: int) -> Optional[dict]:
    fs = C.DP_FS_HZ
    current_n = int(round(2.0 * fs))
    baseline_n = int(round(C.BASELINE_WINDOW_S * fs))
    current_start = index + 1 - current_n
    baseline_start = current_start - baseline_n
    if baseline_start < 0:
        return None
    baseline = np.asarray(eus_env[baseline_start:current_start], dtype=np.float64)
    baseline_ok = np.asarray(eus_valid[baseline_start:current_start], dtype=bool) & np.isfinite(baseline)
    current = np.asarray(eus_env[current_start:index + 1], dtype=np.float64)
    current_ok = np.asarray(eus_valid[current_start:index + 1], dtype=bool) & np.isfinite(current)
    if baseline_ok.sum() < int(round(C.EUS_MIN_BASELINE_S * fs)) or current_ok.mean() < 0.95:
        return None
    median, mad = robust_location_scale(baseline[baseline_ok])
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        return None
    z = (current - median) / mad
    valid_indices = np.flatnonzero(current_ok)
    z_valid = z[current_ok]
    slope = float(np.polyfit(valid_indices.astype(float) / fs, z_valid, 1)[0]) if z_valid.size > 1 else np.nan
    pressure = np.asarray(delta[current_start:index + 1], dtype=np.float64)
    aligned = current_ok & np.isfinite(pressure)
    pressure_dpdt = np.r_[0.0, np.diff(pressure) * fs]
    if aligned.sum() > 2 and np.std(z[aligned]) > 0 and np.std(pressure_dpdt[aligned]) > 0:
        coupling = float(np.corrcoef(z[aligned], pressure_dpdt[aligned])[0, 1])
    else:
        coupling = 0.0
    return {
        "eus_relative_tonic_occupancy": float(np.mean(z_valid > C.TONIC_MAD_MULTIPLIER)),
        "eus_relative_envelope_slope": slope,
        "eus_dpdt_coupling_2s": coupling if np.isfinite(coupling) else 0.0,
    }


def _bandpower(values: np.ndarray, fs: float, low: float, high: float) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered * np.hanning(centered.size))) ** 2
    frequencies = np.fft.rfftfreq(centered.size, d=1.0 / fs)
    return float(power[(frequencies >= low) & (frequencies <= high)].sum())


def _low_frequency_feature(cycle: Dict, delta: np.ndarray, index: int) -> tuple[float, str]:
    fs = C.DP_FS_HZ
    spectral_n = int(round(10.0 * fs))
    baseline_n = int(round(C.BASELINE_WINDOW_S * fs))
    current_start = index + 1 - spectral_n
    baseline_start = current_start - baseline_n
    if baseline_start < 0:
        return np.nan, "INSUFFICIENT_10S_PLUS_25S_HISTORY"
    values = np.asarray(delta[baseline_start:index + 1], dtype=np.float64)
    valid = np.asarray(cycle["cmg_valid_100hz"][baseline_start:index + 1], dtype=bool)
    if not np.isfinite(values).all() or valid.mean() < 0.995:
        return np.nan, "PRESSURE_INVALID_IN_SPECTRAL_HISTORY"
    current = np.asarray(delta[current_start:index + 1], dtype=np.float64)
    current_fraction = _bandpower(current, fs, 0.2, 0.6) / max(
        _bandpower(current, fs, 0.2, 20.0), np.finfo(float).eps)
    baseline_fractions = []
    chunk_n = int(round(5.0 * fs))
    for chunk in range(5):
        start = baseline_start + chunk * chunk_n
        segment = np.asarray(delta[start:start + chunk_n], dtype=np.float64)
        baseline_fractions.append(_bandpower(segment, fs, 0.2, 0.6) / max(
            _bandpower(segment, fs, 0.2, 20.0), np.finfo(float).eps))
    reference = float(np.median(baseline_fractions))
    epsilon = np.finfo(float).eps * max(1.0, current_fraction, reference)
    return float(np.log((current_fraction + epsilon) / (reference + epsilon))), ""


def _metric_values(frame: pd.DataFrame) -> dict:
    scored = frame[frame["p_nvc"].notna()].copy()
    y = scored["target"].to_numpy(dtype=int)
    pred = scored["predicted_nvc"].astype(bool).to_numpy()
    result = {
        "n_events": int(len(frame)), "n_scorable": int(len(scored)),
        "n_nvc_scorable": int((y == 1).sum()), "n_prevoid_scorable": int((y == 0).sum()),
        "AUROC": np.nan, "AUPRC": np.nan, "sensitivity": np.nan,
        "specificity": np.nan, "PPV": np.nan, "NPV": np.nan,
        "balanced_accuracy": np.nan, "F1": np.nan,
        "TN": 0, "FP": 0, "FN": 0, "TP": 0,
    }
    if len(scored) == 0:
        return result
    if np.unique(y).size == 2:
        result["AUROC"] = float(roc_auc_score(y, scored["p_nvc"]))
        result["AUPRC"] = float(average_precision_score(y, scored["p_nvc"]))
        result["balanced_accuracy"] = float(balanced_accuracy_score(y, pred))
    result["sensitivity"] = float(recall_score(y, pred, pos_label=1, zero_division=0))
    result["specificity"] = float(recall_score(y, pred, pos_label=0, zero_division=0))
    result["PPV"] = float(precision_score(y, pred, pos_label=1, zero_division=0))
    result["NPV"] = float(precision_score(y, pred, pos_label=0, zero_division=0))
    result["F1"] = float(f1_score(y, pred, pos_label=1, zero_division=0))
    if np.unique(y).size == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        result.update({"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)})
    return result


def expanded_animal_class_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["subject", "target"]).size().to_dict()
    subjects = sorted(frame["subject"].astype(str).unique())
    if not subjects:
        raise ValueError("No training animals")
    class_sets = {
        subject: sorted(target for (animal, target), count in counts.items()
                        if animal == subject and count > 0)
        for subject in subjects
    }
    if any(not targets for targets in class_sets.values()):
        raise ValueError("A training animal has no scorable events")
    return np.asarray([
        1.0 / (len(subjects) * len(class_sets[subject]) * counts[(subject, target)])
        for subject, target in zip(frame["subject"].astype(str), frame["target"])
    ], dtype=np.float64)
