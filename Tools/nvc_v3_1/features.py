"""Causal dynamic features used by V3.1."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from Tools.dsd_feature_extraction import config as C
from Tools.dsd_feature_extraction.features import robust_location_scale
from .config import EPSILON, TRAJECTORY_FEATURES


def _slope(values: np.ndarray, fs: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2 or not np.isfinite(values).all():
        return np.nan
    return float(np.polyfit(np.arange(values.size, dtype=float) / fs, values, 1)[0])


def causal_trajectory_features(
    delta: np.ndarray,
    eus_env: np.ndarray,
    eus_valid: np.ndarray,
    index: int,
    event: Dict,
    adaptive: Dict[str, np.ndarray],
    epsilon: float = EPSILON,
) -> Optional[dict]:
    """Return the ten preregistered features using samples at or before index."""
    fs = float(C.DP_FS_HZ)
    n25, n50, n100, n200 = (int(round(v * fs)) for v in (0.25, 0.5, 1.0, 2.0))
    confirm_index = int(event["confirm_index"])
    onset_index = max(0, min(index, int(event.get("start_index", confirm_index))))
    trough_index = max(0, min(index, int(event.get("local_trough_index", onset_index))))
    if index < max(n200, 2) or index <= trough_index:
        return None
    pressure = np.asarray(delta, dtype=np.float64)
    history = pressure[trough_index:index + 1]
    recent = pressure[index - n200:index + 1]
    if not np.isfinite(history).all() or not np.isfinite(recent).all():
        return None
    dpdt = np.diff(recent) * fs
    event_dpdt = np.diff(history) * fs
    if dpdt.size < n200 or event_dpdt.size == 0:
        return None
    recent25, previous25 = dpdt[-n25:], dpdt[-2 * n25:-n25]
    recent50, previous50 = dpdt[-n50:], dpdt[-2 * n50:-n50]
    slope25 = float(np.mean(recent25))
    prev25 = float(np.mean(previous25))
    slope50 = float(np.mean(recent50))
    prev50 = float(np.mean(previous50))
    sigma = float(adaptive["sigma_dpdt"][index])
    confirm = float(adaptive["adaptive_confirm"][index])
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(confirm) or confirm <= 0:
        return None

    baseline_end = onset_index
    baseline_start = max(0, baseline_end - int(round(C.BASELINE_WINDOW_S * fs)))
    baseline = np.asarray(eus_env[baseline_start:baseline_end], dtype=float)
    baseline_ok = np.asarray(eus_valid[baseline_start:baseline_end], dtype=bool) & np.isfinite(baseline)
    current_start = max(0, index - n200 + 1)
    current_eus = np.asarray(eus_env[current_start:index + 1], dtype=float)
    current_ok = np.asarray(eus_valid[current_start:index + 1], dtype=bool) & np.isfinite(current_eus)
    if baseline_ok.sum() < int(round(C.EUS_MIN_BASELINE_S * fs)) or current_ok.mean() < 0.95:
        return None
    med, mad = robust_location_scale(baseline[baseline_ok])
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        return None
    z = (current_eus[current_ok] - med) / mad
    z_slope = _slope(z, fs)
    onset_half = np.asarray(eus_env[onset_index:min(index + 1, onset_index + n50)], dtype=float)
    onset_half = onset_half[np.isfinite(onset_half)]
    onset_level = float(np.median(onset_half)) if onset_half.size else med
    aligned_pressure = pressure[current_start:index + 1]
    aligned_dpdt = np.r_[0.0, np.diff(aligned_pressure) * fs]
    aligned = current_ok & np.isfinite(aligned_dpdt)
    corr = 0.0
    if aligned.sum() > 2:
        ez = (current_eus[aligned] - med) / mad
        if np.std(ez) > 0 and np.std(aligned_dpdt[aligned]) > 0:
            corr = float(np.corrcoef(ez, aligned_dpdt[aligned])[0, 1])
    max_positive = max(float(np.max(event_dpdt)), epsilon)
    result = {
        "slope_change_250ms": (slope25 - prev25) / sigma,
        "slope_change_500ms": (slope50 - prev50) / sigma,
        "slope_ratio_500ms": slope50 / (abs(prev50) + epsilon),
        "pressure_curvature": (slope25 - slope50) / sigma,
        "current_dpdt_to_max_positive_so_far": slope25 / max_positive,
        "peak_to_current_drop_so_far": (float(np.max(history)) - float(history[-1])) / confirm,
        "positive_slope_occupancy": float(np.mean(event_dpdt > 0)),
        "eus_delta_from_event_onset": (float(np.median(current_eus[current_ok])) - onset_level) / mad,
        "eus_envelope_slope_trajectory": z_slope,
        "causal_pressure_eus_corr": corr if np.isfinite(corr) else 0.0,
    }
    return result if all(np.isfinite(result[name]) for name in TRAJECTORY_FEATURES) else None


def assert_trajectory_causal(
    delta: np.ndarray,
    eus_env: np.ndarray,
    eus_valid: np.ndarray,
    index: int,
    event: Dict,
    adaptive: Dict[str, np.ndarray],
) -> bool:
    first = causal_trajectory_features(delta, eus_env, eus_valid, index, event, adaptive)
    d2, e2 = np.asarray(delta).copy(), np.asarray(eus_env).copy()
    d2[index + 1:] = 1e12
    e2[index + 1:] = -1e12
    second = causal_trajectory_features(d2, e2, eus_valid, index, event, adaptive)
    if first is None or second is None:
        return first is None and second is None
    return first.keys() == second.keys() and all(
        np.isclose(first[key], second[key], equal_nan=True) for key in first
    )
