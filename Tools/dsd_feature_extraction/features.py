"""Shared causal EUS stream processing and decision-time feature extraction."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, sosfilt

from . import config as C


def causal_eus_envelope_100hz(cycle: Dict) -> Tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(cycle["eus_raw_native"], dtype=np.float64)
    fs = float(cycle["eus_fs_native"])
    if fs <= 2 * C.EUS_BANDPASS_HZ[1]:
        raise RuntimeError(f"EUS fs={fs} is incompatible with frozen bandpass")
    finite = np.isfinite(raw)
    if not finite.all():
        good = np.flatnonzero(finite)
        if good.size < int(fs):
            return np.full(len(cycle["t_abs_s"]), np.nan), np.zeros(len(cycle["t_abs_s"]), dtype=bool)
        raw = raw.copy(); raw[~finite] = np.interp(np.flatnonzero(~finite), good, raw[good])
    sos = butter(C.EUS_FILTER_ORDER, C.EUS_BANDPASS_HZ, btype="bandpass", fs=fs, output="sos")
    band = sosfilt(sos, raw)
    b, a = iirnotch(C.EUS_NOTCH_HZ, C.EUS_NOTCH_Q, fs=fs)
    band = lfilter(b, a, band)
    env_sos = butter(C.EUS_FILTER_ORDER, C.EUS_ENVELOPE_LP_HZ, btype="lowpass", fs=fs, output="sos")
    envelope = np.maximum(sosfilt(env_sos, np.abs(band)), 0.0)
    target_t = np.asarray(cycle["t_abs_s"], dtype=np.float64)
    native_t = np.asarray(cycle["t_eus_abs_native"], dtype=np.float64)
    # Absolute-time bin aggregation to 100 Hz; no future bin is used by a decision.
    edges = np.r_[target_t, target_t[-1] + 1.0 / C.DP_FS_HZ]
    idx = np.searchsorted(native_t, edges, side="left")
    sums = np.r_[0.0, np.cumsum(envelope)]
    counts = np.diff(idx)
    agg = np.full(target_t.size, np.nan)
    ok = counts > 0
    agg[ok] = (sums[idx[1:][ok]] - sums[idx[:-1][ok]]) / counts[ok]
    valid = ok & np.asarray(cycle["eus_valid_100hz"], dtype=bool) & np.isfinite(agg)
    return agg, valid


def robust_location_scale(x: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)) * 1.4826)
    return med, mad


def feature_at_index(cycle: Dict, delta: np.ndarray, eus_env: np.ndarray, eus_valid: np.ndarray,
                     index: int, adaptive: Optional[Dict[str, np.ndarray]] = None,
                     event: Optional[Dict] = None) -> Optional[Dict[str, float]]:
    fs = C.DP_FS_HZ
    w = int(C.FEATURE_WINDOW_S * fs)
    if index < w:
        return None
    sl = slice(index - w + 1, index + 1)
    d = np.asarray(delta[sl], dtype=np.float64)
    e = np.asarray(eus_env[sl], dtype=np.float64)
    ev = np.asarray(eus_valid[sl], dtype=bool)
    if not np.isfinite(d).all() or ev.mean() < 0.95:
        return None
    dpdt = np.r_[0.0, np.diff(d) * fs]
    half = max(1, int(0.5 * fs))
    if adaptive is None:
        confirm = C.CONFIRM_THRESHOLD_MMHG
        start_window = np.full(d.size, C.CANDIDATE_THRESHOLD_MMHG)
        sigma_dpdt = 1.0
    else:
        confirm = float(adaptive["adaptive_confirm"][index])
        start_window = np.asarray(adaptive["adaptive_start"][sl], dtype=np.float64)
        sigma_dpdt = float(adaptive["sigma_dpdt"][index])
    if not np.isfinite(confirm) or confirm <= 0 or not np.isfinite(sigma_dpdt) or sigma_dpdt <= 0:
        return None
    trough_index = int(event["local_trough_index"]) if event is not None and event.get("local_trough_index") is not None else index - w + 1
    trough_index = max(0, min(index, trough_index))
    local = np.asarray(delta[trough_index:index + 1], dtype=np.float64)
    if not np.isfinite(local).all():
        return None
    prominence_to_now = float(np.max(local) - local[0])
    above = np.maximum(d - start_window, 0.0)
    features = {
        "current_delta_p_norm": float(d[-1] / confirm),
        "peak_prominence_norm": float(prominence_to_now / confirm),
        "mean_dpdt_norm": float(np.mean(dpdt[-half:]) / sigma_dpdt),
        "max_dpdt_norm": float(np.max(dpdt) / sigma_dpdt),
        "time_above_adaptive_start_s": float(np.sum(d > start_window) / fs),
        "pressure_auc_norm": float(np.trapz(above, dx=1.0 / fs) / confirm),
    }
    baseline_end = index - w
    baseline_start = max(0, baseline_end - int(C.EUS_LOCAL_BASELINE_S * fs))
    base = eus_env[baseline_start:baseline_end]
    base_valid = eus_valid[baseline_start:baseline_end] & np.isfinite(base)
    if base_valid.sum() < int(C.EUS_MIN_BASELINE_S * fs):
        return None
    med, mad = robust_location_scale(base[base_valid])
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        return None
    z = (e - med) / mad
    t = np.arange(z.size, dtype=np.float64) / fs
    slope = float(np.polyfit(t, z, 1)[0])
    threshold = med + C.TONIC_MAD_MULTIPLIER * mad
    corr = float(np.corrcoef(z, dpdt)[0, 1]) if np.std(z) > 0 and np.std(dpdt) > 0 else 0.0
    features.update({
        "eus_robust_rms": float(np.sqrt(np.mean(z ** 2))),
        "eus_envelope_slope": slope,
        "eus_tonic_occupancy": float(np.mean(e > threshold)),
        "eus_dpdt_correlation": corr if np.isfinite(corr) else 0.0,
        "eus_baseline_median": med,
        "eus_baseline_mad": mad,
        "eus_tonic_change": float(np.mean(e > threshold)),
    })
    return features


def assert_causal_invariance(cycle: Dict, delta: np.ndarray, eus_env: np.ndarray, eus_valid: np.ndarray,
                             index: int, adaptive=None, event=None) -> bool:
    first = feature_at_index(cycle, delta, eus_env, eus_valid, index, adaptive, event)
    if first is None:
        return True
    d2, e2 = delta.copy(), eus_env.copy()
    d2[index + 1:] = 1e6; e2[index + 1:] = -1e6
    adaptive2 = None if adaptive is None else {k: np.asarray(v).copy() for k, v in adaptive.items()}
    if adaptive2 is not None:
        for value in adaptive2.values():
            if value.ndim and len(value) > index + 1 and np.issubdtype(value.dtype, np.number):
                value[index + 1:] = np.nan
    second = feature_at_index(cycle, d2, e2, eus_valid, index, adaptive2, event)
    if first.keys() != second.keys():
        return False
    return all(np.isclose(first[k], second[k], equal_nan=True) for k in first)


def decision_feature_at_index(cycle: Dict, delta: np.ndarray, eus_env: np.ndarray, eus_valid: np.ndarray,
                              adaptive: Dict[str, np.ndarray], index: int, event: Optional[Dict] = None,
                              require_eus: bool = True) -> Optional[Dict[str, float]]:
    """Features available at one decision time; every peak and slope is historical only."""
    fs = C.DP_FS_HZ; half25 = int(C.FEATURE_WINDOW_S * fs); half05 = int(.5 * fs); half025 = int(.25 * fs)
    if index < half05 or not (0 <= index < len(delta)) or not np.isfinite(delta[index]): return None
    conf = float(adaptive["adaptive_confirm"][index]); sigma_d = float(adaptive["sigma_dpdt"][index])
    if not np.isfinite(conf) or not np.isfinite(sigma_d) or conf <= 0 or sigma_d <= 0: return None
    start_i = int(event.get("start_index", max(0, index - half25))) if event else max(0, index - half25)
    trough_i = int(event.get("local_trough_index", start_i)) if event else start_i
    trough_i = min(index, max(0, trough_i)); hist = np.asarray(delta[trough_i:index + 1], dtype=float)
    if hist.size < 2 or not np.isfinite(hist).all(): return None
    dpdt = np.r_[0.0, np.diff(np.asarray(delta[max(0, index - half25 + 1):index + 1], dtype=float)) * fs]
    if not np.isfinite(dpdt).all(): return None
    recent025 = dpdt[-half025:]; recent05 = dpdt[-half05:]; previous025 = dpdt[-2 * half025:-half025]
    if previous025.size < half025: previous025 = np.pad(previous025, (half025 - previous025.size, 0), mode="edge")
    window_delta = np.asarray(delta[max(0, index - half05 + 1):index + 1], dtype=float)
    if not np.isfinite(window_delta).all(): return None
    auc = float(np.trapz(np.maximum(window_delta - float(adaptive["adaptive_start"][index]), 0), dx=1 / fs) / conf)
    feat = {
        "current_delta_p_norm": float(delta[index] / conf),
        "peak_delta_p_to_now_norm": float((np.max(hist) - hist[0]) / conf),
        "mean_dpdt_0p25s_norm": float(np.mean(recent025) / sigma_d),
        "mean_dpdt_0p5s_norm": float(np.mean(recent05) / sigma_d),
        "dpdt_change": float((np.mean(recent025) - np.mean(previous025)) / sigma_d),
        "peak_to_current_drop_norm": float((np.max(hist) - delta[index]) / conf),
        "negative_dpdt_occupancy_0p5s": float(np.mean(recent05 < 0)),
        "pressure_auc_growth_0p5s_norm": auc,
    }
    if not require_eus:
        return feat
    # EUS is normalized only with samples before this decision's 2-s feature window.
    e_end = index - half25; e_start = max(0, e_end - int(C.EUS_LOCAL_BASELINE_S * fs))
    base = np.asarray(eus_env[e_start:e_end], dtype=float); base_ok = np.asarray(eus_valid[e_start:e_end], bool) & np.isfinite(base)
    cur = np.asarray(eus_env[max(0, index - half05 + 1):index + 1], dtype=float)
    cur_ok = np.asarray(eus_valid[max(0, index - half05 + 1):index + 1], bool) & np.isfinite(cur)
    if base_ok.sum() < int(C.EUS_MIN_BASELINE_S * fs) or cur_ok.mean() < .95: return None
    med, mad = robust_location_scale(base[base_ok])
    if not np.isfinite(mad) or mad <= np.finfo(float).eps: return None
    z = (cur[cur_ok] - med) / mad; tt = np.arange(z.size, dtype=float) / fs
    feat["eus_tonic_occupancy"] = float(np.mean(z > C.TONIC_MAD_MULTIPLIER))
    feat["eus_envelope_slope"] = float(np.polyfit(tt, z, 1)[0]) if z.size > 1 else 0.0
    return feat


def _window_bandpowers(values: np.ndarray, fs: float, bands: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    """Deterministic one-sided Hann-periodogram band powers."""
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.mean(values))
    tapered = centered * np.hanning(values.size)
    power = np.abs(np.fft.rfft(tapered)) ** 2
    frequencies = np.fft.rfftfreq(values.size, d=1.0 / fs)
    return tuple(float(power[(frequencies >= low) & (frequencies <= high)].sum())
                 for low, high in bands)


def v2_time_frequency_features(cycle: Dict, delta: np.ndarray, eus_env: np.ndarray,
                               eus_valid: np.ndarray, index: int) -> Optional[Dict[str, float]]:
    """Five preregistered causal V2 features at one decision sample.

    The current five-second window ends at ``index``.  Its baseline is the
    preceding 25 seconds split into five non-overlapping five-second windows;
    no current or future sample enters a baseline statistic.
    """
    fs = C.DP_FS_HZ
    current_n = int(5.0 * fs)
    baseline_n = int(25.0 * fs)
    if index + 1 < current_n + baseline_n:
        return None
    current_start = index + 1 - current_n
    baseline_start = current_start - baseline_n
    pressure_valid = np.asarray(cycle["cmg_valid_100hz"], dtype=bool)
    delta = np.asarray(delta, dtype=np.float64)
    eus_env = np.asarray(eus_env, dtype=np.float64)
    eus_valid = np.asarray(eus_valid, dtype=bool)
    full = slice(baseline_start, index + 1)
    if (not np.isfinite(delta[full]).all() or pressure_valid[full].mean() < 0.995
            or not np.isfinite(eus_env[full]).all() or eus_valid[full].mean() < 0.95):
        return None

    pressure_bands = ((0.2, 20.0), (0.2, 0.6), (5.0, 20.0))
    current_pressure = _window_bandpowers(delta[current_start:index + 1], fs, pressure_bands)
    baseline_pressure = []
    baseline_eus = []
    for window in range(5):
        start = baseline_start + window * current_n
        stop = start + current_n
        baseline_pressure.append(_window_bandpowers(delta[start:stop], fs, pressure_bands))
        baseline_eus.append(_window_bandpowers(eus_env[start:stop], fs, ((0.5, 20.0), (3.0, 9.0))))
    baseline_pressure = np.asarray(baseline_pressure, dtype=np.float64)
    baseline_eus = np.asarray(baseline_eus, dtype=np.float64)
    current_eus = _window_bandpowers(eus_env[current_start:index + 1], fs, ((0.5, 20.0), (3.0, 9.0)))

    def safe_ratio(value: float, baseline_values: np.ndarray) -> float:
        reference = float(np.median(baseline_values))
        epsilon = np.finfo(float).eps * max(1.0, abs(reference), abs(float(value)))
        return float(np.log((float(value) + epsilon) / (reference + epsilon)))

    pressure_total = current_pressure[0]
    pressure_low_fraction = current_pressure[1] / max(pressure_total, np.finfo(float).eps)
    pressure_high_fraction = current_pressure[2] / max(pressure_total, np.finfo(float).eps)
    baseline_total = baseline_pressure[:, 0]
    baseline_low_fraction = baseline_pressure[:, 1] / np.maximum(baseline_total, np.finfo(float).eps)
    baseline_high_fraction = baseline_pressure[:, 2] / np.maximum(baseline_total, np.finfo(float).eps)
    eus_total = current_eus[0]
    eus_burst_fraction = current_eus[1] / max(eus_total, np.finfo(float).eps)
    baseline_eus_fraction = baseline_eus[:, 1] / np.maximum(baseline_eus[:, 0], np.finfo(float).eps)

    one_second = int(fs)
    p_recent = delta[index + 1 - one_second:index + 1]
    e_recent = eus_env[index + 1 - one_second:index + 1]
    dpdt = np.r_[0.0, np.diff(p_recent) * fs]
    corr = float(np.corrcoef(e_recent, dpdt)[0, 1]) if np.std(e_recent) > 0 and np.std(dpdt) > 0 else 0.0
    return {
        "pressure_bandpower_0p2_20_ratio_5s": safe_ratio(pressure_total, baseline_total),
        "pressure_low_band_fraction_0p2_0p6_ratio_5s": safe_ratio(pressure_low_fraction, baseline_low_fraction),
        "pressure_high_band_fraction_5_20_ratio_5s": safe_ratio(pressure_high_fraction, baseline_high_fraction),
        "eus_burst_band_fraction_3_9_ratio_5s": safe_ratio(eus_burst_fraction, baseline_eus_fraction),
        "eus_dpdt_correlation_1s": corr if np.isfinite(corr) else 0.0,
    }
