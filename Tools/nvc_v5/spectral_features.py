"""Causal pressure and raw-EUS spectral features used by V3.2.

All functions take an explicit decision index/time and only read samples at or
before that time.  Missing/short history is reported to the caller rather than
being imputed or filled from the future.
"""
from __future__ import annotations

from typing import Iterable
import numpy as np
from scipy.signal import detrend, stft

from . import config as C


def _periodogram(values: np.ndarray, fs: float):
    x = np.asarray(values, dtype=float)
    if x.size < 4 or not np.isfinite(x).all():
        return np.array([]), np.array([])
    x = detrend(x - np.mean(x), type="linear")
    x = x * np.hanning(x.size)
    p = np.abs(np.fft.rfft(x)) ** 2 / max(float(x.size * fs), C.EPSILON)
    return np.fft.rfftfreq(x.size, 1.0 / fs), p


def _band_power(x: np.ndarray, fs: float, low: float, high: float) -> float:
    f, p = _periodogram(x, fs)
    if f.size == 0:
        return np.nan
    mask = (f >= float(low) - 1e-12) & (f <= float(high) + 1e-12)
    return float(np.trapz(p[mask], f[mask])) if mask.any() else np.nan


def _safe_log_ratio(current: float, baseline: float) -> float:
    if not np.isfinite(current) or not np.isfinite(baseline):
        return np.nan
    return float(np.log((current + C.EPSILON) / (baseline + C.EPSILON)))


def causal_pressure_spectral_features(delta: np.ndarray, index: int, event: dict,
                                      fs: float = C.DP_FS_HZ) -> tuple[dict, str]:
    """Compute the preregistered M1 pressure spectral ratios."""
    n = int(round(C.PRESSURE_SPEC_WINDOW_S * fs))
    base_n = int(round(C.BASELINE_WINDOW_S * fs))
    idx = int(index)
    onset = int(event.get("start_index", event.get("confirm_index", idx)))
    cur_start = idx - n + 1
    base_end = min(idx, onset)  # onset is exclusive; never cross current event
    base_start = base_end - base_n
    x = np.asarray(delta, dtype=float)
    if idx >= len(x) or cur_start < 0 or base_start < 0 or base_end <= base_start:
        return {}, "PRESSURE_SPECTRAL_BASELINE_INSUFFICIENT"
    current = x[cur_start:idx + 1]
    baseline = x[base_start:base_end]
    if current.size != n or baseline.size != base_n or not np.isfinite(current).all() or not np.isfinite(baseline).all():
        return {}, "PRESSURE_SPECTRAL_BASELINE_INSUFFICIENT"
    bands = ((0.2, 0.6), (0.2, 20.0))
    cur = [_band_power(current, fs, *b) for b in bands]
    windows = np.array([baseline[j:j + n] for j in range(0, base_n, n)], dtype=float)
    refs = [float(np.nanmedian([_band_power(w, fs, *b) for w in windows])) for b in bands]
    out = {
        "pressure_power_0p2_0p6": cur[0], "pressure_auc_0p2_20": cur[1],
        "pressure_power_0p2_0p6_rel": _safe_log_ratio(cur[0], refs[0]),
        "pressure_auc_0p2_20_rel": _safe_log_ratio(cur[1], refs[1]),
        "pressure_spec_window_start_s": float(cur_start / fs),
        "pressure_spec_window_end_s": float(idx / fs),
        "pressure_spec_baseline_start_s": float(base_start / fs),
        "pressure_spec_baseline_end_s": float(base_end / fs),
    }
    if not all(np.isfinite(out[k]) for k in C.PRESSURE_SPECTRAL_FEATURES):
        return {}, "PRESSURE_SPECTRAL_BASELINE_INSUFFICIENT"
    return out, ""


def stft_frequency_names(high_hz: float = C.COMMON_EUS_HIGH_HZ) -> tuple[str, ...]:
    high = int(np.floor(float(high_hz) / C.EUS_BIN_HZ) * C.EUS_BIN_HZ)
    return tuple(f"eus_stft_bin_{int(lo)}_{int(lo + C.EUS_BIN_HZ)}"
                 for lo in np.arange(C.EUS_STFT_LOW_HZ, high, C.EUS_BIN_HZ))


def _stft_vector(values: np.ndarray, fs: float, high_hz: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float)
    if x.size < max(8, int(round(C.EUS_STFT_WINDOW_S * fs))) or not np.isfinite(x).all():
        return np.array([]), np.array([])
    nperseg = int(round(C.EUS_STFT_WINDOW_S * fs))
    f, _, z = stft(x - np.mean(x), fs=fs, window="hann", nperseg=nperseg,
                   noverlap=nperseg // 2, detrend="linear", boundary=None,
                   padded=False, scaling="psd")
    p = np.log10(np.maximum(np.abs(z) ** 2, C.EPSILON))
    mask = (f >= C.EUS_STFT_LOW_HZ - 1e-9) & (f <= float(high_hz) + 1e-9)
    if not mask.any() or p.shape[1] == 0:
        return np.array([]), np.array([])
    return f[mask], np.nanmedian(p[mask], axis=1)


def causal_raw_eus_stft_features(cycle: dict, decision_time_s: float, event: dict,
                                  high_hz: float = C.COMMON_EUS_HIGH_HZ) -> tuple[dict, str]:
    """Return 4-Hz binned relative log PSD of native EUS using causal windows."""
    raw = np.asarray(cycle.get("eus_raw_native", []), dtype=float)
    times = np.asarray(cycle.get("t_eus_abs_native", []), dtype=float)
    fs = float(cycle.get("eus_fs_native", np.nan))
    if raw.size == 0 or times.size != raw.size or not np.isfinite(fs) or fs <= 0:
        return {}, "RAW_EUS_UNAVAILABLE"
    if not np.isfinite(decision_time_s):
        return {}, "EUS_INVALID"
    onset_idx = int(event.get("start_index", event.get("confirm_index", -1)))
    ptime = np.asarray(cycle.get("t_abs_s", []), dtype=float)
    onset_time = float(ptime[onset_idx]) if 0 <= onset_idx < ptime.size else np.nan
    if not np.isfinite(onset_time):
        return {}, "EUS_SPECTRAL_BASELINE_INSUFFICIENT"
    cur_a, cur_b = float(decision_time_s - 2.0), float(decision_time_s)
    base_a, base_b = float(onset_time - C.BASELINE_WINDOW_S), float(onset_time)
    ci = np.flatnonzero((times >= cur_a - 1e-10) & (times <= cur_b + 1e-10))
    bi = np.flatnonzero((times >= base_a - 1e-10) & (times < base_b - 1e-10))
    need_cur = int(round(2.0 * fs))
    need_base = int(round(C.BASELINE_WINDOW_S * fs))
    if ci.size < need_cur * 0.95 or bi.size < need_base * 0.95:
        return {}, "EUS_HISTORY_INSUFFICIENT" if ci.size < need_cur * .95 else "EUS_SPECTRAL_BASELINE_INSUFFICIENT"
    cf, cv = _stft_vector(raw[ci], fs, high_hz)
    bf, bv = _stft_vector(raw[bi], fs, high_hz)
    if cf.size == 0 or bf.size == 0:
        return {}, "EUS_INVALID"
    names = stft_frequency_names(high_hz)
    out = {}
    for i, name in enumerate(names):
        lo = C.EUS_STFT_LOW_HZ + i * C.EUS_BIN_HZ
        hi = lo + C.EUS_BIN_HZ
        c = float(np.median(cv[(cf >= lo) & (cf < hi)])) if np.any((cf >= lo) & (cf < hi)) else np.nan
        b = float(np.median(bv[(bf >= lo) & (bf < hi)])) if np.any((bf >= lo) & (bf < hi)) else np.nan
        out[name] = c - b if np.isfinite(c) and np.isfinite(b) else np.nan
    if not out or not np.isfinite(np.asarray(list(out.values()), dtype=float)).all():
        return {}, "EUS_INVALID"
    out.update({"eus_spec_window_start_s": cur_a, "eus_spec_window_end_s": cur_b,
                "eus_spec_baseline_start_s": base_a, "eus_spec_baseline_end_s": base_b})
    return out, ""


def causal_eus_compact_bands(cycle: dict, decision_time_s: float, event: dict,
                             bands: Iterable[tuple[float, float]], high_hz: float) -> tuple[dict, str]:
    """Compact raw-EUS band powers; uses the same causal 25 s baseline."""
    raw = np.asarray(cycle.get("eus_raw_native", []), dtype=float)
    times = np.asarray(cycle.get("t_eus_abs_native", []), dtype=float)
    fs = float(cycle.get("eus_fs_native", np.nan))
    ptime = np.asarray(cycle.get("t_abs_s", []), dtype=float)
    onset_idx = int(event.get("start_index", event.get("confirm_index", -1)))
    if raw.size == 0 or raw.size != times.size or ptime.size == 0 or not np.isfinite(fs):
        return {}, "RAW_EUS_UNAVAILABLE"
    onset_time = float(ptime[onset_idx]) if 0 <= onset_idx < ptime.size else np.nan
    if not np.isfinite(onset_time):
        return {}, "EUS_SPECTRAL_BASELINE_INSUFFICIENT"
    ci = np.flatnonzero((times <= decision_time_s + 1e-10) & (times >= decision_time_s - 2.0 - 1e-10))
    bi = np.flatnonzero((times < onset_time - 1e-10) & (times >= onset_time - C.BASELINE_WINDOW_S - 1e-10))
    if ci.size < 0.95 * 2.0 * fs or bi.size < 0.95 * C.BASELINE_WINDOW_S * fs:
        return {}, "EUS_HISTORY_INSUFFICIENT" if ci.size < .95 * 2.0 * fs else "EUS_SPECTRAL_BASELINE_INSUFFICIENT"
    out = {}
    for lo, hi in bands:
        if hi > high_hz + 1e-9:
            continue
        cur = _band_power(raw[ci], fs, lo, hi); base = _band_power(raw[bi], fs, lo, hi)
        out[f"eus_relative_log_bandpower_{int(lo)}_{int(hi)}"] = _safe_log_ratio(cur, base)
    if not out or not np.isfinite(np.asarray(list(out.values()), dtype=float)).all():
        return {}, "EUS_INVALID"
    return out, ""
