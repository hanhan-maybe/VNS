"""Causal signal processing for SPARC 338 PRE_STIM exports."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

try:
    from sparc338_config import DISPLAY_FS_HZ
except ImportError:  # package import from project root
    from Tools.sparc338_config import DISPLAY_FS_HZ

FS_TARGET = DISPLAY_FS_HZ


def causal_fill_nonfinite(raw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Forward-fill invalid samples without consulting future samples.

    The original array remains untouched in the raw export.  A validity mask and
    gap audit are returned so downstream QC can reject long acquisition gaps.
    """
    x = np.asarray(raw, dtype=np.float64)
    valid = np.isfinite(x)
    cleaned = np.empty_like(x)
    previous = 0.0
    for index, value in enumerate(x):
        if np.isfinite(value):
            previous = float(value)
        cleaned[index] = previous

    invalid = ~valid
    if invalid.size:
        changes = np.diff(np.r_[False, invalid, False].astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        lengths = ends - starts
    else:
        starts = ends = lengths = np.empty(0, dtype=int)
    longest = int(lengths.max()) if lengths.size else 0
    qc = {
        "n_nonfinite_samples": int(invalid.sum()),
        "nonfinite_fraction": float(invalid.mean()) if invalid.size else 0.0,
        "n_nonfinite_runs": int(lengths.size),
        "longest_nonfinite_gap_s": float(longest / fs) if fs > 0 else np.nan,
        "fill_method": "CAUSAL_FORWARD_HOLD_INITIAL_ZERO",
    }
    return cleaned, valid, qc


def group_stim_trains(stim_times_s: Iterable[float], gap_s: float = 1.0):
    times = np.asarray(list(stim_times_s), dtype=np.float64)
    if times.size == 0:
        return []
    times.sort()
    split = np.flatnonzero(np.diff(times) > gap_s) + 1
    groups = np.split(times, split)
    rows = []
    for i, group in enumerate(groups, 1):
        duration = float(group[-1] - group[0])
        intervals = np.diff(group)
        positive_intervals = intervals[intervals > 0]
        rows.append({
            "train_id": f"stim_train_{i:02d}",
            "start_s": float(group[0]),
            "end_s": float(group[-1]),
            "duration_s": duration,
            "pulse_count": int(group.size),
            "mean_frequency_hz": float(np.mean(1.0 / positive_intervals)) if positive_intervals.size else None,
        })
    return rows


def build_phase_segments(record_duration_s: float, trains):
    if not trains:
        return []
    rows = []
    phase_id = 1
    first = trains[0]
    rows.append({"phase_id": phase_id, "phase_type": "PRE_STIM", "start_s": 0.0,
                 "end_s": first["start_s"], "duration_s": first["start_s"], "stim_train_id": ""})
    phase_id += 1
    for i, train in enumerate(trains):
        rows.append({"phase_id": phase_id, "phase_type": "STIM_ON", "start_s": train["start_s"],
                     "end_s": train["end_s"], "duration_s": train["duration_s"],
                     "stim_train_id": train["train_id"]})
        phase_id += 1
        next_start = trains[i + 1]["start_s"] if i + 1 < len(trains) else record_duration_s
        if next_start > train["end_s"]:
            rows.append({"phase_id": phase_id, "phase_type": "POST_STIM_OFF", "start_s": train["end_s"],
                         "end_s": next_start, "duration_s": next_start - train["end_s"], "stim_train_id": ""})
            phase_id += 1
    return rows


def _filter_causal(x: np.ndarray, fs: float, cutoff, kind: str, order: int = 4):
    if x.size == 0:
        return x.astype(np.float64)
    sos = butter(order, cutoff, btype=kind, fs=fs, output="sos")
    zi = sosfilt_zi(sos) * float(x[0])
    y, _ = sosfilt(sos, x.astype(np.float64, copy=False), zi=zi)
    return y


def _integer_stages(factor: int) -> List[int]:
    stages = []
    remaining = int(factor)
    for candidate in (10, 8, 5, 4, 3, 2):
        while remaining % candidate == 0 and remaining > 1:
            stages.append(candidate)
            remaining //= candidate
    if remaining > 1:
        stages.append(remaining)
    return stages


def causal_downsample(x: np.ndarray, fs: float, target_fs: float, passband_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    ratio = fs / target_fs
    rounded = int(round(ratio))
    if abs(ratio - rounded) < 1e-9:
        y = x.astype(np.float64, copy=False)
        current_fs = float(fs)
        for factor in _integer_stages(rounded):
            next_fs = current_fs / factor
            cutoff = min(passband_hz, 0.4 * next_fs)
            y = _filter_causal(y, current_fs, cutoff, "lowpass")
            y = y[::factor]
            current_fs = next_fs
        t = np.arange(y.size, dtype=np.float64) / target_fs
        return y.astype(np.float32), t

    # Non-integer sampling (e.g. 1/(21*5 us)): causal low-pass followed by
    # latest-available-sample selection. No future input sample is accessed.
    y = _filter_causal(x, fs, min(passband_hz, 0.4 * target_fs), "lowpass")
    duration = x.size / fs
    n_out = int(np.ceil(duration * target_fs - 1e-12))
    t = np.arange(n_out, dtype=np.float64) / target_fs
    indices = np.floor(t * fs + 1e-12).astype(np.int64)
    indices = np.minimum(indices, x.size - 1)
    return y[indices].astype(np.float32), t


def preprocess_bladder(raw: np.ndarray, fs: float, return_qc: bool = False):
    clean, valid, qc = causal_fill_nonfinite(raw, fs)
    result, time_s = causal_downsample(clean, fs, FS_TARGET, passband_hz=40.0)
    if not return_qc:
        return result, time_s
    mask, _ = causal_downsample(valid.astype(np.float64), fs, FS_TARGET, passband_hz=40.0)
    return result, time_s, mask >= 0.999, qc


def preprocess_eus(raw: np.ndarray, fs: float, return_qc: bool = False):
    clean, valid, qc = causal_fill_nonfinite(raw, fs)
    band = _filter_causal(clean, fs, [50.0, 500.0], "bandpass")
    envelope = _filter_causal(np.abs(band), fs, 20.0, "lowpass")
    envelope = np.maximum(envelope, 0.0)
    result, time_s = causal_downsample(envelope, fs, FS_TARGET, passband_hz=20.0)
    result = np.maximum(result, 0.0)
    if not return_qc:
        return result, time_s
    mask, _ = causal_downsample(valid.astype(np.float64), fs, FS_TARGET, passband_hz=20.0)
    return result, time_s, mask >= 0.999, qc


def _causal_sample_to_time(values, source_start_s: float, target_time_s: np.ndarray):
    values = np.asarray(values)
    source_time = float(source_start_s) + np.arange(values.size, dtype=np.float64) / FS_TARGET
    indices = np.searchsorted(source_time, target_time_s, side="right") - 1
    valid = (indices >= 0) & (indices < values.size)
    indices = np.clip(indices, 0, max(0, values.size - 1))
    sampled = values[indices] if values.size else np.empty(target_time_s.size, dtype=values.dtype)
    return sampled, valid


def align_100hz(bladder, eus, first_stim_s: float,
                bladder_start_s: float = 0.0, eus_start_s: float = 0.0):
    """Align two 100 Hz causal streams on their absolute time axis."""
    start = max(float(bladder_start_s), float(eus_start_s))
    expected = int(np.ceil((float(first_stim_s) - start) * FS_TARGET - 1e-12))
    if expected <= 0:
        return np.empty(0), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    t = start + np.arange(expected, dtype=np.float64) / FS_TARGET
    strict = t < float(first_stim_s)
    bladder_out, bladder_ok = _causal_sample_to_time(bladder, bladder_start_s, t)
    eus_out, eus_ok = _causal_sample_to_time(eus, eus_start_s, t)
    strict &= bladder_ok & eus_ok
    return t[strict], bladder_out[strict], eus_out[strict]


def resample_valid_100hz(valid, source_start_s: float, target_time_s: np.ndarray):
    sampled, ok = _causal_sample_to_time(np.asarray(valid, dtype=np.uint8), source_start_s, target_time_s)
    return (sampled.astype(bool) & ok)
