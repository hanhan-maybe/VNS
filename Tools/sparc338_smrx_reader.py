"""Read-only helpers for CED Spike2 SMRX files used by SPARC dataset 338."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
WAVE_TYPES = {"Adc", "RealWave"}
EVENT_TYPES = {"EventRise", "EventFall", "EventBoth"}
MARKER_TYPES = {"Marker", "TextMark", "RealMark", "AdcMark"}


def type_name(value: Any) -> str:
    return str(value).split(".")[-1]


def actual_sample_rate(time_base_s: float, divide_ticks: int) -> float:
    return 1.0 / (float(time_base_s) * int(divide_ticks))


def channel_time_metadata(f, channel: int, time_base_s: float) -> Dict[str, Any]:
    """Inspect the channel time origin exposed by the installed SonPy build.

    SonPy versions in common use expose waveform reads in file ticks but do not
    expose a separate per-channel start-time method; in that case the SMRX file
    origin (tick zero) is the only reliable origin and is recorded explicitly.
    We never infer a non-zero origin from sample count alone.
    """
    first_time = getattr(f, "FirstTime", None)
    if callable(first_time):
        try:
            value = float(first_time(int(channel), 0, int(f.MaxTime()))) * float(time_base_s)
        except Exception:
            value = np.nan
        if np.isfinite(value):
            max_time_s = np.nan
            max_method = getattr(f, "ChannelMaxTime", None)
            if callable(max_method):
                try:
                    max_time_s = float(max_method(int(channel))) * float(time_base_s)
                except Exception:
                    max_time_s = np.nan
            return {"start_s": value, "channel_max_time_s": max_time_s,
                    "wave_time_origin": "SONPY_FIRST_TIME", "time_axis_reliable": True}
    method_names = (
        "ChannelStartTime", "GetChannelStartTime", "ChannelTimeStart",
        "GetChannelTimeStart", "WaveStartTime", "GetWaveStartTime",
    )
    for name in method_names:
        method = getattr(f, name, None)
        if not callable(method):
            continue
        try:
            value = float(method(int(channel)))
        except Exception:
            continue
        if np.isfinite(value):
            return {"start_s": value, "channel_max_time_s": np.nan,
                    "wave_time_origin": "SONPY_CHANNEL_START_TIME", "time_axis_reliable": True}
    max_time_s = np.nan
    max_method = getattr(f, "ChannelMaxTime", None)
    if callable(max_method):
        try:
            max_time_s = float(max_method(int(channel))) * float(time_base_s)
        except Exception:
            max_time_s = np.nan
    return {"start_s": 0.0, "channel_max_time_s": max_time_s,
            "wave_time_origin": "SMRX_FILE_TICK_ZERO_REQUIRES_END_CHECK", "time_axis_reliable": False}


def validate_wave_time_axis(f, row: Dict[str, Any], n_samples: int,
                            start_s: float, end_s: float, sample_rate_hz: float) -> bool:
    """Verify origin and contiguous pre-stim coverage against SMRX metadata."""
    method = getattr(f, "ChannelMaxTime", None)
    if not callable(method) or n_samples <= 0:
        return False
    try:
        last_s = float(method(int(row["channel"]))) * float(f.GetTimeBase())
    except Exception:
        return False
    expected_count = int(np.ceil((float(end_s) - float(start_s)) * float(sample_rate_hz) - 1e-12))
    if int(n_samples) < max(0, expected_count - 1):
        return False
    expected_last = float(start_s) + (max(0, expected_count - 1)) / float(sample_rate_hz)
    tolerance = max(2.0 / float(sample_rate_hz), 2.0 * float(f.GetTimeBase()))
    return bool(last_s + tolerance >= expected_last)


def open_smrx(path: Path):
    try:
        import sonpy as sp
    except ImportError as exc:
        raise RuntimeError(
            "SonPy is required only for reading .smrx files. Install the same SonPy build "
            "used by the acquisition workstation before running stage 1. "
            f"Original import error: {exc}"
        ) from exc
    f = sp.SonFile(str(path), True)
    if f.GetOpenError() != 0:
        raise RuntimeError(f"Cannot open {path}: SonPy error {f.GetOpenError()}")
    return f


def close_smrx(f) -> None:
    """Close a SonPy file across SonPy versions without hiding read errors."""
    for name in ("Close", "close"):
        method = getattr(f, name, None)
        if callable(method):
            method()
            return


def scan_inventory(f, subject: str) -> List[Dict[str, Any]]:
    tb = float(f.GetTimeBase())
    rows: List[Dict[str, Any]] = []
    for ch in range(f.MaxChannels()):
        typ = f.ChannelType(ch)
        if int(typ) == 0:
            continue
        divide = int(f.ChannelDivide(ch))
        kind = type_name(typ)
        rows.append({
            "subject": subject,
            "channel": ch,
            "type": kind,
            "title": f.GetChannelTitle(ch),
            "units": f.GetChannelUnits(ch),
            "comment": f.GetChannelComment(ch),
            "sample_rate_hz": actual_sample_rate(tb, divide) if kind in WAVE_TYPES else float(f.GetIdealRate(ch)),
            "divide_ticks": divide,
            **(channel_time_metadata(f, ch, tb) if kind in WAVE_TYPES else {
                "start_s": np.nan, "channel_max_time_s": np.nan,
                "wave_time_origin": "EVENT_OR_MARKER_TICKS", "time_axis_reliable": True,
            }),
            "selected_role": "OTHER",
        })
    return rows


def _norm(text: Any) -> str:
    return str(text or "").strip().casefold()


def match_channels(rows: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    warnings: List[str] = []

    def choose(role: str, candidates: Sequence[Tuple[int, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda item: (-item[0], int(item[1]["channel"])))
        best_score, best = ordered[0]
        tied = [r for score, r in ordered if score == best_score]
        if len(tied) > 1:
            warnings.append(
                f"Ambiguous {role} candidates with equal score: {[r['channel'] for r in tied]}"
            )
            return None
        if len(candidates) > 1:
            warnings.append(
                f"Multiple {role} candidates {[r['channel'] for _, r in ordered]}; selected highest-score channel {best['channel']}"
            )
        return best

    bladder_candidates = []
    eus_raw_candidates = []
    eus_filtered_candidates = []
    stim_candidates = []
    leak_candidates = []
    volume_candidates = []
    keyboard_candidates = []
    for row in rows:
        title, units, kind = _norm(row["title"]), _norm(row["units"]), row["type"]
        if kind in WAVE_TYPES:
            score = 0
            if title == "cmg pres": score = 100
            elif title == "cmg": score = 90
            elif "bladder" in title: score = 80
            elif "pressure" in title and "mmhg" in units: score = 70
            if score: bladder_candidates.append((score, row))
            if title == "eus": eus_raw_candidates.append((100 if kind == "Adc" else 90, row))
            elif title == "ceus": eus_filtered_candidates.append((100 if kind == "RealWave" else 80, row))
            if title in {"leak", "leaks"} or title.startswith("leak"):
                leak_candidates.append((100 if title in {"leak", "leaks"} else 80, row))
            if title == "volume":
                volume_candidates.append((100, row))
        if kind in EVENT_TYPES and title == "stim":
            stim_candidates.append((100, row))
        if kind in MARKER_TYPES and title == "keyboard":
            keyboard_candidates.append((100, row))

    selected = {
        "BLADDER": choose("BLADDER", bladder_candidates),
        "EUS_RAW": choose("EUS_RAW", eus_raw_candidates),
        "EUS_FILTERED": choose("EUS_FILTERED", eus_filtered_candidates),
        "STIM": choose("STIM", stim_candidates),
        "LEAK": choose("LEAK", leak_candidates),
        "VOLUME": choose("VOLUME", volume_candidates),
        "KEYBOARD": choose("KEYBOARD", keyboard_candidates),
    }
    for role, row in selected.items():
        if row is not None:
            row["selected_role"] = {
                "LEAK": "URINE_DROP_SIGNAL",
                # A title is a channel label, not evidence that the source is
                # a calibrated continuous weight transducer.  Semantic
                # classification is deferred to the urine-evidence audit.
                "VOLUME": "URINE_SIGNAL_CANDIDATE",
                "KEYBOARD": "KEYBOARD_METADATA",
            }.get(role, role)
    for required in ("BLADDER", "EUS_RAW", "STIM"):
        if selected[required] is None:
            warnings.append(f"Missing required channel: {required}")
    return selected, warnings


def read_event_times(f, channel: int, time_base_s: float) -> np.ndarray:
    ticks = f.ReadEvents(int(channel), 100_000_000, 0, f.MaxTime())
    return np.asarray(ticks, dtype=np.float64) * float(time_base_s)


def _marker_text(mark: Any) -> str:
    """Preserve full TextMark content when the installed SonPy exposes it."""
    for name in ("Text", "text", "Marker", "marker", "Value", "value"):
        value = getattr(mark, name, None)
        if value is None:
            continue
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(value, str):
            return value.rstrip("\x00")
    code = int(mark.Code1)
    return chr(code) if 32 <= code <= 126 else ""


def read_keyboard(f, channel: int, time_base_s: float, end_s: Optional[float] = None):
    end_tick = f.MaxTime() if end_s is None else int(np.ceil(end_s / time_base_s))
    marks = f.ReadMarkers(int(channel), 1_000_000, 0, end_tick)
    times, codes, texts = [], [], []
    for mark in marks:
        code = int(mark.Code1)
        times.append(float(mark.Tick * time_base_s))
        codes.append([code, int(mark.Code2), int(mark.Code3), int(mark.Code4)])
        texts.append(_marker_text(mark))
    width = max(1, max((len(text) for text in texts), default=1))
    return (np.asarray(times, dtype=np.float64), np.asarray(codes, dtype=np.int16),
            np.asarray(texts, dtype=f"U{width}"))


def read_wave_before(f, row: Dict[str, Any], end_s: float, chunk_samples: int = 2_000_000,
                     start_s: float | None = None) -> np.ndarray:
    ch = int(row["channel"])
    divide = int(row["divide_ticks"])
    tb = float(f.GetTimeBase())
    origin_s = float(row.get("start_s", 0.0) if start_s is None else start_s)
    if not np.isfinite(origin_s) or origin_s < 0 or float(end_s) <= origin_s:
        return np.empty(0, dtype=np.float32)
    start_tick = int(np.rint(origin_s / tb))
    end_tick = int(np.ceil(float(end_s) / tb))
    total = int(np.ceil(max(0, end_tick - start_tick) / divide))
    parts = []
    for sample0 in range(0, total, chunk_samples):
        count = min(chunk_samples, total - sample0)
        tick0 = start_tick + sample0 * divide
        tick1 = min(end_tick, tick0 + count * divide)
        values = np.asarray(f.ReadFloats(ch, count, tick0, tick1), dtype=np.float32)
        if values.size:
            parts.append(values)
    if not parts:
        return np.empty(0, dtype=np.float32)
    data = np.concatenate(parts)
    # Enforce [0, end_s): the final requested block may contain one boundary sample.
    n_strict = int(np.ceil((float(end_s) - origin_s) * float(row["sample_rate_hz"])))
    return data[:n_strict]
