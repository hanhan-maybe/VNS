"""Quality metrics and quicklook plots for SPARC 338 PRE_STIM data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _finite_stats(x):
    finite = np.asarray(x)[np.isfinite(x)]
    if finite.size == 0:
        return {"min": None, "max": None, "median": None, "std": None}
    return {"min": float(np.min(finite)), "max": float(np.max(finite)),
            "median": float(np.median(finite)), "std": float(np.std(finite))}


def _edge_fraction(x):
    if len(x) == 0:
        return 0.0
    values = np.asarray(x)
    return float(max(np.mean(values == np.min(values)), np.mean(values == np.max(values))))


def setup_check(bladder_raw, bladder_fs, eus_raw, eus_fs):
    b = np.asarray(bladder_raw[: int(min(len(bladder_raw), 60 * bladder_fs))], dtype=np.float64)
    e = np.asarray(eus_raw[: int(min(len(eus_raw), 60 * eus_fs))], dtype=np.float64)
    warnings: List[str] = []
    b_finite = b[np.isfinite(b)]
    e_finite = e[np.isfinite(e)]
    pressure_range = float(np.ptp(b_finite)) if b_finite.size else None
    pressure_jump = float(np.max(np.abs(np.diff(b_finite)))) if b_finite.size > 1 else None
    bladder_clip = _edge_fraction(b_finite)
    eus_clip = _edge_fraction(e_finite)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(e)):
        warnings.append("possible_setup_or_artifact_at_start: non-finite samples")
    if pressure_range is not None and pressure_range > 100:
        warnings.append("possible_setup_or_artifact_at_start: pressure range > 100 mmHg in first 60 s")
    if pressure_jump is not None and pressure_jump > 20:
        warnings.append("possible_setup_or_artifact_at_start: pressure single-sample jump > 20 mmHg")
    if bladder_clip > 0.01:
        warnings.append("possible_setup_or_artifact_at_start: repeated bladder ADC value > 1%")
    if eus_clip > 0.01:
        warnings.append("possible_setup_or_artifact_at_start: repeated EUS ADC value > 1%")
    return {
        "setup_warning": bool(warnings), "warnings": warnings,
        "first_60s_pressure_range": pressure_range,
        "first_60s_pressure_max_abs_step": pressure_jump,
        "first_60s_bladder_repeated_value_fraction": bladder_clip,
        "first_60s_eus_repeated_value_fraction": eus_clip,
    }


def build_summary(subject, record_duration_s, first_stim_s, bladder_raw, bladder_fs,
                  eus_raw, eus_fs, time_s, bladder_100, eus_100,
                  leak_event_count, keyboard_event_count, warnings):
    bstats = _finite_stats(bladder_raw)
    estats = _finite_stats(eus_100)
    setup = setup_check(bladder_raw, bladder_fs, eus_raw, eus_fs)
    all_warnings = list(warnings) + setup["warnings"]
    if bstats["min"] is not None and bstats["min"] < -50:
        all_warnings.append("PRE_STIM bladder pressure below -50 mmHg; inspect raw transient/artifact")
    if bstats["max"] is not None and bstats["max"] > 250:
        all_warnings.append("PRE_STIM bladder pressure above 250 mmHg; inspect saturation/artifact")
    duration_100 = len(time_s) / 100.0
    checks = {
        "pre_stim_at_least_5min": first_stim_s >= 300,
        "bladder_present": len(bladder_raw) > 0,
        "eus_present": len(eus_raw) > 0,
        "stim_present": np.isfinite(first_stim_s) and first_stim_s > 0,
        "length_100hz_within_0_01s": abs(duration_100 - first_stim_s) <= 0.01 + 1e-9,
        "time_strictly_increasing": len(time_s) < 2 or bool(np.all(np.diff(time_s) > 0)),
        "no_large_nan_fraction": float(np.mean(~np.isfinite(bladder_raw))) < 0.01 and float(np.mean(~np.isfinite(eus_raw))) < 0.01,
        "pre_stim_excludes_first_stim": len(time_s) == 0 or float(time_s[-1]) < first_stim_s,
        "first_stim_valid": np.isfinite(first_stim_s) and 0 < first_stim_s <= record_duration_s,
    }
    if not all(checks.values()):
        all_warnings.extend([f"QC failed: {k}" for k, ok in checks.items() if not ok])
    return {
        "subject": subject, "record_duration_s": record_duration_s,
        "first_stim_s": first_stim_s, "pre_stim_duration_s": first_stim_s,
        "bladder_fs_original": bladder_fs, "eus_fs_original": eus_fs,
        "bladder_n_samples_raw": int(len(bladder_raw)), "eus_n_samples_raw": int(len(eus_raw)),
        "samples_100Hz": int(len(time_s)),
        "bladder_nan_fraction": float(np.mean(~np.isfinite(bladder_raw))) if len(bladder_raw) else 1.0,
        "eus_nan_fraction": float(np.mean(~np.isfinite(eus_raw))) if len(eus_raw) else 1.0,
        "bladder_min": bstats["min"], "bladder_max": bstats["max"],
        "bladder_median": bstats["median"], "bladder_std": bstats["std"],
        "eus_envelope_min": estats["min"], "eus_envelope_max": estats["max"],
        "eus_envelope_median": estats["median"], "eus_envelope_std": estats["std"],
        "leak_event_count": int(leak_event_count), "keyboard_event_count": int(keyboard_event_count),
        **setup, "checks": checks, "warnings": all_warnings,
    }


def make_quicklooks(out_dir: Path, subject: str, first_stim_s: float, time_s, bladder, eus,
                    urine_mode="NONE", urine_time=None, urine_trace=None,
                    drop_times=None, urine_units="", urine_status=""):
    stride = max(1, len(time_s) // 20000)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True, constrained_layout=True)
    axes[0].plot(time_s[::stride] / 60, bladder[::stride], lw=0.7)
    axes[0].set_ylabel("CMG (mmHg)")
    axes[1].plot(time_s[::stride] / 60, eus[::stride], lw=0.7, color="tab:orange")
    axes[1].set_ylabel("EUS env. (mV)")
    drop_times = np.asarray(drop_times if drop_times is not None else [])
    if urine_mode == "VOLUME" and urine_trace is not None and len(urine_trace):
        n = len(urine_trace); s = max(1, n // 20000)
        initial = float(np.nanmedian(np.asarray(urine_trace)[:max(1, min(n, 1000))]))
        axes[2].plot(np.asarray(urine_time)[::s] / 60, np.asarray(urine_trace)[::s] - initial,
                     lw=.7, color="tab:green")
        axes[2].set_ylabel(f"Relative Volume\n({urine_units or 'raw unit'})")
        axes[2].set_title(f"Urine output / voiding evidence — Volume candidate ({urine_status})")
    elif urine_mode == "DROP_EVENTS" and drop_times.size:
        axes[2].step(np.r_[0, drop_times] / 60, np.arange(drop_times.size + 1), where="post", color="tab:green")
        axes[2].set_ylabel("Cumulative drops")
        axes[2].set_title("Urine output / voiding evidence — urine drop events")
    elif urine_mode == "DROP_UNRESOLVED":
        axes[2].text(.5, .5, "Leak/drop signal — unresolved", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_ylabel("Urine evidence")
        axes[2].set_title("Urine output / voiding evidence")
    elif urine_mode == "VOLUME_REJECTED":
        axes[2].text(.5, .55, "Volume candidate failed CMG correspondence QC", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].text(.5, .4, "No confirmed synchronized urine-output channel", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_ylabel("Urine evidence")
        axes[2].set_title("Urine output / voiding evidence")
    else:
        axes[2].text(.5, .5, "No synchronized urine-output channel available", ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_ylabel("Urine evidence")
        axes[2].set_title("Urine output / voiding evidence")
    axes[2].set_xlabel("Time (min)")
    fig.suptitle(f"{subject} PRE_STIM: {first_stim_s/60:.2f} min; first_stim_s={first_stim_s:.6f}")
    fig.savefig(out_dir / "pre_stim_quicklook.png", dpi=150)
    plt.close(fig)

    start = max(0.0, first_stim_s - 30.0)
    mask = time_s >= start
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time_s[mask], bladder[mask], lw=0.8)
    axes[1].plot(time_s[mask], eus[mask], lw=0.8, color="tab:orange")
    for ax in axes:
        ax.axvline(first_stim_s, color="red", ls="--", label="first_stim_s")
        ax.legend(loc="upper right")
    axes[0].set_ylabel("CMG (mmHg)")
    axes[1].set_ylabel("EUS env. (mV)")
    axes[1].set_xlabel("Time (s)")
    fig.suptitle(f"{subject}: final 30 s before first stimulation")
    fig.savefig(out_dir / "pre_stim_tail_check.png", dpi=150)
    plt.close(fig)
