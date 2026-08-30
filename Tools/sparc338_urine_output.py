"""Urine-output auxiliary signal parsing and QC; never creates VOID/NVC labels."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

try:
    from sparc338_preprocessing import FS_TARGET, causal_downsample, causal_fill_nonfinite
    from sparc338_common import write_csv_atomic
except ImportError:  # Package import for tests and programmatic use.
    from Tools.sparc338_preprocessing import FS_TARGET, causal_downsample, causal_fill_nonfinite
    from Tools.sparc338_common import write_csv_atomic
try:
    from sparc338_config import SUBJECT_REGISTRY
except ImportError:
    from Tools.sparc338_config import SUBJECT_REGISTRY


# Dataset-specific manual review decisions recorded after inspecting five local
# CMG/urine-signal windows per Volume animal. Metrics remain in the QC output.
VOLUME_VISUAL_REVIEW = {
    subject: (row["urine_review_status"], row["review_note"])
    for subject, row in SUBJECT_REGISTRY.items()
    if row["urine_source"] in {"CONTINUOUS_WEIGHT", "NOT_URINE_OUTPUT"}
}

LEAK_VISUAL_REVIEW = {
    subject: row["review_note"]
    for subject, row in SUBJECT_REGISTRY.items()
    if row["urine_source"] in {"LEAK_BUTTON_EVENT", "UNRESOLVED"}
}


def candidate_contractions(time_s, bladder_100, n_candidates: int = 5):
    x = np.asarray(bladder_100, dtype=np.float64)
    if x.size == 0:
        return []
    finite = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    prominence = max(2.0, 0.35 * float(np.nanstd(finite)))
    peaks, props = find_peaks(finite, distance=int(30 * FS_TARGET), prominence=prominence)
    if peaks.size < n_candidates:
        peaks, props = find_peaks(finite, distance=int(15 * FS_TARGET), prominence=max(1.0, prominence / 2))
    if peaks.size == 0:
        return []
    ranking = np.argsort(props["prominences"])[::-1][:n_candidates]
    selected = sorted((int(peaks[i]), float(props["prominences"][i])) for i in ranking)
    return [{"candidate_id": i + 1, "index": idx, "peak_s": float(time_s[idx]), "cmg_prominence": prom}
            for i, (idx, prom) in enumerate(selected)]


def volume_display(raw, fs: float):
    clean, _, _ = causal_fill_nonfinite(raw, fs)
    display, time_s = causal_downsample(clean, fs, FS_TARGET, passband_hz=5.0)
    derivative = np.r_[0.0, np.diff(display).astype(np.float64) * FS_TARGET].astype(np.float32)
    return time_s, display, derivative


def assess_volume_correspondence(time_s, volume_100, candidates):
    x = np.asarray(volume_100, dtype=np.float64)
    if x.size == 0 or not candidates:
        return 0, 0.0, "UNRESOLVED", [], "No candidate contractions or Volume data"
    ten = int(10 * FS_TARGET)
    # Background 10-s changes away from candidate contractions establish an adaptive noise floor.
    protected = np.zeros(x.size, dtype=bool)
    for c in candidates:
        i = c["index"]
        protected[max(0, i - 2 * ten):min(x.size, i + 2 * ten)] = True
    starts = np.arange(0, max(0, x.size - ten), int(FS_TARGET), dtype=int)
    starts = starts[(starts + ten < x.size) & ~protected[starts] & ~protected[np.minimum(starts + ten, x.size - 1)]]
    background = np.abs(x[starts + ten] - x[starts]) if starts.size else np.array([0.0])
    overall_range = float(np.nanpercentile(x, 99) - np.nanpercentile(x, 1))
    threshold = max(float(np.nanpercentile(background, 95)), 0.02 * overall_range, 1e-6)
    metrics = []
    for c in candidates:
        i = c["index"]
        pre = x[max(0, i - ten):max(1, i - int(2 * FS_TARGET))]
        post = x[i:min(x.size, i + ten)]
        baseline = float(np.nanmedian(pre)) if pre.size else float(x[i])
        signed = float(post[np.nanargmax(np.abs(post - baseline))] - baseline) if post.size else 0.0
        metrics.append({**c, "volume_change": signed, "abs_volume_change": abs(signed),
                        "adaptive_change_threshold": threshold, "corresponding_change": abs(signed) >= threshold})
    count = sum(m["corresponding_change"] for m in metrics)
    fraction = count / len(metrics)
    if fraction >= 0.8:
        classification = "CONFIRMED_URINE_OUTPUT"
    elif fraction >= 0.6:
        classification = "LIKELY_URINE_OUTPUT"
    elif fraction <= 0.2:
        classification = "NOT_URINE_OUTPUT"
    else:
        classification = "UNRESOLVED"
    notes = (f"Adaptive QC: candidate post-peak excursion compared with non-candidate 10-s change; "
             f"threshold={threshold:.6g} {count}/{len(metrics)} correspond. Requires visual review.")
    return count, fraction, classification, metrics, notes


def parse_drop_button(raw, fs: float, start_s: float = 0.0):
    x = np.asarray(raw, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.empty(0), "UNRESOLVED", "No finite Leak/drop samples"
    baseline = float(np.nanmedian(finite))
    upper = float(np.nanpercentile(finite, 99))
    noise = float(np.nanmedian(np.abs(finite - baseline))) * 1.4826
    separation = upper - baseline
    # Adaptive bimodality check; this does not assign VOID labels.
    if separation <= max(20 * noise, 0.25 * float(np.ptp(finite))):
        return np.empty(0), "UNRESOLVED", "No separated high button/drop state detected"
    threshold = baseline + 0.5 * separation
    high = x > threshold
    rises = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    if rises.size:
        keep = np.r_[True, np.diff(rises) >= int(max(1, 0.1 * fs))]
        rises = rises[keep]
    # SonPy waveform reads may begin at a non-zero absolute channel origin.
    times = float(start_s) + rises.astype(np.float64) / fs
    status = "PASS" if times.size else "UNRESOLVED"
    notes = f"Adaptive button-state threshold={threshold:.6g} raw units; {times.size} rising edges"
    return times, status, notes


def _as_keyboard_rows(keyboard_events):
    """Normalize marker rows without depending on a particular CSV schema."""
    rows = []
    for event in keyboard_events or []:
        if isinstance(event, dict):
            try:
                t = float(event.get("time_s", np.nan))
            except (TypeError, ValueError):
                t = np.nan
            rows.append({"time_s": t, "code1": event.get("code1", event.get("value", "")),
                         "code2": event.get("code2", ""), "code3": event.get("code3", ""),
                         "code4": event.get("code4", ""), "text": str(event.get("text", event.get("value", "")))})
        else:
            rows.append({"time_s": float(event), "code1": "", "code2": "", "code3": "",
                         "code4": "", "text": ""})
    return [row for row in rows if np.isfinite(row["time_s"])]


def _persistent_transitions(time_s, raw):
    """Find persistent level changes on a compact absolute-time representation."""
    t = np.asarray(time_s, dtype=np.float64)
    x = np.asarray(raw, dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(x)
    t, x = t[finite], x[finite]
    if t.size < 3:
        return t, x, [], {"plateau_fraction": 0.0, "nonzero_difference_fraction": 0.0,
                          "unique_value_ratio": 0.0, "intra_plateau_noise": np.nan}
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    # Median block reduction rejects the high-rate ADC quantisation noise while
    # retaining absolute transition timing.  It does not imply a weight unit.
    block = max(1, int(round(0.01 / max(np.median(np.diff(t)), 1e-9))))
    n = (len(x) // block) * block
    if n >= block:
        xb = np.nanmedian(x[:n].reshape(-1, block), axis=1)
        tb = np.nanmedian(t[:n].reshape(-1, block), axis=1)
    else:
        xb, tb = x, t
    d = np.diff(xb)
    mad = 1.4826 * float(np.nanmedian(np.abs(d - np.nanmedian(d)))) if d.size else 0.0
    dynamic = float(np.nanpercentile(xb, 99) - np.nanpercentile(xb, 1)) if xb.size else 0.0
    # Seed from the local long-plateau difference distribution.  A multiple of
    # the full signal span would make small valid F35 updates disappear after
    # a startup/reset excursion, so span is deliberately not the primary
    # threshold here.
    step_threshold = max(2.0 * mad, 0.03)
    plateau_tol = max(3.0 * mad, step_threshold * 0.1, 1e-6)
    large = np.flatnonzero(np.abs(d) >= step_threshold)
    groups = []
    for idx in large:
        if not groups or idx - groups[-1][-1] > 20:  # at least 0.2 s apart at 100 Hz
            groups.append([int(idx)])
        else:
            groups[-1].append(int(idx))
    transitions = []
    previous_end = float(tb[0])
    for number, group in enumerate(groups, 1):
        i0, i1 = group[0], group[-1] + 1
        pre = xb[max(0, i0 - 200):i0]
        post = xb[min(len(xb), i1 + 1):min(len(xb), i1 + 201)]
        if pre.size == 0 or post.size == 0:
            continue
        before, after = float(np.nanmedian(pre)), float(np.nanmedian(post))
        signed = after - before
        if abs(signed) < max(1.5 * mad, 0.03):
            continue
        transitions.append({
            "transition_id": f"T{number:03d}",
            "transition_time_s": float(tb[i0]),
            "signed_change_raw": signed,
            "transition_duration_s": max(0.0, float(tb[i1]) - float(tb[i0])),
            "preceding_plateau_duration_s": max(0.0, float(tb[i0]) - previous_end),
            "before_level_raw": before, "after_level_raw": after,
        })
        previous_end = float(tb[i1])
    plateau = float(np.mean(np.abs(d) <= plateau_tol)) if d.size else 0.0
    nonzero = float(np.mean(np.abs(d) > plateau_tol)) if d.size else 0.0
    rounded = np.round(xb, decimals=max(0, int(np.ceil(-np.log10(max(plateau_tol, 1e-9))))))
    intra = float(np.nanmedian(np.abs(d[np.abs(d) <= plateau_tol]))) if np.any(np.abs(d) <= plateau_tol) else np.nan
    features = {"plateau_fraction": plateau, "nonzero_difference_fraction": nonzero,
                "unique_value_ratio": float(np.unique(rounded).size / max(1, rounded.size)),
                "intra_plateau_noise": intra, "transition_count": len(transitions),
                "step_threshold_raw": step_threshold, "compact_sample_count": int(len(xb))}
    return tb, xb, transitions, features


def detect_discrete_volume_episodes(volume_time_s, volume_raw, pressure_time_s,
                                    pressure, first_stim_s):
    """Detect and confirm discrete Volume episodes on the complete pre-stim axis.

    The detector is signal-only: derivative seeds are merged by local plateau
    evidence, then each episode is matched once to an all-CMG peak table.  It
    never sees cycle counts, NVC labels, or model scores.
    """
    compact_t, compact_x, seeds, features = _persistent_transitions(volume_time_s, volume_raw)
    # Merge fragments of one directional update when no stable plateau exists
    # between them.  The eight-second bound is a shared protocol constant, not
    # a subject-specific tuning knob.
    merged = []
    merge_s = 8.0
    for seed in seeds:
        if not merged:
            merged.append(dict(seed, raw_transition_count=1)); continue
        prior = merged[-1]
        gap = float(seed["transition_time_s"] - prior["transition_time_s"])
        same_direction = np.sign(seed["signed_change_raw"]) == np.sign(prior["signed_change_raw"])
        # Search the compact absolute axis for the intervening plateau.
        i0 = int(np.searchsorted(compact_t, prior["transition_time_s"], side="left"))
        i1 = int(np.searchsorted(compact_t, seed["transition_time_s"], side="left"))
        between = compact_x[min(i0 + 1, len(compact_x)):max(i0 + 1, i1)]
        local_noise = float(features.get("intra_plateau_noise", np.nan))
        # A sub-second quiet gap is still part of one slow staircase update;
        # only a sustained plateau blocks the mandated <=8 s merge.
        sample_dt = float(np.nanmedian(np.diff(compact_t))) if len(compact_t) > 1 else 0.01
        stable = bool(between.size and between.size * sample_dt >= 2.0 and
                      np.nanpercentile(np.abs(np.diff(between)), 75)
                      <= max(3.0 * local_noise, 1e-6))
        if gap <= merge_s and same_direction and not stable:
            prior["transition_duration_s"] = max(
                prior["transition_duration_s"],
                seed["transition_time_s"] + seed["transition_duration_s"] - prior["transition_time_s"],
            )
            prior["signed_change_raw"] += seed["signed_change_raw"]
            prior["raw_transition_count"] += 1
        else:
            merged.append(dict(seed, raw_transition_count=1))
    # Recompute persistent before/after levels from the complete absolute axis.
    t = np.asarray(volume_time_s, dtype=float); x = np.asarray(volume_raw, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x); t, x = t[valid], x[valid]
    order = np.argsort(t, kind="stable"); t, x = t[order], x[order]
    cmg_t = np.asarray(pressure_time_s, dtype=float); cmg_x = np.asarray(pressure, dtype=float)
    cmg_valid = np.isfinite(cmg_t) & np.isfinite(cmg_x)
    cmg_t, cmg_x = cmg_t[cmg_valid], cmg_x[cmg_valid]
    cmg_peaks = np.empty(0, dtype=float)
    if cmg_t.size:
        clean = np.nan_to_num(cmg_x, nan=float(np.nanmedian(cmg_x)))
        peaks, _ = find_peaks(clean, distance=30 * 100,
                              prominence=max(3.0, 0.25 * float(np.nanstd(clean))))
        cmg_peaks = cmg_t[peaks]
    used_cmg = set(); episodes = []
    abs_seed_values = [abs(float(row["signed_change_raw"])) for row in merged] or [0.0]
    primary = float(np.nanmedian(abs_seed_values))
    for index, row in enumerate(merged, 1):
        onset = float(row["transition_time_s"])
        i = int(np.searchsorted(t, onset, side="left"))
        pre0 = int(np.searchsorted(t, onset - 5.0, side="left"))
        post1 = int(np.searchsorted(t, onset + 5.0, side="right"))
        pre = x[pre0:i]
        post = x[min(len(x), i + 1):post1]
        before = float(np.nanmedian(pre)) if pre.size else np.nan
        after = float(np.nanmedian(post)) if post.size else np.nan
        net = after - before if np.isfinite(before) and np.isfinite(after) else float(row["signed_change_raw"])
        candidates = [(abs(float(peak) - onset), j, float(peak))
                      for j, peak in enumerate(cmg_peaks) if j not in used_cmg and abs(float(peak) - onset) <= 15.0]
        matched = min(candidates) if candidates else None
        artifact = bool(onset < 60.0 and matched is None and
                        abs(float(row.get("signed_change_raw", net))) > max(3.0 * primary, 0.5))
        if artifact:
            match_status, reason = "ARTIFACT_EXCLUDED", "STARTUP_RESET_ARTIFACT"
        elif matched is None:
            match_status, reason = "UNMATCHED", "NO_ONE_TO_ONE_CMG_MATCH"
        else:
            used_cmg.add(matched[1]); match_status, reason = "MATCHED", ""
        episodes.append({
            "episode_id": f"V{index:03d}", "onset_s": onset,
            "offset_s": onset + float(row.get("transition_duration_s", 0.0)),
            "duration_s": float(row.get("transition_duration_s", 0.0)),
            "before_level_raw": before, "after_level_raw": after,
            "net_change_raw": net, "direction": "UP" if net > 0 else "DOWN" if net < 0 else "FLAT",
            "raw_transition_count": int(row.get("raw_transition_count", 1)),
            "artifact_flag": artifact, "exclusion_reason": reason,
            "matched_cmg_event_id": f"C{matched[1] + 1:03d}" if matched else "",
            "matched_cmg_peak_s": matched[2] if matched else np.nan,
            "match_dt_s": matched[2] - onset if matched else np.nan,
            "match_status": match_status,
        })
    features = dict(features, raw_transition_count=len(seeds), merged_episode_count=len(episodes),
                    artifact_episode_count=sum(bool(e["artifact_flag"]) for e in episodes),
                    matched_void_episode_count=sum(e["match_status"] == "MATCHED" for e in episodes))
    return {"transitions": seeds, "episodes": episodes, "features": features,
            "cmg_peak_count": int(cmg_peaks.size),
            "matched_void_episode_count": int(sum(e["match_status"] == "MATCHED" for e in episodes))}


def _match_once(transitions, key, times, tolerance_s=10.0):
    used = set()
    dt_key = {"nearest_leak_time_s": "nearest_leak_dt_s",
              "nearest_keyboard_time_s": "nearest_keyboard_dt_s",
              "nearest_cmg_peak_s": "nearest_cmg_peak_dt_s"}.get(key, f"{key}_dt_s")
    values = np.asarray(times, dtype=np.float64)
    for row in transitions:
        row[key] = np.nan
        row[dt_key] = np.nan
        if values.size == 0:
            continue
        candidates = [(abs(float(t) - row["transition_time_s"]), i, float(t))
                      for i, t in enumerate(values) if i not in used]
        if not candidates:
            continue
        delta, i, value = min(candidates)
        if delta <= tolerance_s:
            used.add(i); row[key] = value; row[dt_key] = value - row["transition_time_s"]
    return used


def audit_urine_evidence(subject, volume_time_s, volume_raw, channel_metadata,
                         leak_times_s=None, keyboard_events=None,
                         pressure_time_s=None, pressure=None):
    """Audit urine-signal semantics without using cycles, NVC labels, or models."""
    compact_t, compact_x, transitions, features = _persistent_transitions(volume_time_s, volume_raw)
    leak = np.asarray(leak_times_s if leak_times_s is not None else [], dtype=np.float64)
    keyboard = _as_keyboard_rows(keyboard_events)
    _match_once(transitions, "nearest_leak_time_s", leak, tolerance_s=10.0)
    keyboard_times = np.asarray([row["time_s"] for row in keyboard], dtype=np.float64)
    used_keyboard = _match_once(transitions, "nearest_keyboard_time_s", keyboard_times, tolerance_s=10.0)
    for row in transitions:
        k = row.get("nearest_keyboard_time_s")
        marker = next((item for i, item in enumerate(keyboard) if i in used_keyboard and item["time_s"] == k), None)
        if marker:
            row.update(nearest_keyboard_code1=marker["code1"], nearest_keyboard_code2=marker["code2"],
                       nearest_keyboard_code3=marker["code3"], nearest_keyboard_code4=marker["code4"],
                       nearest_keyboard_text=marker["text"])
        else:
            row.update(nearest_keyboard_code1="", nearest_keyboard_code2="", nearest_keyboard_code3="",
                       nearest_keyboard_code4="", nearest_keyboard_text="")
    cmg_peaks = np.empty(0, dtype=np.float64)
    if pressure_time_s is not None and pressure is not None:
        pt, px = np.asarray(pressure_time_s, dtype=float), np.asarray(pressure, dtype=float)
        finite = np.isfinite(pt) & np.isfinite(px)
        if finite.any():
            clean = np.nan_to_num(px[finite], nan=float(np.nanmedian(px[finite])))
            prom = max(3.0, 0.25 * float(np.nanstd(clean)))
            peaks, _ = find_peaks(clean, distance=30 * 100, prominence=prom)
            cmg_peaks = pt[finite][peaks]
    _match_once(transitions, "nearest_cmg_peak_s", cmg_peaks, tolerance_s=15.0)
    for row in transitions:
        row["transition_id"] = row["transition_id"]
    channel_metadata = dict(channel_metadata or {})
    title = str(channel_metadata.get("title", ""))
    units = str(channel_metadata.get("units", ""))
    sync_leak = sum(np.isfinite(row.get("nearest_leak_time_s", np.nan)) for row in transitions)
    sync_keyboard = sum(np.isfinite(row.get("nearest_keyboard_time_s", np.nan)) for row in transitions)
    if features["transition_count"] and features["plateau_fraction"] >= 0.80:
        if sync_keyboard / features["transition_count"] >= 0.50:
            source_type, reason = "VOID_MARKER_EVENT", "Persistent staircase transitions have one-to-one Keyboard correspondence"
        elif sync_leak / features["transition_count"] >= 0.50:
            source_type, reason = "LEAK_BUTTON_EVENT", "Persistent staircase transitions have one-to-one Leak correspondence"
        else:
            source_type, reason = "DISCRETE_STABLE_VOLUME", "Long plateaus and discrete persistent updates; title/units do not prove continuous weight"
    elif features["transition_count"] and features["unique_value_ratio"] > 0.10 and features["plateau_fraction"] < 0.80 and not (sync_leak or sync_keyboard):
        source_type, reason = "CONTINUOUS_WEIGHT", "Waveform has continuously varying levels and no synchronized marker evidence"
    else:
        source_type, reason = "UNRESOLVED", "Evidence is insufficient or internally conflicting for acquisition semantics"
    for row in transitions:
        row.update(subject=subject, channel_title=title, channel_units=units)
    contract = {
        "subject": subject, "evidence_type": source_type, "urine_source_type": source_type,
        "acquisition_semantics": source_type, "physiological_correspondence_status": "AUDITED",
        "reason": reason, "channel_metadata": channel_metadata, "features": features,
        "transition_count": len(transitions), "leak_count": int(leak.size),
        "keyboard_count": len(keyboard), "cmg_peak_count": int(cmg_peaks.size),
        "transition_leak_matches": int(sync_leak), "transition_keyboard_matches": int(sync_keyboard),
        "transition_cmg_matches": int(sum(np.isfinite(row.get("nearest_cmg_peak_s", np.nan)) for row in transitions)),
        "time_origin": "absolute", "cycle_generation_allowed": source_type != "UNRESOLVED",
        "legacy_cycle_results_status": "LEGACY_NOT_AUTHORITATIVE", "nvc_pipeline_not_run": True,
    }
    correspondences = []
    for row in transitions:
        correspondences.append({"subject": subject, **row})
    return {"subject": subject, "source_type": source_type, "reason": reason,
            "features": features, "transitions": transitions, "correspondence": correspondences,
            "contract": contract, "compact_time_s": compact_t, "compact_value": compact_x}


def write_volume_qc_csv(path: Path, row: Dict[str, Any]):
    fields = ["subject", "volume_channel", "type", "units", "sample_rate_hz", "n_candidate_voids",
              "n_corresponding_volume_changes", "correspondence_fraction", "classification", "notes"]
    write_csv_atomic(path, [row], fields)


def make_candidate_plots(out_dir: Path, subject: str, time_s, bladder, eus, candidates,
                         urine_mode: str, urine_time=None, urine_trace=None, urine_derivative=None,
                         drop_times=None, source_units: str = ""):
    check_dir = out_dir / "void_output_check"
    check_dir.mkdir(parents=True, exist_ok=True)
    drop_times = np.asarray(drop_times if drop_times is not None else [])
    for c in candidates:
        peak = c["peak_s"]; mask = (time_s >= peak - 10) & (time_s <= peak + 10)
        rows = 4 if urine_mode.startswith("VOLUME") else 3
        fig, axes = plt.subplots(rows, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
        axes[0].plot(time_s[mask], bladder[mask], lw=.8); axes[0].set_ylabel("CMG (mmHg)")
        axes[1].plot(time_s[mask], eus[mask], lw=.8, color="tab:orange"); axes[1].set_ylabel("EUS env. (mV)")
        if urine_mode.startswith("VOLUME"):
            umask = (urine_time >= peak - 10) & (urine_time <= peak + 10)
            axes[2].plot(urine_time[umask], urine_trace[umask], lw=.8, color="tab:green")
            axes[2].set_ylabel(f"Volume ({source_units or 'raw unit'})")
            axes[3].plot(urine_time[umask], urine_derivative[umask], lw=.7, color="tab:purple")
            axes[3].set_ylabel("dVolume/dt\n(derived)")
        elif urine_mode == "DROP_EVENTS":
            local = drop_times[(drop_times >= peak - 10) & (drop_times <= peak + 10)]
            axes[2].vlines(local, 0, 1, color="tab:green")
            axes[2].set_ylim(0, 1); axes[2].set_ylabel("Drop events")
        else:
            axes[2].text(.5, .5, "Leak/drop signal — unresolved", ha="center", va="center", transform=axes[2].transAxes)
            axes[2].set_ylabel("Urine evidence")
        for ax in axes:
            ax.axvline(peak, color="red", ls="--", lw=.8)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{subject} candidate contraction {c['candidate_id']:02d}; candidate only, no VOID/NVC label")
        fig.savefig(check_dir / f"{subject}_candidate_void_{c['candidate_id']:02d}.png", dpi=150)
        plt.close(fig)
