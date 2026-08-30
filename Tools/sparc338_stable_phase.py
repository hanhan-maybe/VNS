"""Phase A stable-baseline selection using pressure and urine evidence only."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import find_peaks


FS_DISPLAY = 100.0


def detect_contractions(time_s, pressure):
    """Find all pre-stimulation CMG contraction candidates."""
    x = np.asarray(pressure, dtype=np.float64)
    clean = np.nan_to_num(x, nan=float(np.nanmedian(x)))
    prominence = max(3.0, 0.25 * float(np.nanstd(x)))
    peaks, props = find_peaks(clean, distance=int(30 * FS_DISPLAY), prominence=prominence,
                              width=int(FS_DISPLAY))
    rows = []
    for j, peak in enumerate(peaks):
        pre0 = max(0, peak - int(30 * FS_DISPLAY))
        pre1 = max(pre0 + 1, peak - int(15 * FS_DISPLAY))
        baseline = float(np.nanmedian(x[pre0:pre1]))
        delta = float(x[peak] - baseline)
        level = baseline + 0.20 * max(float(props["prominences"][j]), delta)
        left = peak
        while left > max(0, peak - int(40 * FS_DISPLAY)) and x[left] > level:
            left -= 1
        right = peak
        while right < min(len(x) - 1, peak + int(40 * FS_DISPLAY)) and x[right] > level:
            right += 1
        if right <= left:
            continue
        rows.append({
            "peak_index": int(peak), "void_start_s": float(time_s[left]),
            "cmg_peak_s": float(time_s[peak]), "void_end_s": float(time_s[right]),
            "pre_void_baseline_pressure": baseline, "cmg_peak_pressure": float(x[peak]),
            "deltaP": delta, "cmg_prominence": float(props["prominences"][j]),
        })
    return rows


def confirm_with_urine(contractions, urine_mode, drop_times=None,
                       volume_time=None, volume_trace=None, marker_times=None):
    """Require synchronized urine evidence for every confirmed event."""
    confirmed, rejected = [], []
    drops = np.asarray(drop_times if drop_times is not None else [], dtype=float)
    vt = np.asarray(volume_time if volume_time is not None else [], dtype=float)
    vv = np.asarray(volume_trace if volume_trace is not None else [], dtype=float)
    markers = np.asarray(marker_times if marker_times is not None else [], dtype=float)
    used_markers = set()
    vrange = float(np.nanpercentile(vv, 99) - np.nanpercentile(vv, 1)) if vv.size else 0.0
    for contraction in contractions:
        row = dict(contraction)
        start, end = row["void_start_s"], row["void_end_s"]
        if urine_mode == "LEAK":
            local = drops[(drops >= start - 1.0) & (drops <= end + 5.0)]
            if local.size >= 2:
                row.update(urine_output_onset_s=float(local[0]), urine_output_per_cycle=float(local.size),
                           void_output_duration_s=float(local[-1] - local[0]), urine_confirmed=True)
                confirmed.append(row)
            else:
                rejected.append({**row, "exclusion_reason": "URINE_UNCONFIRMED"})
        elif urine_mode in {"VOID_MARKER", "DISCRETE_VOID_EVENT"}:
            candidates = [(abs(float(marker) - float(row["cmg_peak_s"])), i, float(marker))
                          for i, marker in enumerate(markers) if i not in used_markers
                          and start - 1.0 <= marker <= end + 5.0]
            if candidates:
                _, index, marker = min(candidates)
                used_markers.add(index)
                row.update(urine_output_onset_s=marker, urine_output_per_cycle=np.nan,
                           void_output_duration_s=0.0, urine_confirmed=True)
                confirmed.append(row)
            else:
                rejected.append({**row, "exclusion_reason": "URINE_MARKER_UNCONFIRMED"})
        elif urine_mode == "VOLUME" and vv.size:
            pre = vv[(vt >= start - 5.0) & (vt <= start - 1.0)]
            post_mask = (vt >= start - 1.0) & (vt <= end + 5.0)
            post_t, post = vt[post_mask], vv[post_mask]
            base = float(np.nanmedian(pre)) if pre.size else float(post[0]) if post.size else 0.0
            increase = post - base
            max_increase = float(np.nanmax(increase)) if increase.size else 0.0
            threshold = max(0.03, 0.03 * vrange)
            crossing = np.flatnonzero(increase >= max(threshold, 0.10 * max_increase))
            if max_increase >= threshold and crossing.size:
                onset_index = int(crossing[0])
                active = np.flatnonzero(increase >= max(threshold, 0.90 * max_increase))
                output_end = float(post_t[active[0]]) if active.size else float(post_t[-1])
                row.update(urine_output_onset_s=float(post_t[onset_index]),
                           urine_output_per_cycle=max_increase,
                           void_output_duration_s=max(0.0, output_end - float(post_t[onset_index])),
                           urine_confirmed=True)
                confirmed.append(row)
            else:
                rejected.append({**row, "exclusion_reason": "URINE_UNCONFIRMED"})
        else:
            rejected.append({**row, "exclusion_reason": "URINE_EVIDENCE_UNAVAILABLE"})
    return confirmed, rejected


def _settled_end(time_s, pressure, row, next_start_s, first_stim_s):
    """Find return near the pre-void filling level after the main contraction."""
    limit = min(float(next_start_s), float(first_stim_s))
    mask = (time_s >= row["cmg_peak_s"]) & (time_s <= limit)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return row["void_end_s"]
    level = row["pre_void_baseline_pressure"] + max(1.0, 0.10 * row["deltaP"])
    below = np.asarray(pressure[indices] <= level, dtype=bool)
    hold = max(1, int(0.50 * FS_DISPLAY))
    sustained = np.convolve(below.astype(int), np.ones(hold, dtype=int), mode="valid") >= hold
    hits = np.flatnonzero(sustained)
    return float(time_s[indices[hits[0]]]) if hits.size else float(row["void_end_s"])


def _local_cv(values, index, radius=2):
    values = np.asarray(values, dtype=float)
    local = values[max(0, index-radius):min(len(values), index+radius+1)]
    local = local[np.isfinite(local)]
    if local.size < 2:
        return np.nan
    mean = float(np.mean(np.abs(local)))
    return float(np.std(local, ddof=1) / mean) if mean > np.finfo(float).eps else np.nan


def _robust_z(matrix):
    matrix = np.asarray(matrix, dtype=float)
    med = np.nanmedian(matrix, axis=0)
    mad = np.nanmedian(np.abs(matrix - med), axis=0)
    scale = 1.4826 * np.where(mad > 1e-9, mad, np.nan)
    return np.abs(matrix - med) / scale


def _contiguous_runs(cycles):
    runs, current = [], []
    for row in cycles:
        if row["stability_candidate"] == "STABLE_CANDIDATE":
            if current and row["original_cycle_number"] != current[-1]["original_cycle_number"] + 1:
                runs.append(current); current = []
            current.append(row)
        elif current:
            runs.append(current); current = []
    if current:
        runs.append(current)
    return runs


def select_nearest_stable_run(cycles, max_cycles=5):
    """Select only the tail of the nearest contiguous stable run; never backfill."""
    runs = _contiguous_runs(cycles)
    selected_run = runs[-1] if runs else []
    return runs, selected_run, selected_run[-max_cycles:]


def build_stable_baseline(time_s, pressure, first_stim_s, urine_mode, urine_data,
                          compute_nvc_candidate: bool = True,
                          use_urine_quantity_for_stability: bool = True):
    """Return all confirmed cycles, the nearest stable run, and selection audit."""
    contractions = detect_contractions(time_s, pressure)
    confirmed, rejected = confirm_with_urine(contractions, urine_mode, **urine_data)
    confirmed = sorted((row for row in confirmed if row["cmg_peak_s"] < first_stim_s),
                       key=lambda row: row["cmg_peak_s"])
    cycles = []
    for i, source in enumerate(confirmed):
        row = dict(source)
        next_start = confirmed[i+1]["void_start_s"] if i+1 < len(confirmed) else first_stim_s
        cycle_end = _settled_end(time_s, pressure, row, next_start, first_stim_s)
        row.update(original_cycle_number=i+1, original_cycle_id=f"C{i+1:02d}",
                   cycle_start_s=(cycles[-1]["cycle_end_s"] if cycles else np.nan),
                   cycle_end_s=cycle_end, complete_cycle=bool(cycles and cycle_end < first_stim_s),
                   artifact_overlap=False, baseline_id="", reference_baseline_candidate=False,
                   selected_for_dsd_validation=False,
                   exclusion_reason="", selection_reason="")
        if cycles:
            row["ICI_s"] = row["cmg_peak_s"] - cycles[-1]["cmg_peak_s"]
            row["cycle_duration_s"] = cycle_end - row["cycle_start_s"]
            cmask = (time_s >= row["cycle_start_s"]) & (time_s <= cycle_end)
            local = np.asarray(pressure[cmask], dtype=float)
            row["artifact_overlap"] = bool(local.size and (
                np.nanmin(local) < -50 or np.nanmax(local) > 100 or
                (local.size > 1 and np.nanmax(np.abs(np.diff(local))) > 20)))
            filling_mask = ((time_s >= row["cycle_start_s"] + 1.0)
                            & (time_s <= row["void_start_s"] - 1.0))
            filling_pressure = np.asarray(pressure[filling_mask], dtype=float)
            if compute_nvc_candidate and filling_pressure.size >= int(2 * FS_DISPLAY):
                nvc, _ = find_peaks(np.nan_to_num(filling_pressure, nan=float(np.nanmedian(filling_pressure))),
                                    prominence=max(2.0, 0.15*row["deltaP"]),
                                    distance=int(FS_DISPLAY), width=max(1, int(0.2*FS_DISPLAY)))
                row["nvc_count_candidate"] = int(len(nvc))
            elif compute_nvc_candidate:
                row["nvc_count_candidate"] = 0
            else:
                row["nvc_count_candidate"] = None
        else:
            row["ICI_s"] = np.nan; row["cycle_duration_s"] = np.nan
            row["nvc_count_candidate"] = np.nan
        cycles.append(row)

    feature_names = ("ICI_s", "cycle_duration_s", "pre_void_baseline_pressure",
                     "cmg_peak_pressure", "deltaP", "urine_output_per_cycle")
    stability_feature_count = 6 if use_urine_quantity_for_stability else 5
    feature_matrix = np.asarray([[row.get(name, np.nan) for name in feature_names] for row in cycles], float)
    z = _robust_z(feature_matrix) if cycles else np.empty((0, len(feature_names)))
    # Subject-internal robust z scores are unstable when only a handful of
    # complete void intervals are available.  In that setting a physiologically
    # small change can be divided by a near-zero MAD and become a false outlier
    # (for example Dataset164 STxF31).  Preserve the scores for audit, but do not
    # let them veto stability until at least five complete intervals exist.
    complete_cycle_count = sum(bool(row.get("complete_cycle", False)) for row in cycles)
    robust_outlier_qc_enabled = complete_cycle_count >= 5
    for i, row in enumerate(cycles):
        for col, name in enumerate(feature_names):
            values = feature_matrix[:, col]
            row[f"{name}_local_CV"] = _local_cv(values, i)
            row[f"{name}_robust_z"] = float(z[i, col]) if np.isfinite(z[i, col]) else np.nan
        cmg_outliers = sum(np.isfinite(z[i, j]) and z[i, j] > 3.5
                           for j in range(stability_feature_count))
        row["robust_outlier_qc_enabled"] = robust_outlier_qc_enabled
        row["robust_outlier_feature_count"] = int(cmg_outliers)
        if not row["complete_cycle"]:
            row["stability_candidate"] = "TRANSITIONAL"
            row["exclusion_reason"] = "RECORDING_BOUNDARY_NO_PRIOR_COMPLETE_VOID"
        elif row["artifact_overlap"]:
            row["stability_candidate"] = "ARTIFACT"
            row["exclusion_reason"] = "SEVERE_PRESSURE_ARTIFACT_OVERLAP"
        elif robust_outlier_qc_enabled and cmg_outliers:
            row["stability_candidate"] = "TRANSITIONAL"
            row["exclusion_reason"] = "ROBUST_CMG_CYCLE_OUTLIER_QC"
        else:
            row["stability_candidate"] = "STABLE_CANDIDATE"

    runs, selected_run, selected = select_nearest_stable_run(cycles, max_cycles=5)
    for j, row in enumerate(selected, 1):
        row["baseline_id"] = f"B{j}"
        row["reference_baseline_candidate"] = True
        row["selection_reason"] = "REFERENCE_BASELINE_CANDIDATE_IN_NEAREST_STABLE_RUN"
    for row in cycles:
        if row["stability_candidate"] == "STABLE_CANDIDATE" and not row["reference_baseline_candidate"]:
            row["exclusion_reason"] = "EARLIER_THAN_LAST_FIVE_IN_SELECTED_RUN" if row in selected_run else "EARLIER_STABLE_RUN"

    if len(selected) >= 5:
        status = "PASS_5_CYCLES"
    elif len(selected) == 4:
        status = "PASS_4_CYCLES"
    elif len(selected) == 3:
        status = "LIMITED_3_CYCLES"
    else:
        status = "INSUFFICIENT_STABLE_BASELINE"
    last_end = cycles[-1]["cycle_end_s"] if cycles else 0.0
    tail_mask = (time_s >= last_end) & (time_s < first_stim_s)
    tail_t, tail_p = np.asarray(time_s[tail_mask]), np.asarray(pressure[tail_mask])
    incomplete_end = False
    if tail_t.size >= int(10 * FS_DISPLAY):
        late = float(np.nanmedian(tail_p[-int(2 * FS_DISPLAY):]))
        smooth_size = max(1, int(2 * FS_DISPLAY))
        smooth = np.convolve(np.nan_to_num(tail_p, nan=float(np.nanmedian(tail_p))),
                             np.ones(smooth_size)/smooth_size, mode="valid")
        incomplete_end = bool(smooth.size and late - float(np.min(smooth)) >= 1.0)

    first_stable_number = runs[0][0]["original_cycle_number"] if runs else 10**9

    audit = {
        "n_pre_stim_confirmed_voids": len(confirmed), "n_complete_cycles": sum(r["complete_cycle"] for r in cycles),
        "n_stable_candidate_cycles": sum(r["stability_candidate"] == "STABLE_CANDIDATE" for r in cycles),
        "n_contiguous_stable_runs": len(runs),
        "selected_run_start_s": selected_run[0]["cycle_start_s"] if selected_run else np.nan,
        "selected_run_end_s": selected_run[-1]["cycle_end_s"] if selected_run else np.nan,
        "distance_to_first_stim_s": first_stim_s-selected_run[-1]["cycle_end_s"] if selected_run else np.nan,
        "n_selected_baseline_cycles": len(selected), "baseline_selection_status": status,
        "incomplete_pre_stim_end": incomplete_end,
        "n_setup_acclimation_cycles": sum(
            r["original_cycle_number"] < first_stable_number for r in cycles),
        "n_transitional_cycles": sum(r["stability_candidate"] == "TRANSITIONAL" for r in cycles),
        "n_artifact_cycles": sum(r["stability_candidate"] == "ARTIFACT" for r in cycles),
        "rejected_contractions": rejected,
    }
    return cycles, selected_run, selected, audit
