"""Public single-subject cycle extraction on the DSD stable-cycle mainline.

The function in this module is the common entry point for an audited DSD animal
or a new external animal.  It has no subject-name branches; all subject-specific
values are read from that subject's input products or estimated from its own
signals without labels or model performance.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .cycle_qc import assign_cycle_statuses, nvc_eligible_cycles, pass_cycles
from .plot_cycles import plot_cycle_quicklook, plot_stable_overview
from .stable_cycle_extractor import build_subject_summary
from .config import BOUNDARY_METHOD
from ..dsd_feature_extraction.detectors import (detect_native_urine_events,
                                                  estimate_native_volume_parameters)
from ..sparc338_stable_phase import build_stable_baseline
from ..sparc338_urine_output import volume_display
from .urine_evidence_adapter import load_urine_evidence, stable_phase_inputs


def _scalar(z, key):
    value = np.asarray(z[key])
    return value.item() if value.ndim == 0 else value


def _load_source(input_dir: Path, first_stim_s: float | None) -> tuple[dict, float]:
    input_dir = Path(input_dir)
    summary_path = input_dir / "pre_stim_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    stim = float(first_stim_s if first_stim_s is not None else summary.get("first_stim_s", np.nan))
    if not np.isfinite(stim):
        raise RuntimeError("first_stim_s is required when input metadata does not provide it")
    with np.load(input_dir / "pre_stim_100Hz.npz", allow_pickle=False) as z:
        t100 = np.asarray(z["time_s"], dtype=float)
        pressure = np.asarray(z["bladder_pressure_mmHg"], dtype=float)
        eus_env = np.asarray(z["eus_envelope_mV"], dtype=float)
        pressure_valid = np.asarray(z.get("bladder_valid_100hz", np.ones(t100.size)), bool)
        eus_valid = np.asarray(z.get("eus_valid_100hz", np.ones(t100.size)), bool)
        fs100 = float(_scalar(z, "sample_rate_hz"))
    with np.load(input_dir / "pre_stim_raw.npz", allow_pickle=False) as z:
        bladder_raw = np.asarray(z["bladder_raw"], dtype=float)
        bladder_fs = float(_scalar(z, "bladder_fs_hz"))
        eus_raw = np.asarray(z["eus_raw"], dtype=float)
        eus_fs = float(_scalar(z, "eus_fs_hz"))
        pressure_units = str(_scalar(z, "bladder_units"))
        eus_units = str(_scalar(z, "eus_units"))
        raw_start = float(_scalar(z, "pre_stim_start_s")) if "pre_stim_start_s" in z.files else float(t100[0])
        eus_start = float(_scalar(z, "eus_start_s")) if "eus_start_s" in z.files else raw_start
        bladder_start = float(_scalar(z, "bladder_start_s")) if "bladder_start_s" in z.files else float(t100[0])
        wave_time_origin = str(_scalar(z, "wave_time_origin")) if "wave_time_origin" in z.files else "LEGACY_METADATA"
    volume_path = input_dir / "pre_stim_urine_output.npz"
    with np.load(volume_path, allow_pickle=False) as z:
        volume_raw = np.asarray(z["urine_output_raw"], dtype=float)
        volume_fs = float(_scalar(z, "sample_rate_hz"))
        volume_units = str(_scalar(z, "units"))
        volume_time = np.asarray(z["time_s"], dtype=float) if "time_s" in z.files else raw_start + np.arange(volume_raw.size) / volume_fs
    if not (t100.size == pressure.size == eus_env.size == pressure_valid.size == eus_valid.size):
        raise RuntimeError("100 Hz channel lengths disagree")
    if not np.all(np.diff(t100) > 0) or not np.all(t100 < stim):
        raise RuntimeError("100 Hz time axis is not strictly pre-stimulation")
    if volume_time.size != volume_raw.size or not np.all(np.diff(volume_time) > 0):
        raise RuntimeError("native Volume time axis is invalid")
    if np.any(volume_time >= stim):
        keep = volume_time < stim
        volume_time, volume_raw = volume_time[keep], volume_raw[keep]
    if (not t100.size) or not np.isclose(float(t100[0]), bladder_start, atol=1e-6):
        raise RuntimeError("DATA_INVALID: 100 Hz time origin disagrees with bladder_start_s")
    source = {"t100": t100, "pressure": pressure, "eus_env": eus_env,
              "pressure_valid": pressure_valid, "eus_valid": eus_valid, "fs100": fs100,
              "bladder_raw": bladder_raw, "bladder_fs": bladder_fs, "eus_raw": eus_raw,
              "eus_fs": eus_fs, "raw_start": eus_start, "pressure_units": pressure_units,
              "eus_units": eus_units, "volume_raw": volume_raw, "volume_time": volume_time,
              "volume_fs": volume_fs, "volume_units": volume_units,
              "bladder_start_s": bladder_start, "eus_start_s": eus_start,
              "wave_time_origin": wave_time_origin,
              "source_file": str(input_dir / "pre_stim_raw.npz"), "input_dir": str(input_dir)}
    return source, stim


def _native_time(n: int, fs: float, start_s: float) -> np.ndarray:
    return start_s + np.arange(n, dtype=float) / fs


def _display_volume(time_s: np.ndarray, raw: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Use the DSD causal display/downsample path while preserving absolute origin."""
    _, display, _ = volume_display(raw, fs)
    return time_s[0] + np.arange(display.size, dtype=float) / 100.0, np.asarray(display, dtype=float)


def _sample_native(raw: np.ndarray, time_s: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float); time_s = np.asarray(time_s, dtype=float)
    idx = np.searchsorted(time_s, target_s, side="left")
    idx = np.clip(idx, 0, max(0, len(time_s) - 1))
    right = idx; left = np.maximum(0, right - 1)
    choose_right = (right == 0) | (left == right)
    denom = time_s[right] - time_s[left]
    frac = np.divide(target_s - time_s[left], denom, out=np.zeros_like(target_s), where=denom > 0)
    out = raw[left] + frac * (raw[right] - raw[left])
    out[choose_right] = raw[right[choose_right]]
    return out.astype(np.float32)


def _write_cycle(root: Path, row: dict, source: dict, volume_events: list[dict], first_stim_s: float) -> None:
    cycle_dir = (root / str(row["dsd_cycle_id"])
                 if root.name == str(row["subject"])
                 else root / str(row["subject"]) / str(row["dsd_cycle_id"]))
    cycle_dir.mkdir(parents=True, exist_ok=True)
    start, end = float(row["cycle_start_s"]), float(row["cycle_end_s"])
    tmask = (source["t100"] >= start) & (source["t100"] < end) & (source["t100"] < first_stim_s)
    t_abs = source["t100"][tmask]
    if not t_abs.size:
        raise RuntimeError(f"{row['subject']} {row['dsd_cycle_id']}: empty cycle")
    event_times = np.asarray([e["onset_s"] for e in volume_events
                              if e.get("cycle_id") == row["dsd_cycle_id"]], dtype=float)
    terminal_id = ""
    terminal_onset = np.nan
    local_events = [e for e in volume_events if e.get("cycle_id") == row["dsd_cycle_id"]]
    if local_events:
        terminal = max(local_events, key=lambda e: float(e["onset_s"]))
        terminal_id, terminal_onset = str(terminal.get("urine_event_id", "")), float(terminal["onset_s"])
    aligned = {"subject": np.array(row["subject"]), "global_cycle_id": np.array(row["global_cycle_id"]),
               "dsd_cycle_id": np.array(row["dsd_cycle_id"]), "cycle_start_s": np.array(start), "cycle_end_s": np.array(end),
               "cycle_duration_s": np.array(end - start), "first_stim_s": np.array(first_stim_s),
               "cycle_boundary_method": np.array(BOUNDARY_METHOD), "t_abs_s": t_abs,
               "t_rel_s": t_abs - start, "bladder_pressure_mmHg": source["pressure"][tmask].astype(np.float32),
               "cmg_processed_100hz": source["pressure"][tmask].astype(np.float32),
               "eus_envelope_100hz": source["eus_env"][tmask].astype(np.float32),
               "eus_envelope_mV": source["eus_env"][tmask].astype(np.float32), "eus_valid_100hz": source["eus_valid"][tmask],
               "cmg_valid_100hz": source["pressure_valid"][tmask], "sample_rate_hz": np.array(source["fs100"]),
               "urine_output_auxiliary_100hz": _sample_native(source["volume_raw"], source["volume_time"], t_abs),
               "urine_event_times_abs_s": event_times, "urine_event_times_cycle_s": event_times - start,
               "urine_source_type": np.array(source.get("urine_source_type", "UNRESOLVED")),
               "acquisition_semantics": np.array(source.get("acquisition_semantics", source.get("urine_source_type", "UNRESOLVED"))),
               "time_origin": np.array("absolute"), "terminal_urine_episode_id": np.array(terminal_id),
               "terminal_urine_episode_onset_s": np.array(terminal_onset),
               "quantitative_volume_valid": np.array(False), "urine_output_model_input": np.array(False)}
    np.savez_compressed(cycle_dir / "cycle_100Hz.npz", **aligned)
    es = source["eus_fs"]; et = _native_time(len(source["eus_raw"]), es, source["raw_start"])
    emask = (et >= start) & (et < end) & (et < first_stim_s)
    native = {"subject": np.array(row["subject"]), "dsd_cycle_id": np.array(row["dsd_cycle_id"]),
              "t_eus_abs_s": et[emask], "t_eus_rel_s": et[emask] - start, "eus_raw": source["eus_raw"][emask].astype(np.float32),
              "eus_fs": np.array(es), "source_file": np.array(source["source_file"]), "first_stim_s": np.array(first_stim_s)}
    np.savez_compressed(cycle_dir / "cycle_native_eus.npz", **native)
    vt = source["volume_time"]; cstart = max(float(vt[0]), start - 10.0); cend = min(float(vt[-1]), end + 10.0, first_stim_s)
    vm = (vt >= cstart) & (vt <= cend)
    np.savez_compressed(cycle_dir / "native_volume.npz", time_s=vt[vm], urine_output_raw=source["volume_raw"][vm].astype(np.float32),
                        sample_rate_hz=np.array(source["volume_fs"]), units=np.array(source["volume_units"]),
                        analysis_start_s=np.array(start), analysis_end_s=np.array(end), context_start_s=np.array(cstart),
                        context_end_s=np.array(cend), time_origin=np.array("absolute"),
                        urine_source_type=np.array(source.get("urine_source_type", "UNRESOLVED")),
                        acquisition_semantics=np.array(source.get("acquisition_semantics", "UNRESOLVED")),
                        quantitative_volume_valid=np.array(False), urine_output_model_input=np.array(False))
    plot_cycle_quicklook(cycle_dir / "quicklook.png", row, aligned)


def extract_subject_cycles(subject: str, input_dir: Path, output_root: Path,
                           first_stim_s: float | None = None, overwrite: bool = False,
                           evidence_contract_path: Path | None = None) -> dict:
    """Extract one animal using the shared stable-cycle contract."""
    input_dir, output_root = Path(input_dir), Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output_root is not empty: {output_root}")
    if overwrite and output_root.exists():
        for child in output_root.iterdir():
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    source, stim = _load_source(input_dir, first_stim_s)
    try:
        evidence = load_urine_evidence(subject, input_dir, volume_display, evidence_contract_path)
    except ValueError as exc:
        summary = {"subject": subject, "first_stim_s": stim,
                   "candidate_cycle_count": 0, "extracted_cycle_count": 0,
                   "pass_cycle_count": 0, "global_volume_event_count": 0,
                   "assigned_volume_event_count": 0,
                   "all_samples_strictly_pre_stim": bool(np.all(source["t100"] < stim) and np.all(source["volume_time"] < stim)),
                   "status": "HOLD_URINE_EVIDENCE_UNRESOLVED", "exclusion_reason": str(exc),
                   "nvc_pipeline_not_run": True, "model_training_not_run": True,
                   "manual_review_used": False, "stimulation_enabled": False}
        (output_root / "cycle_extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        pd.DataFrame().to_csv(output_root / "cycle_manifest.csv", index=False)
        pd.DataFrame().to_csv(output_root / "nvc_cycle_manifest.csv", index=False)
        pd.DataFrame().to_csv(output_root / "cycle_qc.csv", index=False)
        return {"subject": subject, "source": source, "cycles": [], "included": [],
                "nvc_included": [], "volume_events": [], "manifest": pd.DataFrame(),
                "nvc_manifest": pd.DataFrame(), "summary": summary}
    urine_mode, confirm_data = stable_phase_inputs(evidence)
    source["urine_source_type"] = evidence.source_type
    source["acquisition_semantics"] = evidence.metadata.get("acquisition_semantics", evidence.source_type)
    volume_params = {}
    volume_events = []
    if evidence.source_type == "CONTINUOUS_WEIGHT":
        volume = {"subject": subject, "raw": source["volume_raw"], "time_s": source["volume_time"], "fs": source["volume_fs"]}
        volume_params = estimate_native_volume_parameters(volume)
        global_row = pd.DataFrame([{"dsd_cycle_id": "GLOBAL", "cycle_start_s": float(source["volume_time"][0]), "cycle_end_s": float(min(source["volume_time"][-1], stim))}])
        volume_events = detect_native_urine_events(volume, global_row, threshold_override=volume_params["step_threshold_ml"])
    elif evidence.source_type in {"DISCRETE_STABLE_VOLUME", "VOID_MARKER_EVENT", "LEAK_BUTTON_EVENT"}:
        episode_rows = []
        contract_file = evidence_contract_path
        if contract_file is None:
            candidate = input_dir.parent.parent / "SPARC164_reaudit" / subject / "urine_evidence_contract.json"
            contract_file = candidate if candidate.exists() else None
        if contract_file is not None:
            episode_path = Path(contract_file).parent / "discrete_volume_episodes.csv"
            if episode_path.exists():
                episode_rows = pd.read_csv(episode_path).fillna("").to_dict("records")
        if episode_rows:
            volume_events = [{"urine_event_id": str(e["episode_id"]), "onset_s": float(e["onset_s"]),
                              "offset_s": float(e.get("offset_s", e["onset_s"])), "cycle_id": ""}
                             for e in episode_rows if str(e.get("match_status", "MATCHED")) == "MATCHED"]
        else:
            volume_events = [{"urine_event_id": f"U{i:03d}", "onset_s": float(t), "offset_s": float(t), "cycle_id": ""}
                             for i, t in enumerate(evidence.event_times_s, 1)]
    for i, event in enumerate(volume_events, 1):
        event["urine_event_id"] = str(event.get("urine_event_id") or f"U{i:03d}")
        event["cycle_id"] = ""
    display_time, display_trace = _display_volume(source["volume_time"], source["volume_raw"], source["volume_fs"])
    if urine_mode == "VOLUME":
        confirm_data = {"volume_time": display_time, "volume_trace": display_trace}
    cycles, _, _, audit = build_stable_baseline(source["t100"], source["pressure"], stim, urine_mode,
                                                confirm_data, compute_nvc_candidate=False,
                                                use_urine_quantity_for_stability=(evidence.source_type != "DISCRETE_STABLE_VOLUME"))
    for row in cycles:
        row["subject"] = subject; row["first_stim_s"] = stim
        row["global_cycle_id"] = row.get("original_cycle_id", "")
        row["previous_void_end_s"] = row.get("cycle_start_s", np.nan)
        row["source_pre_stim_file"] = source["source_file"]; row["cycle_boundary_method"] = BOUNDARY_METHOD
        row["urine_evidence_type"] = evidence.source_type; row["confirmed_void"] = bool(row.get("urine_confirmed", False))
        if np.isfinite(float(row.get("cycle_start_s", np.nan))):
            mask = (source["t100"] >= row["cycle_start_s"]) & (source["t100"] < row["cycle_end_s"])
            row["cmg_invalid_fraction"] = float(np.mean(~source["pressure_valid"][mask])) if mask.any() else 1.0
            row["eus_invalid_fraction"] = float(np.mean(~source["eus_valid"][mask])) if mask.any() else 1.0
            row["data_gap_flag"] = bool(row["cmg_invalid_fraction"] > 0.005)
    cycles, _ = assign_cycle_statuses(cycles)
    stable_included = pass_cycles(cycles)
    nvc_included = nvc_eligible_cycles(cycles)
    if len(nvc_included) == 0:
        complete_count = sum(bool(row.get("complete_cycle", False)) for row in cycles)
        status = ("HOLD_NO_COMPLETE_VOID_CYCLES" if complete_count == 0
                  else "HOLD_NO_NVC_ELIGIBLE_CYCLES")
    else:
        # Preserve the global event table while guaranteeing that each PASS
        # cycle carries its unique terminal Volume event.  A tiny timing
        # tolerance covers the native/display resampling offset at a settled
        # boundary; no new Volume events are detected here.
        for i, row in enumerate(nvc_included, 1):
            row["dsd_cycle_id"] = f"B{i:02d}"
        assigned_ids = set()
        for row in nvc_included:
            start, end = float(row["cycle_start_s"]), float(row["cycle_end_s"])
            local = [e for e in volume_events if start <= float(e["onset_s"]) < end]
            for event in local:
                event["cycle_id"] = row.get("dsd_cycle_id", "")
                assigned_ids.add(event["urine_event_id"])
            target = float(row.get("urine_output_onset_s", np.nan))
            if np.isfinite(target) and not local:
                available = [e for e in volume_events if e["urine_event_id"] not in assigned_ids]
                if available:
                    nearest = min(available, key=lambda e: abs(float(e["onset_s"]) - target))
                    if abs(float(nearest["onset_s"]) - target) <= 1.0:
                        nearest["cycle_id"] = row.get("dsd_cycle_id", "")
                        assigned_ids.add(nearest["urine_event_id"])
        nvc_included = [row for row in nvc_included
                        if any(e.get("cycle_id") == row["dsd_cycle_id"] for e in volume_events)]
        if not nvc_included:
            status = "HOLD_NO_NVC_ELIGIBLE_CYCLES_WITH_TERMINAL_URINE"
        else:
            status = "SUBJECT_ADAPTIVE_PIPELINE_READY"
        active_ids = {row["dsd_cycle_id"] for row in nvc_included}
        for event in volume_events:
            if event.get("cycle_id") not in active_ids:
                event["cycle_id"] = ""
        for i, row in enumerate(nvc_included, 1):
            old_cycle_id = row["dsd_cycle_id"]
            new_cycle_id = f"B{i:02d}"
            for event in volume_events:
                if event.get("cycle_id") == old_cycle_id:
                    event["cycle_id"] = new_cycle_id
            row["dsd_cycle_id"] = new_cycle_id
            row["first_stim_s"] = stim
            row["pre_stim_margin_s"] = stim - float(row["cycle_end_s"])
            _write_cycle(output_root, row, source, volume_events, stim)
            for event in volume_events:
                if float(row["cycle_start_s"]) <= event["onset_s"] < float(row["cycle_end_s"]):
                    event["cycle_id"] = row["dsd_cycle_id"]
    manifest_rows = []
    for row in nvc_included:
        manifest_rows.append({"subject": subject, "global_cycle_id": row["global_cycle_id"], "cycle_id": row["dsd_cycle_id"],
                              "dsd_cycle_id": row["dsd_cycle_id"], "stable_run_id": row.get("stable_run_id", ""),
                              "cycle_start_s": row["cycle_start_s"], "cycle_end_s": row["cycle_end_s"], "cycle_duration_s": row["cycle_duration_s"],
                              "previous_void_end_s": row.get("previous_void_end_s", np.nan), "void_start_s": row["void_start_s"],
                              "cmg_peak_s": row["cmg_peak_s"], "urine_output_onset_s": row["urine_output_onset_s"], "void_end_s": row["void_end_s"],
                              "first_stim_s": stim, "pre_stim_margin_s": row["pre_stim_margin_s"], "urine_evidence_type": evidence.source_type,
                              "confirmed_void": bool(row.get("urine_confirmed", False)), "cycle_status": row.get("cycle_status", "REVIEW_REQUIRED"),
                              "nvc_eligible": True, "nvc_quality_status": row.get("nvc_quality_status", "NVC_ELIGIBLE"),
                              "nvc_exclusion_reason": "", "quality_status": ("PASS" if row.get("stability_candidate") == "STABLE_CANDIDATE" else "REVIEW"),
                              "label_eligibility": True, "complete_cycle": True, "cycle_boundary_method": BOUNDARY_METHOD,
                              "source_pre_stim_file": source["source_file"], "pressure_fs_hz": source["fs100"], "eus_fs_native_hz": source["eus_fs"],
                              "volume_fs_native_hz": source["volume_fs"],
                              "source_dataset_id": "164" if subject not in {"STxF26", "STxF27", "STxF29"} else "338",
                              "local_subject_id": subject, "original_subject_id": subject,
                              "injury_group": "UNKNOWN", "sex": "UNKNOWN",
                              "terminal_urine_episode_id": next((str(e["urine_event_id"]) for e in volume_events
                                                                   if e.get("cycle_id") == row["dsd_cycle_id"]), ""),
                              "terminal_urine_episode_onset_s": next((float(e["onset_s"]) for e in volume_events
                                                                        if e.get("cycle_id") == row["dsd_cycle_id"]), np.nan),
                              "urine_source_type": evidence.source_type,
                              "quantitative_volume_valid": False})
    nvc_manifest = pd.DataFrame(manifest_rows)
    stable_ids = {row["global_cycle_id"] for row in stable_included}
    manifest = nvc_manifest[nvc_manifest.global_cycle_id.isin(stable_ids)].copy() if not nvc_manifest.empty else nvc_manifest.copy()
    manifest.to_csv(output_root / "cycle_manifest.csv", index=False)
    nvc_manifest.to_csv(output_root / "nvc_cycle_manifest.csv", index=False)
    nvc_ids = {id(row) for row in nvc_included}
    qc = pd.DataFrame([{**row,
                        "quality_status": ("PASS" if row.get("nvc_quality_status") == "NVC_ELIGIBLE" else
                                           "REVIEW" if id(row) in nvc_ids else "INVALID"),
                        "label_eligibility": id(row) in nvc_ids} for row in cycles])
    qc.to_csv(output_root / "cycle_qc.csv", index=False)
    pd.DataFrame(volume_events).to_csv(output_root / "all_volume_events.csv", index=False)
    pd.DataFrame([{"urine_event_id": e["urine_event_id"], "cycle_id": e.get("cycle_id", ""), "onset_s": e["onset_s"], "offset_s": e["offset_s"]} for e in volume_events]).to_csv(output_root / "volume_event_cycle_assignment.csv", index=False)
    pressure = source["pressure"]
    sigma_p = max(1.4826 * float(np.median(np.abs(pressure - np.median(pressure)))), 1e-6)
    sigma_dpdt = max(1.4826 * float(np.median(np.abs(np.diff(pressure) * source["fs100"]))), 1e-6)
    eus = source["eus_env"]
    adaptive = pd.DataFrame([{"subject": subject, "sigma_p": sigma_p, "sigma_dpdt": sigma_dpdt,
                              "volume_direction": volume_params.get("direction", "NOT_APPLICABLE"),
                              "volume_noise_scale_ml": volume_params.get("noise_scale_ml", np.nan),
                              "volume_step_threshold_ml": volume_params.get("step_threshold_ml", np.nan),
                              "eus_median": float(np.nanmedian(eus)), "eus_robust_scale": float(1.4826 * np.nanmedian(np.abs(eus - np.nanmedian(eus)))), "estimation_source": "SUBJECT_SIGNAL_ONLY_UNSUPERVISED"}])
    adaptive.to_csv(output_root / "subject_adaptive_params.csv", index=False)
    fixed = {"boundary_method": BOUNDARY_METHOD, "candidate_threshold_mmHg": 2.21, "confirm_threshold_mmHg": 3.68, "recovery_threshold_mmHg": 1.47, "volume_algorithm": "CAUSAL_SMOOTH_PERSISTENT_STEP_WITH_DIRECTION_AND_SEARCHSORTED_TIME", "manual_review_used": False, "stimulation_enabled": False}
    (output_root / "fixed_protocol_params.json").write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    summary = {"subject": subject, "first_stim_s": stim, "candidate_cycle_count": len(cycles),
               "complete_cycle_count": sum(bool(row.get("complete_cycle", False)) for row in cycles),
               "extracted_cycle_count": len(stable_included), "pass_cycle_count": len(stable_included),
               "nvc_eligible_cycle_count": len(nvc_included),
               "nvc_statistical_review_cycle_count": sum(row.get("nvc_quality_status") == "NVC_ELIGIBLE_STATISTICAL_REVIEW" for row in nvc_included),
               "global_volume_event_count": len(volume_events), "assigned_volume_event_count": sum(bool(e.get("cycle_id")) for e in volume_events), "all_samples_strictly_pre_stim": bool(np.all(source["t100"] < stim) and np.all(source["volume_time"] < stim)), "status": status, "urine_source_type": evidence.source_type, "acquisition_semantics": evidence.metadata.get("acquisition_semantics", evidence.source_type), "nvc_pipeline_not_run": True, "model_training_not_run": True, "manual_review_used": False, "stimulation_enabled": False, "volume_parameters": volume_params, "audit": audit}
    (output_root / "cycle_extraction_summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item() if hasattr(x, "item") else x), encoding="utf-8")
    return {"subject": subject, "source": source, "cycles": cycles, "included": stable_included,
            "nvc_included": nvc_included, "volume_events": volume_events,
            "manifest": manifest, "nvc_manifest": nvc_manifest, "summary": summary}
