"""Transactional entry point for auditable SPARC338 DSD cycle extraction."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sparc338_common import (
        cleanup_staging, commit_directory, make_staging_directory, read_csv,
        sha256_file, write_csv_atomic, write_json_atomic,
    )
except ImportError:
    from Tools.sparc338_common import (
        cleanup_staging, commit_directory, make_staging_directory, read_csv,
        sha256_file, write_csv_atomic, write_json_atomic,
    )

from .config import (
    BASELINE_ROOT, CANDIDATE_FIELDS, MANIFEST_FIELDS, OUTPUT_ROOT,
    REFERENCE_BASELINE_FIELDS, REFERENCE_STATS_FIELDS, SUBJECT_REGISTRY, SUBJECTS, SUMMARY_FIELDS,
    VALIDATION_ROOT, VOID_FIELDS,
)
from .plot_cycles import plot_cycle_quicklook, plot_stable_overview
from .stable_cycle_extractor import build_subject_summary, cycle_arrays, extract_subject
from .subject_pipeline import extract_subject_cycles
from ..sparc338_urine_output import audit_urine_evidence, detect_discrete_volume_episodes


COMPARISON_FIELDS = (
    "old_void_id", "old_void_time", "new_void_id", "new_void_time",
    "matched", "time_difference_s", "reason",
)


def write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    write_csv_atomic(path, rows, fields)


def _input_files(baseline_root: Path) -> list[Path]:
    files: list[Path] = []
    for subject in SUBJECTS:
        subject_dir = baseline_root / subject
        files.extend(path for path in subject_dir.rglob("*") if path.is_file())
    if VALIDATION_ROOT.exists():
        files.extend(path for path in VALIDATION_ROOT.rglob("*") if path.is_file())
    return sorted(set(path.resolve() for path in files), key=str)


def audit_subject_urine_evidence(subject: str, input_dir: Path, output_root: Path) -> dict:
    """Run the evidence-only semantic audit on a complete PRE_STIM bundle."""
    input_dir, output_root = Path(input_dir), Path(output_root)
    with np.load(input_dir / "pre_stim_urine_output.npz", allow_pickle=False) as z:
        volume_time = np.asarray(z["time_s"], dtype=float)
        volume_raw = np.asarray(z["urine_output_raw"], dtype=float)
    with np.load(input_dir / "pre_stim_100Hz.npz", allow_pickle=False) as z:
        pressure_time = np.asarray(z["time_s"], dtype=float)
        pressure = np.asarray(z["bladder_pressure_mmHg"], dtype=float)
    summary = json.loads((input_dir / "pre_stim_summary.json").read_text(encoding="utf-8")) if (input_dir / "pre_stim_summary.json").exists() else {}
    first_stim_s = float(summary.get("first_stim_s", pressure_time[-1] if pressure_time.size else 0.0))
    inventory = read_csv(input_dir / "channel_inventory.csv") if (input_dir / "channel_inventory.csv").exists() else []
    volume_rows = [row for row in inventory if row.get("selected_role") in {
        "URINE_SIGNAL_CANDIDATE", "URINE_VOLUME_CONTINUOUS_CANDIDATE"
    } or str(row.get("title", "")).casefold() == "volume"]
    channel_metadata = dict(volume_rows[0]) if volume_rows else {
        "title": "", "units": "", "comment": "", "type": "", "sample_rate_hz": ""
    }
    events = read_csv(input_dir / "pre_stim_events.csv") if (input_dir / "pre_stim_events.csv").exists() else []
    leak_times = []
    keyboard = []
    for row in events:
        try:
            row["time_s"] = float(row.get("time_s", "nan"))
        except (TypeError, ValueError):
            continue
        if row.get("event_type") == "LEAK":
            leak_times.append(row["time_s"])
        elif row.get("event_type") == "KEYBOARD":
            keyboard.append(row)
    channel_metadata["semantic_role"] = "URINE_SIGNAL_CANDIDATE"
    channel_metadata["selected_role"] = "URINE_SIGNAL_CANDIDATE"
    # Keep the generic correspondence audit, but use the discrete episode
    # detector as the cycle-anchor authority for Dataset 164.
    audit_result = audit_urine_evidence(subject, volume_time, volume_raw, channel_metadata,
                                        leak_times, keyboard, pressure_time, pressure)
    result = detect_discrete_volume_episodes(volume_time, volume_raw, pressure_time,
                                             pressure, first_stim_s)
    subject_out = output_root / subject
    subject_out.mkdir(parents=True, exist_ok=True)
    matched_count = int(result["matched_void_episode_count"])
    axis_ok = bool(len(volume_time) == len(volume_raw) and
                   (len(volume_time) < 2 or np.all(np.diff(volume_time) > 0)))
    pre_stim_ok = bool(not len(volume_time) or np.all(volume_time < first_stim_s))
    physiology = "PASS" if matched_count >= 2 else "FAIL"
    cycle_allowed = bool(axis_ok and pre_stim_ok and result["features"]["raw_transition_count"] > 0
                         and physiology == "PASS" and matched_count >= 2)
    semantics = [{"subject": subject, "evidence_type": "DISCRETE_STABLE_VOLUME",
                  "physiological_correspondence_status": physiology,
                  "acquisition_semantics": "DISCRETE_STABLE_VOLUME",
                  "reason": "Discrete episodes require one-to-one CMG correspondence; Keyboard is metadata only",
                  **result["features"],
                  "channel_type": channel_metadata.get("type", ""),
                  "channel_title": channel_metadata.get("title", ""),
                  "channel_units": channel_metadata.get("units", ""),
                  "channel_comment": channel_metadata.get("comment", ""),
                  "channel_start_s": channel_metadata.get("start_s", ""),
                  "sample_rate_hz": channel_metadata.get("sample_rate_hz", ""),
                  "legacy_cycle_results_status": "LEGACY_NOT_AUTHORITATIVE"}]
    write_csv(subject_out / "urine_signal_semantics.csv", semantics,
              tuple(semantics[0].keys()))
    transitions = [{"subject": subject, **row} for row in result["transitions"]]
    transition_fields = tuple(sorted({key for row in transitions for key in row})) if transitions else (
        "subject", "transition_id", "transition_time_s", "signed_change_raw",
        "transition_duration_s", "preceding_plateau_duration_s")
    write_csv(subject_out / "discrete_volume_transitions.csv", transitions, transition_fields)
    write_csv(subject_out / "urine_transition_events.csv", transitions, transition_fields)
    episodes = [{"subject": subject, **row} for row in result["episodes"]]
    episode_fields = tuple(sorted({key for row in episodes for key in row})) if episodes else (
        "subject", "episode_id", "onset_s", "offset_s", "duration_s", "before_level_raw",
        "after_level_raw", "net_change_raw", "direction", "raw_transition_count",
        "artifact_flag", "exclusion_reason", "matched_cmg_event_id", "matched_cmg_peak_s",
        "match_dt_s", "match_status")
    write_csv(subject_out / "discrete_volume_episodes.csv", episodes, episode_fields)
    matches = [row for row in episodes if row.get("match_status") == "MATCHED"]
    write_csv(subject_out / "discrete_volume_cmg_matches.csv", matches, episode_fields)
    # Retain the generic marker correspondence artifact for audit consumers.
    write_csv(subject_out / "urine_marker_correspondence.csv", matches, episode_fields)
    contract = {
        "subject": subject, "source_dataset_id": "164", "evidence_type": "DISCRETE_STABLE_VOLUME",
        "urine_source_type": "DISCRETE_STABLE_VOLUME", "acquisition_semantics": "DISCRETE_STABLE_VOLUME",
        "audit_completed": True, "event_detection_status": "PASS" if result["features"]["raw_transition_count"] else "FAIL",
        "physiological_correspondence_status": physiology,
        "raw_transition_count": int(result["features"]["raw_transition_count"]),
        "merged_episode_count": int(result["features"]["merged_episode_count"]),
        "artifact_episode_count": int(result["features"]["artifact_episode_count"]),
        "matched_void_episode_count": matched_count,
        "cycle_anchor_allowed": cycle_allowed, "void_confirmation_allowed": cycle_allowed,
        "quantitative_volume_allowed": False, "quantitative_volume_valid": False,
        "model_feature_allowed": False, "cycle_generation_allowed": cycle_allowed,
        "channel_metadata": channel_metadata, "features": result["features"],
        "cmg_peak_count": int(result["cmg_peak_count"]), "leak_count": int(len(leak_times)),
        "keyboard_count": int(len(keyboard)), "transition_leak_matches": 0,
        "transition_keyboard_matches": 0,
        "transition_cmg_matches": matched_count, "time_origin": "absolute",
        "volume_sample_count": int(len(volume_raw)), "volume_time_strictly_increasing": axis_ok,
        "all_volume_samples_pre_stim": pre_stim_ok,
        "source_input_dir": str(input_dir.resolve()), "audit_output_dir": str(subject_out.resolve()),
        "legacy_cycle_results_status": "LEGACY_NOT_AUTHORITATIVE",
        "nvc_pipeline_not_run": True, "model_training_not_run": True, "stimulation_enabled": False,
    }
    write_json_atomic(subject_out / "urine_evidence_contract.json", contract)
    return contract


def extract_external_cohort_cycles(subjects, input_root: Path, audit_root: Path,
                                   output_root: Path, source_dataset_id: str = "164",
                                   overwrite: bool = False) -> dict:
    """Thin batch wrapper around the shared single-subject cycle entry point."""
    input_root, audit_root, output_root = Path(input_root), Path(audit_root), Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries, manifests, nvc_manifests, qcs, urine_events, assignments = [], [], [], [], [], []
    for subject in subjects:
        contract_path = audit_root / subject / "urine_evidence_contract.json"
        subject_out = output_root / subject
        if overwrite and subject_out.exists():
            shutil.rmtree(subject_out)
        result = extract_subject_cycles(subject, input_root / subject, subject_out,
                                        overwrite=False, evidence_contract_path=contract_path)
        summary = dict(result["summary"])
        summaries.append(summary)
        if not result["manifest"].empty:
            manifests.append(result["manifest"])
        if not result["nvc_manifest"].empty:
            nvc_manifests.append(result["nvc_manifest"])
        for name, sink in (("cycle_qc.csv", qcs), ("all_volume_events.csv", urine_events),
                           ("volume_event_cycle_assignment.csv", assignments)):
            path = subject_out / name
            if path.exists():
                frame = pd.read_csv(path)
                if not frame.empty:
                    frame.insert(0, "subject", subject) if "subject" not in frame.columns else None
                    sink.append(frame)
    def concat_write(frames, name):
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output_root / name, index=False)
        else:
            pd.DataFrame().to_csv(output_root / name, index=False)
    concat_write(manifests, "cycle_manifest.csv")
    concat_write(nvc_manifests, "nvc_cycle_manifest.csv")
    concat_write(qcs, "cycle_qc.csv")
    concat_write(urine_events, "all_volume_events.csv")
    concat_write(assignments, "volume_episode_cycle_assignment.csv")
    pd.DataFrame(summaries).to_csv(output_root / "subject_cycle_summary.csv", index=False)
    confirmed = []
    for subject, summary in zip(subjects, summaries):
        subject_out = output_root / subject
        path = subject_out / "cycle_manifest.csv"
        if path.exists():
            frame = pd.read_csv(path)
            if not frame.empty:
                frame.insert(0, "subject", subject) if "subject" not in frame.columns else None
                confirmed.append(frame)
    concat_write(confirmed, "confirmed_void_episodes.csv")
    (output_root / "pipeline_contract.json").write_text(json.dumps({
        "source_dataset_id": str(source_dataset_id), "subjects": list(subjects),
        "cycle_boundary": "PREVIOUS_SETTLED_VOID_END_TO_CURRENT_SETTLED_VOID_END",
        "stable_cycle_manifest": "cycle_manifest.csv",
        "nvc_eligibility_manifest": "nvc_cycle_manifest.csv",
        "nvc_eligibility_independent_of_sustained_stable_onset": True,
        "urine_source_type": "DISCRETE_STABLE_VOLUME", "quantitative_volume_valid": False,
        "urine_output_model_input": False, "nvc_pipeline_not_run": True,
        "stimulation_enabled": False,
    }, indent=2), encoding="utf-8")
    return {"subjects": list(subjects), "summaries": summaries,
            "cycle_count": int(sum(len(frame) for frame in manifests)),
            "nvc_eligible_cycle_count": int(sum(len(frame) for frame in nvc_manifests))}


def validate_external_cohort(subjects, cycles_root: Path, reference_root: Path,
                              output_root: Path, source_dataset_id: str = "164",
                              overwrite: bool = False) -> dict:
    """Run the existing frozen single-subject validator and aggregate artifacts."""
    cycles_root, reference_root, output_root = Path(cycles_root), Path(reference_root), Path(output_root)
    if overwrite and output_root.exists():
        for child in output_root.iterdir():
            if child.is_file(): child.unlink()
            elif child.is_dir(): shutil.rmtree(child)
    output_root.mkdir(parents=True, exist_ok=True)
    from ..dsd_feature_extraction.subject_nvc_validation import validate_subject_nvc
    summaries = []
    for subject in subjects:
        manifest_path = (cycles_root / "nvc_cycle_manifest.csv"
                         if (cycles_root / "nvc_cycle_manifest.csv").exists()
                         else cycles_root / "cycle_manifest.csv")
        manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        if manifest.empty or "subject" not in manifest.columns or manifest[manifest.subject == subject].empty:
            summaries.append({"subject": subject, "external_validation_only": True,
                              "cycle_count": 0, "valid_cycle_count": 0,
                              "automatic_label_counts": {"NVC_CORE": 0, "PREVOID_PROGRESSIVE": 0,
                                                          "GREY_ZONE": 0, "INVALID": 0},
                              "nvc_cycle_count": 0, "nvc_frequency_per_h": np.nan,
                              "valid_analysis_hours": 0.0, "pressure_event_count": 0,
                              "frozen_model_available": True,
                              "positive_nvc_validation_status": "NOT_EVALUABLE_NO_POSITIVE_EVENTS",
                              "prevoid_negative_validation_status": "NOT_EVALUABLE_NO_CYCLES",
                              "final_status": "HOLD_NO_VALID_CYCLES",
                              "stimulation_enabled": False})
        else:
            summaries.append(validate_subject_nvc(subject, cycles_root, output_root, reference_root,
                                                  output_prefix=f"{subject}_"))
    mapping = {
        "pressure_events.csv": "sparc164_pressure_events.csv",
        "teacher_labels.csv": "sparc164_teacher_labels.csv",
        "nvc_events.csv": "sparc164_nvc_events.csv",
        "event_features.csv": "sparc164_nvc_duration_summary.csv",
        "causal_features.csv": "sparc164_causal_features.csv",
        "frozen_model_scores.csv": "sparc164_frozen_model_scores.csv",
        "frozen_model_metrics.csv": "sparc164_replay_metrics.csv",
    }
    for source_name, target_name in mapping.items():
        frames = []
        for subject in subjects:
            path = output_root / f"{subject}_{source_name}"
            if path.exists():
                frame = pd.read_csv(path)
                if not frame.empty: frames.append(frame)
        if frames: pd.concat(frames, ignore_index=True).to_csv(output_root / target_name, index=False)
    # Replay-event compatibility artifact: teacher/event rows preserve the
    # frozen causal replay timestamps without adding any stimulation action.
    replay_frames = []
    for subject in subjects:
        path = output_root / f"{subject}_pressure_events.csv"
        if path.exists():
            frame = pd.read_csv(path); frame.insert(0, "subject", subject) if "subject" not in frame.columns else None
            replay_frames.append(frame)
    if replay_frames: pd.concat(replay_frames, ignore_index=True).to_csv(output_root / "sparc164_replay_events.csv", index=False)
    pd.DataFrame(summaries).to_csv(output_root / "sparc164_external_subject_metrics.csv", index=False)
    # The frozen bundle has no causal rows in dsd_reference_feature_ranges;
    # report that limitation explicitly rather than fitting a new range.
    pd.DataFrame([{"feature_name": feature, "reference_status": "NOT_AVAILABLE_IN_FROZEN_RANGE_FILE",
                   "external_median": np.nan, "external_iqr": np.nan,
                   "outside_reference_fraction": np.nan}
                  for feature in json.loads((reference_root / "final_model.json").read_text())["feature_order"]
                  ]).to_csv(output_root / "sparc164_feature_range_shift.csv", index=False)
    positive = sum(int(s.get("automatic_label_counts", {}).get("NVC_CORE", 0)) for s in summaries)
    validation_status = "NOT_EVALUABLE_NO_POSITIVE_EVENTS" if positive == 0 else "PARTIALLY_EVALUABLE"
    positive_metrics = [s.get("frozen_model_metrics", {}) for s in summaries
                        if s.get("frozen_model_metrics", {}).get("nvc_event_sensitivity") is not None
                        and s.get("frozen_model_metrics", {}).get("nvc_ppv") is not None]
    f1_values = [2 * float(m["nvc_event_sensitivity"]) * float(m["nvc_ppv"]) /
                 (float(m["nvc_event_sensitivity"]) + float(m["nvc_ppv"]))
                 for m in positive_metrics
                 if float(m["nvc_event_sensitivity"]) + float(m["nvc_ppv"]) > 0]
    final = {"source_dataset_id": str(source_dataset_id), "subjects": list(subjects),
             "external_validation_only": True, "validation_status": validation_status,
             "automatic_nvc_count": positive,
             "macro_f1_positive_animals_only": float(np.mean(f1_values)) if f1_values else None,
             "subject_summaries": summaries, "feature_order_source": str((reference_root / "final_model.json").resolve()),
             "scaler_refit": False, "threshold_refit": False, "decision_delay_refit": False,
             "manual_review_used": False, "stimulation_enabled": False,
             "deployment_status": "HOLD_NO_CAUSAL_SEPARATION"}
    write_json_atomic(output_root / "sparc164_external_validation_summary.json", final)
    return final


def _hash_snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path): sha256_file(path) for path in paths}


def _stxf21_void_comparison(old_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    old_rows = [row for row in old_rows if row.get("subject") == "STxF21"]
    new_rows = [row for row in new_rows if row.get("subject") == "STxF21"]
    new_by_id = {row["void_global_id"]: row for row in new_rows}
    comparison = []
    for old in old_rows:
        new = new_by_id.pop(old["void_global_id"], None)
        difference = (float(new["urine_output_onset_s"]) - float(old["urine_output_onset_s"])
                      if new else np.nan)
        comparison.append({
            "old_void_id": old["void_global_id"],
            "old_void_time": old["urine_output_onset_s"],
            "new_void_id": new["void_global_id"] if new else "",
            "new_void_time": new["urine_output_onset_s"] if new else "",
            "matched": bool(new and np.isclose(difference, 0.0, atol=1e-9)),
            "time_difference_s": difference,
            "reason": ("MATCHED_AUDITED_URINE_EVIDENCE"
                       if new and np.isclose(difference, 0.0, atol=1e-9)
                       else "REMOVED_OR_SHIFTED_AFTER_FULL_RERUN"),
        })
    for new in new_by_id.values():
        comparison.append({
            "old_void_id": "", "old_void_time": "",
            "new_void_id": new["void_global_id"],
            "new_void_time": new["urine_output_onset_s"],
            "matched": False, "time_difference_s": "",
            "reason": "ADDED_AFTER_FULL_RERUN",
        })
    return comparison


def _write_subject(root: Path, result: dict) -> None:
    subject_dir = root / result["subject"]
    subject_dir.mkdir(parents=True, exist_ok=True)
    for row in result["included"]:
        cycle_dir = subject_dir / row["dsd_cycle_id"]
        cycle_dir.mkdir(parents=True, exist_ok=True)
        aligned, native = cycle_arrays(result["cache"], row)
        np.savez_compressed(cycle_dir / "cycle_100Hz.npz", **aligned)
        np.savez(cycle_dir / "cycle_native_eus.npz", **native)
        plot_cycle_quicklook(cycle_dir / "quicklook.png", row, aligned)
    plot_stable_overview(subject_dir / "stable_cycle_overview.png", result)


def _write_report(path: Path, summaries: list[dict], candidates: list[dict]) -> None:
    total = sum(int(row["n_stable_cycles_extracted"]) for row in summaries)
    reference_total = sum(int(row["n_reference_baseline_cycles"]) for row in summaries)
    lines = [
        "# SPARC338 DSD Stable Micturition Cycle Extraction",
        "", "## Frozen scientific contract", "",
        "- Cohort membership comes from the audited subject registry.",
        "- Confirmed voids require CMG plus subject-specific synchronized urine evidence.",
        "- EUS morphology is not used to select cycles; native EUS is preserved for later DSD review.",
        "- Every sample satisfies `0 <= t < first_stim_s`.",
        "- Analysis cycles are all acceptable cycles after sustained stable onset.",
        "- Reference baseline cycles are the final 3-5 cycles of the latest stable run.",
        "- Urine evidence is offline confirmation only and is never an online trigger input.",
        "", "## Subject results", "",
        "| Subject | Confirmed voids | Complete | First stable | Analysis | Runs | Reference | Range |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in summaries:
        reference_range = (f"{row['reference_baseline_first_cycle']}–{row['reference_baseline_last_cycle']}"
                           if row["reference_baseline_first_cycle"] else "none")
        lines.append(
            f"| {row['subject']} | {row['n_confirmed_voids_pre_stim']} | "
            f"{row['n_complete_cycles_pre_stim']} | {row['first_stable_global_cycle']} | "
            f"{row['n_stable_cycles_extracted']} | {row['n_stable_runs']} | "
            f"{row['n_reference_baseline_cycles']} | {reference_range} |"
        )
    lines.extend(["", f"`TOTAL_ANALYSIS_CYCLES = {total}`", "",
                  f"`TOTAL_REFERENCE_BASELINE_CYCLES = {reference_total}`", "",
                  "## Post-onset exclusions", ""])
    for summary in summaries:
        subject = summary["subject"]
        stable_seen = False
        excluded = []
        for row in [item for item in candidates if item["subject"] == subject]:
            if row["is_first_stable_cycle"]:
                stable_seen = True
            if stable_seen and row["cycle_status"] != "PASS_STABLE":
                excluded.append(f"{row['global_cycle_id']}: {row['cycle_status']} ({row['exclusion_reason']})")
        lines.append(f"- {subject}: " + ("; ".join(excluded) if excluded else "none"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_generated(root: Path, manifest: list[dict], references: list[dict]) -> dict:
    errors = []
    reference_keys = {(row["subject"], row["dsd_cycle_id"]) for row in references}
    for row in manifest:
        key = (row["subject"], row["dsd_cycle_id"])
        cycle_dir = root / row["subject"] / row["dsd_cycle_id"]
        aligned_path = cycle_dir / "cycle_100Hz.npz"
        native_path = cycle_dir / "cycle_native_eus.npz"
        if not aligned_path.is_file() or not native_path.is_file():
            errors.append(f"missing cycle files: {key}")
            continue
        with np.load(aligned_path, allow_pickle=False) as data:
            if not np.all(data["t_abs_s"] < float(data["first_stim_s"])):
                errors.append(f"post-stim 100 Hz sample: {key}")
            if bool(data["reference_baseline"]) != (key in reference_keys):
                errors.append(f"reference flag mismatch: {key}")
        with np.load(native_path, allow_pickle=False) as data:
            if not np.all(data["t_eus_abs_s"] < float(data["first_stim_s"])):
                errors.append(f"post-stim native EUS sample: {key}")
    per_subject_reference = {
        subject: sum(row["subject"] == subject for row in references) for subject in SUBJECTS
    }
    for subject, count in per_subject_reference.items():
        if count not in {0, 3, 4, 5}:
            errors.append(f"{subject}: invalid reference baseline count {count}")
    if errors:
        raise RuntimeError("Generated-cycle validation failed: " + "; ".join(errors))
    return {
        "analysis_cycle_count": len(manifest),
        "reference_baseline_count": len(references),
        "reference_baseline_per_subject": per_subject_reference,
        "all_samples_strictly_pre_stim": True,
        "all_cycle_files_present": True,
    }


def _robust_center_scale(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return np.nan, np.nan
    center = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - center)))
    return center, scale


def _reference_stats(references: list[dict]) -> list[dict]:
    rows = []
    for subject in SUBJECTS:
        local = [row for row in references if row["subject"] == subject]
        baseline_center, baseline_scale = _robust_center_scale(
            [float(row["baseline_pressure"]) for row in local]
        )
        delta_center, delta_scale = _robust_center_scale(
            [float(row["delta_p"]) for row in local]
        )
        ici_center, ici_scale = _robust_center_scale([float(row["ici_s"]) for row in local])
        duration_center, duration_scale = _robust_center_scale(
            [float(row["cycle_duration_s"]) for row in local]
        )
        rows.append({
            "subject": subject,
            "n_reference_cycles": len(local),
            "baseline_pressure_median": baseline_center,
            "baseline_pressure_robust_scale": baseline_scale,
            "delta_p_median": delta_center,
            "delta_p_robust_scale": delta_scale,
            "ici_median_s": ici_center,
            "ici_robust_scale_s": ici_scale,
            "cycle_duration_median_s": duration_center,
            "cycle_duration_robust_scale_s": duration_scale,
            "initialization_only": True,
            "refresh_each_session": True,
        })
    return rows


def _build_all(staging: Path, baseline_root: Path,
               previous_voids: list[dict]) -> tuple[list[dict], list[dict]]:
    candidates: list[dict] = []
    manifest: list[dict] = []
    confirmed_voids: list[dict] = []
    summaries: list[dict] = []
    references: list[dict] = []
    for subject in SUBJECTS:
        result = extract_subject(subject, baseline_root)
        _write_subject(staging, result)
        candidates.extend(result["cycles"])
        manifest.extend(result["included"])
        confirmed_voids.extend(result["confirmed_voids"])
        summaries.append(build_subject_summary(result))
        references.extend(result["reference_baseline"])
        print(
            f"{subject}: confirmed={len(result['confirmed_voids'])} "
            f"analysis={len(result['included'])} reference={len(result['reference_baseline'])}",
            flush=True,
        )
    write_csv(staging / "all_confirmed_voids.csv", confirmed_voids, VOID_FIELDS)
    write_csv(staging / "all_candidate_cycles.csv", candidates, CANDIDATE_FIELDS)
    write_csv(staging / "cycle_manifest.csv", manifest, MANIFEST_FIELDS)
    write_csv(staging / "reference_baseline_manifest.csv", references, REFERENCE_BASELINE_FIELDS)
    write_csv(staging / "reference_baseline_stats.csv", _reference_stats(references),
              REFERENCE_STATS_FIELDS)
    write_csv(staging / "subject_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(staging / "STxF21_confirmed_void_comparison.csv",
              _stxf21_void_comparison(previous_voids, confirmed_voids), COMPARISON_FIELDS)
    _write_report(staging / "extraction_report.md", summaries, candidates)
    validation = _validate_generated(staging, manifest, references)
    write_json_atomic(staging / "generated_output_validation.json", validation)
    write_json_atomic(staging / "pipeline_contract.json", {
        "subjects": list(SUBJECTS),
        "subject_registry": {subject: SUBJECT_REGISTRY[subject] for subject in SUBJECTS},
        "analysis_cycle_definition": "ALL_PASS_STABLE_AFTER_EARLIEST_3_CYCLE_SUPPORTED_STABLE_ONSET",
        "reference_baseline_definition": "TAIL_3_TO_5_OF_LATEST_PASS_STABLE_RUN",
        "cycle_boundary": "PREVIOUS_SETTLED_VOID_END_TO_CURRENT_SETTLED_VOID_END",
        "urine_model_input": False,
        "eus_used_for_cycle_selection": False,
    })
    return manifest, summaries


def run(output_root: Path = OUTPUT_ROOT,
        baseline_root: Path = BASELINE_ROOT) -> tuple[list[dict], list[dict]]:
    """Rebuild all four DSD subjects transactionally and remove stale cycle folders."""
    inputs = _input_files(baseline_root)
    before = _hash_snapshot(inputs)
    previous_voids = read_csv(output_root / "all_confirmed_voids.csv")
    staging = make_staging_directory(output_root)
    try:
        manifest, summaries = _build_all(staging, baseline_root, previous_voids)
        after = _hash_snapshot(inputs)
        changed = sorted(path for path in before if before[path] != after.get(path))
        integrity = {
            "protected_file_count": len(inputs),
            "sha256_all_identical": not changed,
            "changed_files": changed,
            "before_sha256": before,
            "after_sha256": after,
        }
        write_json_atomic(staging / "source_integrity.json", integrity)
        if changed:
            raise RuntimeError(f"Protected baseline/validation files changed: {changed}")
        commit_directory(staging, output_root)
        return manifest, summaries
    except Exception:
        cleanup_staging(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("rebuild-dsd", "extract-subject-cycles", "validate-subject-nvc", "audit-urine-evidence", "extract-external-cohort-cycles", "validate-external-cohort"), default="rebuild-dsd")
    parser.add_argument("--subject")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--cycles-root", type=Path)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--first-stim-s", type=float)
    parser.add_argument("--evidence-contract", type=Path)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--source-dataset-id", default="338")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    if args.mode == "extract-subject-cycles":
        if not args.subject or args.input_dir is None:
            parser.error("--subject and --input-dir are required for extract-subject-cycles")
        result = extract_subject_cycles(args.subject, args.input_dir, args.output_root, args.first_stim_s,
                                        args.overwrite, args.evidence_contract)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        return
    if args.mode == "validate-subject-nvc":
        if not args.subject or args.cycles_root is None:
            parser.error("--subject and --cycles-root are required for validate-subject-nvc")
        from ..dsd_feature_extraction.subject_nvc_validation import validate_subject_nvc
        summary = validate_subject_nvc(
            args.subject, args.cycles_root, args.output_root, args.reference_root,
            output_prefix=args.output_prefix,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.mode == "audit-urine-evidence":
        if not args.subject or args.input_dir is None:
            parser.error("--subject and --input-dir are required for audit-urine-evidence")
        audit_root = args.output_root
        contract = audit_subject_urine_evidence(args.subject, args.input_dir, audit_root)
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if args.mode == "extract-external-cohort-cycles":
        if not args.subjects or args.input_root is None or args.audit_root is None:
            parser.error("--subjects, --input-root and --audit-root are required for extract-external-cohort-cycles")
        result = extract_external_cohort_cycles(args.subjects, args.input_root, args.audit_root,
                                                args.output_root, args.source_dataset_id, args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    if args.mode == "validate-external-cohort":
        if not args.subjects or args.cycles_root is None or args.reference_root is None:
            parser.error("--subjects, --cycles-root and --reference-root are required for validate-external-cohort")
        result = validate_external_cohort(args.subjects, args.cycles_root, args.reference_root,
                                          args.output_root, args.source_dataset_id, args.overwrite)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return
    manifest, summaries = run(args.output_root, args.baseline_root)
    print(f"TOTAL_STABLE_DSD_CYCLES={len(manifest)}")
    for row in summaries:
        print(
            f"{row['subject']} first_stable={row['first_stable_global_cycle']} "
            f"analysis={row['n_stable_cycles_extracted']} "
            f"reference={row['n_reference_baseline_cycles']}"
        )


if __name__ == "__main__":
    main()
