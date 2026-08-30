"""Load verified PRE_STIM products and construct all adjacent confirmed-void cycles."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from .config import BASELINE_ROOT, BOUNDARY_METHOD, SUBJECT_REGISTRY
from .cycle_qc import assign_cycle_statuses, pass_cycles
from .urine_evidence_adapter import load_urine_evidence, stable_phase_inputs


TOOLS_ROOT = Path(__file__).resolve().parents[1]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from sparc338_stable_phase import build_stable_baseline  # noqa: E402
from sparc338_urine_output import volume_display  # noqa: E402


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_verified_urine(subject: str, subject_dir: Path) -> tuple[str, dict, dict]:
    """Load only the already-QC'd synchronized urine evidence."""
    evidence = load_urine_evidence(subject, subject_dir, volume_display)
    urine_mode, confirm_data = stable_phase_inputs(evidence)
    export_data = {
        "evidence_type": ("LEAK_BUTTON_EVENT" if evidence.source_type == "LEAK_BUTTON_EVENT"
                          else "CONFIRMED_URINE_OUTPUT"),
        "source_type": evidence.source_type,
        "drop_times": evidence.event_times_s,
        "time_s": evidence.continuous_time_s,
        "trace": evidence.continuous_value,
        "continuous_available": evidence.continuous_available,
        "model_input": evidence.model_input,
        "metadata": evidence.metadata,
    }
    return urine_mode, confirm_data, export_data


def _mapped_cycle(source: dict, subject: str, first_stim_s: float, evidence_type: str,
                  source_file: Path) -> dict:
    row = dict(source)
    row.update({
        "subject": subject,
        "global_cycle_id": source["original_cycle_id"],
        "previous_void_end_s": source["cycle_start_s"],
        "first_stim_s": first_stim_s,
        "pre_stim_margin_s": first_stim_s - float(source["cycle_end_s"]),
        "urine_evidence_type": evidence_type,
        "confirmed_void": bool(source.get("urine_confirmed", False)),
        "ici_s": source.get("ICI_s", np.nan),
        "baseline_pressure": source.get("pre_void_baseline_pressure", np.nan),
        "peak_pressure": source.get("cmg_peak_pressure", np.nan),
        "delta_p": source.get("deltaP", np.nan),
        "urine_output_amount": source.get("urine_output_per_cycle", np.nan),
        "cmg_artifact_flag": bool(source.get("artifact_overlap", False)),
        "ici_local_cv": source.get("ICI_s_local_CV", np.nan),
        "duration_local_cv": source.get("cycle_duration_s_local_CV", np.nan),
        "baseline_pressure_local_cv": source.get("pre_void_baseline_pressure_local_CV", np.nan),
        "peak_pressure_local_cv": source.get("cmg_peak_pressure_local_CV", np.nan),
        "delta_p_local_cv": source.get("deltaP_local_CV", np.nan),
        "urine_output_local_cv": source.get("urine_output_per_cycle_local_CV", np.nan),
        "ici_robust_z": source.get("ICI_s_robust_z", np.nan),
        "duration_robust_z": source.get("cycle_duration_s_robust_z", np.nan),
        "baseline_pressure_robust_z": source.get("pre_void_baseline_pressure_robust_z", np.nan),
        "peak_pressure_robust_z": source.get("cmg_peak_pressure_robust_z", np.nan),
        "delta_p_robust_z": source.get("deltaP_robust_z", np.nan),
        "urine_output_robust_z": source.get("urine_output_per_cycle_robust_z", np.nan),
        "source_pre_stim_file": str(source_file),
        "cycle_boundary_method": BOUNDARY_METHOD,
    })
    return row


def extract_subject(subject: str, baseline_root: Path = BASELINE_ROOT) -> dict:
    """Build and QC every PRE_STIM confirmed-void cycle for one subject."""
    subject_dir = baseline_root / subject
    summary = json.loads((subject_dir / "pre_stim_summary.json").read_text(encoding="utf-8"))
    first_stim_s = float(summary["first_stim_s"])
    processed_path = subject_dir / "pre_stim_100Hz.npz"
    raw_path = subject_dir / "pre_stim_raw.npz"

    with np.load(processed_path, allow_pickle=False) as data:
        time_100hz = data["time_s"].copy()
        pressure_100hz = data["bladder_pressure_mmHg"].copy()
        envelope_100hz = data["eus_envelope_mV"].copy()
        display_fs_hz = float(data["sample_rate_hz"])
        bladder_valid_100hz = (data["bladder_valid_100hz"].copy().astype(bool)
                               if "bladder_valid_100hz" in data.files
                               else np.ones(time_100hz.shape, dtype=bool))
        eus_valid_100hz = (data["eus_valid_100hz"].copy().astype(bool)
                           if "eus_valid_100hz" in data.files
                           else np.ones(time_100hz.shape, dtype=bool))
    with np.load(raw_path, allow_pickle=False) as data:
        bladder_raw = data["bladder_raw"].copy()
        bladder_fs_hz = float(data["bladder_fs_hz"])
        eus_raw = data["eus_raw"].copy()
        eus_fs_hz = float(data["eus_fs_hz"])
        raw_end_s = float(data["pre_stim_end_s"])

    if not np.isclose(first_stim_s, raw_end_s, atol=1e-6):
        raise ValueError(f"{subject}: first_stim_s disagrees with pre_stim_raw metadata")
    if time_100hz.size and not np.all(time_100hz < first_stim_s):
        raise ValueError(f"{subject}: processed PRE_STIM contains samples at/after first stimulation")

    urine_mode, confirm_urine, export_urine = load_verified_urine(subject, subject_dir)
    source_cycles, _, _, audit = build_stable_baseline(
        time_100hz, pressure_100hz, first_stim_s, urine_mode, confirm_urine
    )
    cycles = [
        _mapped_cycle(row, subject, first_stim_s, export_urine["evidence_type"], processed_path)
        for row in source_cycles
    ]
    for row in cycles:
        start = float(row["cycle_start_s"])
        end = float(row["cycle_end_s"])
        if not np.isfinite(start) or not np.isfinite(end):
            row.update(cmg_invalid_fraction=np.nan, eus_invalid_fraction=np.nan,
                       data_gap_flag=True)
            continue
        mask = (time_100hz >= start) & (time_100hz < end)
        cmg_invalid = float(np.mean(~bladder_valid_100hz[mask])) if mask.any() else 1.0
        eus_invalid = float(np.mean(~eus_valid_100hz[mask])) if mask.any() else 1.0
        row.update(
            cmg_invalid_fraction=cmg_invalid,
            eus_invalid_fraction=eus_invalid,
            data_gap_flag=bool(cmg_invalid > 0.005),
        )
    cycles, first_stable_index = assign_cycle_statuses(cycles)
    included = pass_cycles(cycles)
    reference_baseline = [row for row in included if row["reference_baseline"]]

    confirmed_voids = []
    for index, row in enumerate(cycles, 1):
        reliable = all(np.isfinite(float(row[key])) for key in (
            "void_start_s", "cmg_peak_s", "urine_output_onset_s", "void_end_s", "cycle_end_s"
        ))
        confirmed_voids.append({
            "subject": subject,
            "void_global_id": f"V{index:02d}",
            "void_start_s": row["void_start_s"],
            "cmg_peak_s": row["cmg_peak_s"],
            "urine_output_onset_s": row["urine_output_onset_s"],
            "void_end_s": row["void_end_s"],
            "settled_void_end_s": row["cycle_end_s"],
            "urine_evidence_type": row["urine_evidence_type"],
            "confirmed_void": True,
            "source_available": reliable,
            "first_stim_s": first_stim_s,
        })

    cache = {
        "time_100hz": time_100hz,
        "pressure_100hz": pressure_100hz,
        "envelope_100hz": envelope_100hz,
        "bladder_valid_100hz": bladder_valid_100hz,
        "eus_valid_100hz": eus_valid_100hz,
        "display_fs_hz": display_fs_hz,
        "bladder_raw": bladder_raw,
        "bladder_fs_hz": bladder_fs_hz,
        "eus_raw": eus_raw,
        "eus_fs_hz": eus_fs_hz,
        "urine": export_urine,
        "first_stim_s": first_stim_s,
        "baseline_root": baseline_root.resolve(),
    }
    return {
        "subject": subject,
        "cycles": cycles,
        "included": included,
        "reference_baseline": reference_baseline,
        "confirmed_voids": confirmed_voids,
        "first_stable_index": first_stable_index,
        "audit": audit,
        "cache": cache,
    }


def _sample_native(raw: np.ndarray, fs_hz: float, times_s: np.ndarray) -> np.ndarray:
    positions = np.clip(times_s * fs_hz, 0.0, max(0.0, len(raw) - 1.0))
    left = np.floor(positions).astype(np.int64)
    right = np.minimum(left + 1, len(raw) - 1)
    fraction = positions - left
    return ((1.0 - fraction) * raw[left] + fraction * raw[right]).astype(np.float32)


def cycle_arrays(cache: dict, row: dict) -> tuple[dict, dict]:
    """Create aligned 100 Hz data and a lossless native-rate EUS crop."""
    start = float(row["cycle_start_s"])
    end = float(row["cycle_end_s"])
    first_stim_s = float(row["first_stim_s"])
    time_100hz = cache["time_100hz"]
    mask = (time_100hz >= start) & (time_100hz < end) & (time_100hz < first_stim_s)
    t_abs = time_100hz[mask].astype(np.float64)
    if not t_abs.size:
        raise ValueError(f"{row['subject']} {row['global_cycle_id']}: empty 100 Hz crop")

    urine = cache["urine"]
    if urine["source_type"] == "CONTINUOUS_WEIGHT":
        urine_100hz = np.interp(t_abs, urine["time_s"], urine["trace"]).astype(np.float32)
        urine_events = np.empty(0, dtype=np.float64)
    else:
        urine_100hz = np.full(t_abs.shape, np.nan, dtype=np.float32)
        urine_events = urine["drop_times"][(urine["drop_times"] >= start) & (urine["drop_times"] < end)]

    metadata = {
        "subject": np.array(row["subject"]),
        "global_cycle_id": np.array(row["global_cycle_id"]),
        "dsd_cycle_id": np.array(row["dsd_cycle_id"]),
        "cycle_start_s": np.array(start),
        "cycle_end_s": np.array(end),
        "cycle_duration_s": np.array(end - start),
        "void_start_s": np.array(float(row["void_start_s"])),
        "cmg_peak_s": np.array(float(row["cmg_peak_s"])),
        "urine_output_onset_s": np.array(float(row["urine_output_onset_s"])),
        "void_end_s": np.array(float(row["void_end_s"])),
        "first_stim_s": np.array(first_stim_s),
        "cycle_boundary_method": np.array(row["cycle_boundary_method"]),
        "urine_evidence_type": np.array(row["urine_evidence_type"]),
        "urine_source_type": np.array(urine["source_type"]),
        "urine_evidence_source": np.array(urine["source_type"]),
        "fvol_continuous_available": np.array(urine["continuous_available"]),
        "urine_output_model_input": np.array(False),
        "reference_baseline": np.array(bool(row["reference_baseline"])),
        "reference_baseline_id": np.array(row["reference_baseline_id"]),
        "dsd_registry_confirmed": np.array(bool(SUBJECT_REGISTRY[row["subject"]]["dsd_confirmed"])),
    }
    aligned = {
        **metadata,
        "t_rel_s": (t_abs - start).astype(np.float64),
        "t_abs_s": t_abs,
        "cmg_native_sampled_100hz": _sample_native(
            cache["bladder_raw"], cache["bladder_fs_hz"], t_abs
        ),
        # Compatibility alias retained for existing downstream readers.
        "cmg_raw_100hz": _sample_native(cache["bladder_raw"], cache["bladder_fs_hz"], t_abs),
        "cmg_processed_100hz": cache["pressure_100hz"][mask].astype(np.float32),
        "eus_envelope_100hz": cache["envelope_100hz"][mask].astype(np.float32),
        "bladder_pressure_mmHg": cache["pressure_100hz"][mask].astype(np.float32),
        "eus_envelope_mV": cache["envelope_100hz"][mask].astype(np.float32),
        "cmg_valid_100hz": cache["bladder_valid_100hz"][mask],
        "eus_valid_100hz": cache["eus_valid_100hz"][mask],
        "sample_rate_hz": np.array(cache["display_fs_hz"]),
        "urine_output_auxiliary_100hz": urine_100hz,
        "urine_drop_event_times_s": urine_events.astype(np.float64),
        "urine_event_times_abs_s": urine_events.astype(np.float64),
        "urine_event_times_cycle_s": (urine_events - start).astype(np.float64),
        "urine_source_metadata_json": np.array(json.dumps(urine["metadata"], ensure_ascii=False)),
    }

    eus_fs_hz = float(cache["eus_fs_hz"])
    start_sample = int(np.ceil(start * eus_fs_hz - 1e-9))
    end_sample = min(len(cache["eus_raw"]), int(np.ceil(end * eus_fs_hz - 1e-9)))
    sample_indices = np.arange(start_sample, end_sample, dtype=np.int64)
    eus_abs = sample_indices.astype(np.float64) / eus_fs_hz
    native = {
        **metadata,
        "eus_raw": cache["eus_raw"][start_sample:end_sample].astype(np.float32, copy=False),
        "eus_fs": np.array(eus_fs_hz),
        "t_eus_rel_s": eus_abs - start,
        "t_eus_abs_s": eus_abs,
        "source_file": np.array(str(cache["baseline_root"] / row["subject"] / "pre_stim_raw.npz")),
        "source_start_sample": np.array(start_sample, dtype=np.int64),
        "source_end_sample": np.array(end_sample, dtype=np.int64),
    }
    return aligned, native


def build_subject_summary(result: dict) -> dict:
    cycles = result["cycles"]
    included = result["included"]
    first_index = result["first_stable_index"]
    first = cycles[first_index] if first_index is not None else None
    reference = result["reference_baseline"]
    return {
        "subject": result["subject"],
        "first_stim_s": result["cache"]["first_stim_s"],
        "pre_stim_duration_s": result["cache"]["first_stim_s"],
        "n_confirmed_voids_pre_stim": len(result["confirmed_voids"]),
        "n_complete_cycles_pre_stim": sum(bool(row["complete_cycle"]) for row in cycles),
        "n_acclimation_excluded": sum(row["cycle_status"] == "EXCLUDE_ACCLIMATION" for row in cycles),
        "n_artifact_excluded": sum(row["cycle_status"] == "EXCLUDE_PRESSURE_ARTIFACT" for row in cycles),
        "n_data_gap_excluded": sum(row["cycle_status"] == "EXCLUDE_DATA_GAP" for row in cycles),
        "n_transitional_excluded": sum(row["cycle_status"] == "EXCLUDE_TRANSITIONAL" for row in cycles),
        "n_incomplete_excluded": sum(row["cycle_status"] in {"EXCLUDE_INCOMPLETE", "EXCLUDE_PRE_STIM_BOUNDARY"} for row in cycles),
        "first_stable_global_cycle": first["global_cycle_id"] if first else "",
        "first_stable_time_s": first["cycle_start_s"] if first else np.nan,
        "first_stable_void_time_s": first["void_start_s"] if first else np.nan,
        "n_candidates_first_stable_to_stim": len(cycles) - first_index if first_index is not None else 0,
        "n_stable_cycles_extracted": len(included),
        "n_stable_runs": len({row["stable_run_id"] for row in included}),
        "first_extracted_cycle_start_s": included[0]["cycle_start_s"] if included else np.nan,
        "last_extracted_cycle_end_s": included[-1]["cycle_end_s"] if included else np.nan,
        "total_stable_duration_s": sum(float(row["cycle_duration_s"]) for row in included),
        "n_reference_baseline_cycles": len(reference),
        "reference_baseline_first_cycle": reference[0]["dsd_cycle_id"] if reference else "",
        "reference_baseline_last_cycle": reference[-1]["dsd_cycle_id"] if reference else "",
        "reference_baseline_status": ("PASS_REFERENCE_BASELINE"
                                      if len(reference) >= 3
                                      else "HOLD_INSUFFICIENT_REFERENCE_BASELINE"),
        "urine_evidence_type": cycles[0]["urine_evidence_type"] if cycles else "",
        "status": "PASS" if included else "REVIEW_REQUIRED",
    }
