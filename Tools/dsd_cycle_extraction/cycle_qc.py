"""Cycle-level stability decisions using CMG, urine evidence, timing and artifacts only."""
from __future__ import annotations

from typing import Iterable, Optional

from .config import (
    MIN_STABLE_SUPPORT_CYCLES, REFERENCE_BASELINE_MAX_CYCLES,
    REFERENCE_BASELINE_MIN_CYCLES,
)


def find_first_stable_index(cycles: list[dict], support: int = MIN_STABLE_SUPPORT_CYCLES) -> Optional[int]:
    """Return the earliest cycle followed by a sustained acceptable run."""
    for index in range(0, max(0, len(cycles) - support + 1)):
        window = cycles[index:index + support]
        if all(row.get("complete_cycle") and not row.get("data_gap_flag", False)
               and row.get("stability_candidate") == "STABLE_CANDIDATE"
               for row in window):
            return index
    return None


def assign_cycle_statuses(cycles: list[dict]) -> tuple[list[dict], Optional[int]]:
    """Assign auditable inclusion statuses without using EUS morphology."""
    first_index = find_first_stable_index(cycles)
    next_dataset_id = 1
    run_number = 0
    in_run = False

    for index, row in enumerate(cycles):
        row["is_first_stable_cycle"] = bool(first_index is not None and index == first_index)
        row["stable_candidate"] = row.get("stability_candidate") == "STABLE_CANDIDATE"
        row["cycle_stability_status"] = row.get("stability_candidate", "REVIEW_REQUIRED")
        row["dsd_cycle_id"] = ""
        row["stable_run_id"] = ""
        row["reference_baseline"] = False
        row["reference_baseline_id"] = ""

        if not row.get("complete_cycle", False):
            row["cycle_status"] = "EXCLUDE_INCOMPLETE"
            row["exclusion_reason"] = row.get("exclusion_reason") or "INCOMPLETE_CONFIRMED_VOID_INTERVAL"
        elif first_index is None:
            row["cycle_status"] = "REVIEW_REQUIRED"
            row["exclusion_reason"] = "NO_SUSTAINED_STABLE_ONSET"
        elif index < first_index:
            row["cycle_status"] = "EXCLUDE_ACCLIMATION"
            row["exclusion_reason"] = "BEFORE_FIRST_SUSTAINED_STABLE_CYCLE"
        elif row.get("data_gap_flag", False):
            row["cycle_status"] = "EXCLUDE_DATA_GAP"
            row["exclusion_reason"] = "CMG_INVALID_SAMPLE_FRACTION_EXCEEDED_LIMIT"
        elif row.get("artifact_overlap", False):
            row["cycle_status"] = "EXCLUDE_PRESSURE_ARTIFACT"
            row["exclusion_reason"] = "SEVERE_PRESSURE_ARTIFACT_OVERLAP"
        elif row.get("stability_candidate") != "STABLE_CANDIDATE":
            row["cycle_status"] = "EXCLUDE_TRANSITIONAL"
            row["exclusion_reason"] = row.get("exclusion_reason") or "CYCLE_STABILITY_QC_FAILED"
        elif float(row["cycle_end_s"]) >= float(row["first_stim_s"]):
            row["cycle_status"] = "EXCLUDE_PRE_STIM_BOUNDARY"
            row["exclusion_reason"] = "EXCLUDE_INCOMPLETE_PRE_STIM"
        else:
            row["cycle_status"] = "PASS_STABLE"
            row["exclusion_reason"] = ""
            if not in_run:
                run_number += 1
            in_run = True
            row["stable_run_id"] = f"R{run_number:02d}"
            row["dsd_cycle_id"] = f"B{next_dataset_id:02d}"
            next_dataset_id += 1
            continue
        in_run = False

    # Reference baseline and analysis-cycle inclusion are deliberately distinct:
    # - included/PASS_STABLE: all acceptable cycles after sustained stable onset;
    # - reference baseline: tail 3-5 cycles of the latest acceptable run, suitable
    #   for subject/session initialization without changing the analysis dataset.
    included = [row for row in cycles if row.get("cycle_status") == "PASS_STABLE"]
    run_ids = [row["stable_run_id"] for row in included]
    latest_run_id = run_ids[-1] if run_ids else ""
    latest_run = [row for row in included if row["stable_run_id"] == latest_run_id]
    reference = (latest_run[-REFERENCE_BASELINE_MAX_CYCLES:]
                 if len(latest_run) >= REFERENCE_BASELINE_MIN_CYCLES else [])
    for number, row in enumerate(reference, 1):
        row["reference_baseline"] = True
        row["reference_baseline_id"] = f"RB{number:02d}"
        row["selection_reason"] = "TAIL_OF_LATEST_STABLE_RUN_FOR_SUBJECT_SESSION_INITIALIZATION"

    assign_nvc_eligibility(cycles)
    return cycles, first_index


def assign_nvc_eligibility(cycles: list[dict], minimum_duration_s: float = 30.0) -> list[dict]:
    """Assign single-cycle NVC eligibility independently of baseline stability.

    Stability remains useful for baseline initialization and quality grading,
    but a complete confirmed-void interval is not discarded merely because it
    is isolated or has one subject-internal statistical outlier.
    """
    for row in cycles:
        reason = ""
        if not row.get("complete_cycle", False):
            reason = "INCOMPLETE_CONFIRMED_VOID_INTERVAL"
        elif not row.get("confirmed_void", row.get("urine_confirmed", False)):
            reason = "TERMINAL_VOID_NOT_CONFIRMED"
        elif row.get("data_gap_flag", False):
            reason = "CMG_INVALID_SAMPLE_FRACTION_EXCEEDED_LIMIT"
        elif row.get("artifact_overlap", False):
            reason = "SEVERE_PRESSURE_ARTIFACT_OVERLAP"
        elif float(row.get("cycle_end_s", float("inf"))) >= float(row.get("first_stim_s", float("-inf"))):
            reason = "CYCLE_CROSSES_FIRST_STIMULATION"
        elif float(row.get("cycle_duration_s", 0.0)) < minimum_duration_s:
            reason = "INSUFFICIENT_FILLING_DURATION_FOR_NVC"

        eligible = not reason
        statistical_review = bool(
            eligible and row.get("stability_candidate") != "STABLE_CANDIDATE"
        )
        row["nvc_eligible"] = eligible
        row["nvc_exclusion_reason"] = reason
        row["nvc_quality_status"] = (
            "NVC_ELIGIBLE_STATISTICAL_REVIEW" if statistical_review
            else "NVC_ELIGIBLE" if eligible
            else "NVC_EXCLUDED"
        )
    return cycles


def pass_cycles(cycles: Iterable[dict]) -> list[dict]:
    return [row for row in cycles if row.get("cycle_status") == "PASS_STABLE"]


def nvc_eligible_cycles(cycles: Iterable[dict]) -> list[dict]:
    return [row for row in cycles if row.get("nvc_eligible", False)]
