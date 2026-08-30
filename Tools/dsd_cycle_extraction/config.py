"""Schemas for DSD cycle extraction; shared paths live in sparc338_config."""
try:
    from sparc338_config import (
        BASELINE_ROOT, BOUNDARY_METHOD, DSD_CYCLES_ROOT, DSD_SUBJECTS,
        PROJECT_ROOT, RAW_ROOT, REFERENCE_BASELINE_MAX_CYCLES,
        REFERENCE_BASELINE_MIN_CYCLES, STABLE_ONSET_SUPPORT_CYCLES,
        SUBJECT_REGISTRY, VALIDATION_ROOT,
    )
except ImportError:
    from Tools.sparc338_config import (
        BASELINE_ROOT, BOUNDARY_METHOD, DSD_CYCLES_ROOT, DSD_SUBJECTS,
        PROJECT_ROOT, RAW_ROOT, REFERENCE_BASELINE_MAX_CYCLES,
        REFERENCE_BASELINE_MIN_CYCLES, STABLE_ONSET_SUPPORT_CYCLES,
        SUBJECT_REGISTRY, VALIDATION_ROOT,
    )


OUTPUT_ROOT = DSD_CYCLES_ROOT
SUBJECTS = DSD_SUBJECTS
MIN_STABLE_SUPPORT_CYCLES = STABLE_ONSET_SUPPORT_CYCLES

CANDIDATE_FIELDS = (
    "subject", "global_cycle_id", "dsd_cycle_id", "reference_baseline_id",
    "reference_baseline", "cycle_start_s", "cycle_end_s", "cycle_duration_s",
    "previous_void_end_s", "void_start_s", "cmg_peak_s", "urine_output_onset_s",
    "void_end_s", "first_stim_s", "pre_stim_margin_s", "urine_evidence_type",
    "confirmed_void", "stable_candidate", "is_first_stable_cycle", "stable_run_id",
    "cycle_stability_status", "cycle_status", "exclusion_reason", "complete_cycle",
    "ici_s", "baseline_pressure", "peak_pressure", "delta_p", "urine_output_amount",
    "cmg_artifact_flag", "ici_local_cv", "duration_local_cv", "baseline_pressure_local_cv",
    "peak_pressure_local_cv", "delta_p_local_cv", "urine_output_local_cv",
    "ici_robust_z", "duration_robust_z", "baseline_pressure_robust_z",
    "peak_pressure_robust_z", "delta_p_robust_z", "urine_output_robust_z",
    "cmg_invalid_fraction", "eus_invalid_fraction", "data_gap_flag",
    "source_pre_stim_file", "cycle_boundary_method",
)

MANIFEST_FIELDS = (
    "subject", "global_cycle_id", "dsd_cycle_id", "stable_run_id",
    "reference_baseline_id", "reference_baseline",
    "cycle_start_s", "cycle_end_s", "cycle_duration_s", "previous_void_end_s",
    "void_start_s", "cmg_peak_s", "urine_output_onset_s", "void_end_s",
    "first_stim_s", "pre_stim_margin_s", "urine_evidence_type", "confirmed_void",
    "is_first_stable_cycle", "cycle_stability_status", "cycle_status", "ici_s",
    "baseline_pressure", "peak_pressure", "delta_p", "urine_output_amount",
    "cmg_artifact_flag", "complete_cycle", "source_pre_stim_file", "cycle_boundary_method",
    "cmg_invalid_fraction", "eus_invalid_fraction", "data_gap_flag",
)

VOID_FIELDS = (
    "subject", "void_global_id", "void_start_s", "cmg_peak_s", "urine_output_onset_s",
    "void_end_s", "settled_void_end_s", "urine_evidence_type", "confirmed_void",
    "source_available", "first_stim_s",
)

SUMMARY_FIELDS = (
    "subject", "first_stim_s", "pre_stim_duration_s", "n_confirmed_voids_pre_stim",
    "n_complete_cycles_pre_stim", "n_acclimation_excluded", "n_artifact_excluded",
    "n_data_gap_excluded", "n_transitional_excluded", "n_incomplete_excluded",
    "first_stable_global_cycle",
    "first_stable_time_s", "first_stable_void_time_s", "n_candidates_first_stable_to_stim",
    "n_stable_cycles_extracted", "n_stable_runs", "first_extracted_cycle_start_s",
    "last_extracted_cycle_end_s", "total_stable_duration_s", "n_reference_baseline_cycles",
    "reference_baseline_first_cycle", "reference_baseline_last_cycle",
    "reference_baseline_status", "urine_evidence_type", "status",
)

REFERENCE_BASELINE_FIELDS = (
    "subject", "reference_baseline_id", "global_cycle_id", "dsd_cycle_id",
    "cycle_start_s", "cycle_end_s", "void_start_s", "cmg_peak_s",
    "urine_output_onset_s", "void_end_s", "stable_run_id", "selection_reason",
)

REFERENCE_STATS_FIELDS = (
    "subject", "n_reference_cycles", "baseline_pressure_median",
    "baseline_pressure_robust_scale", "delta_p_median", "delta_p_robust_scale",
    "ici_median_s", "ici_robust_scale_s", "cycle_duration_median_s",
    "cycle_duration_robust_scale_s", "initialization_only", "refresh_each_session",
)
