"""Frozen V3.1 experiment configuration."""
from pathlib import Path

from .version_support import (
    C0_FEATURES,
    EUS_FEATURES,
    P_FEATURES,
    SUBJECTS_164,
    SUBJECTS_338,
    SPECTRAL_FEATURES,
)

SUBJECTS = SUBJECTS_338 + SUBJECTS_164
DELAYS_S = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0)
TARGET_LABELS = ("NVC_CORE", "PREVOID_PROGRESSIVE")
LABEL_TO_TARGET = {"NVC_CORE": 1, "PREVOID_PROGRESSIVE": 0}
EPSILON = 1e-6
RANDOM_STATE = 20260221

PE_FEATURES = tuple(P_FEATURES) + tuple(EUS_FEATURES)
TRAJECTORY_FEATURES = (
    "slope_change_250ms",
    "slope_change_500ms",
    "slope_ratio_500ms",
    "pressure_curvature",
    "current_dpdt_to_max_positive_so_far",
    "peak_to_current_drop_so_far",
    "positive_slope_occupancy",
    "eus_delta_from_event_onset",
    "eus_envelope_slope_trajectory",
    "causal_pressure_eus_corr",
)
PE_TRAJECTORY_FEATURES = PE_FEATURES + TRAJECTORY_FEATURES
MODEL_FEATURES = {
    "C0": tuple(C0_FEATURES),
    "P": tuple(P_FEATURES),
    "PE": PE_FEATURES,
    "PE_SPECTRAL_COMMON": PE_FEATURES,
    "PEF": PE_FEATURES + tuple(SPECTRAL_FEATURES),
    "PE_DELAY": PE_FEATURES,
    "PE_TRAJECTORY": PE_TRAJECTORY_FEATURES,
    "CANDIDATE+VOIDGUARD": PE_TRAJECTORY_FEATURES,
}

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V3_ROOT = ROOT / "data" / "NVC_V3"
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "NVC_V3_1"
DEFAULT_338_CYCLES = ROOT / "data" / "DSD_cycles"
DEFAULT_338_REFERENCE = ROOT / "data" / "DSD_nvc_results"
DEFAULT_164_CYCLES = ROOT / "data" / "SPARC164_cycles"
DEFAULT_164_LABELS = ROOT / "data" / "SPARC164_nvc_results" / "sparc164_teacher_labels.csv"
