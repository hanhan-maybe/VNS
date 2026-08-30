"""Frozen V3.2 preregistration and paths.

The registered cohort and PE schema are frozen locally so this version does not
import executable code from another model version.
"""
from pathlib import Path
import numpy as np

from .version_support import EUS_FEATURES, P_FEATURES as LEGACY_P_FEATURES

SUBJECTS_338 = ("STxF26", "STxF27", "STxF29")
SUBJECTS_164 = ("STxF31", "STxF33", "STxF34", "STxF35", "STxF37")
SUBJECTS = SUBJECTS_338 + SUBJECTS_164
PE_FEATURES = tuple(LEGACY_P_FEATURES) + tuple(EUS_FEATURES)
TRAJECTORY_FEATURES = (
    "slope_change_250ms", "slope_change_500ms", "slope_ratio_500ms",
    "pressure_curvature", "current_dpdt_to_max_positive_so_far",
    "peak_to_current_drop_so_far", "positive_slope_occupancy",
    "eus_delta_from_event_onset", "eus_envelope_slope_trajectory",
    "causal_pressure_eus_corr",
)

PRIMARY_DELAY_S = 2.0
DIAGNOSTIC_DELAYS_S = (0.5, 1.0, 2.0, 3.0, 5.0)
DELAYS_S = DIAGNOSTIC_DELAYS_S
PRESSURE_SPEC_WINDOW_S = 5.0
BASELINE_WINDOW_S = 25.0
DP_FS_HZ = 100.0
COMMON_EUS_HIGH_HZ = 1500.0
EUS_STFT_WINDOW_S = 0.25
EUS_STFT_OVERLAP = 0.5
EUS_STFT_LOW_HZ = 20.0
EUS_BIN_HZ = 4.0
EPSILON = 1e-9
RANDOM_STATE = 20260221
TARGET_LABELS = ("NVC_CORE", "PREVOID_PROGRESSIVE")
LABEL_TO_TARGET = {"NVC_CORE": 1, "PREVOID_PROGRESSIVE": 0}

P_FEATURES = (
    "delta_p_current_norm", "delta_p_peak_so_far_norm",
    "pressure_slope_0p5s_norm", "pressure_slope_change_norm",
    "positive_dpdt_fraction_1s", "auc_growth_rate_norm",
)
TRAJECTORY_COMPACT_FEATURES = (
    "pressure_curvature", "peak_to_current_drop_so_far", "positive_slope_occupancy",
)
PRESSURE_SPECTRAL_FEATURES = (
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
)
EUS_COMPACT_BANDS = ((20.0, 60.0), (60.0, 120.0), (120.0, 250.0), (250.0, 500.0))
EUS_COMPACT_FEATURES = tuple(f"eus_relative_log_bandpower_{int(lo)}_{int(hi)}" for lo, hi in EUS_COMPACT_BANDS)
COUPLING_FEATURE = "causal_pressure_eus_corr"

M1_FEATURES = P_FEATURES + PRESSURE_SPECTRAL_FEATURES
M2_FEATURES = tuple(PE_FEATURES) + tuple(
    f"eus_stft_bin_{int(lo)}_{int(lo + EUS_BIN_HZ)}"
    for lo in np.arange(EUS_STFT_LOW_HZ, COMMON_EUS_HIGH_HZ, EUS_BIN_HZ)
)
M3_FEATURES = P_FEATURES + TRAJECTORY_COMPACT_FEATURES + PRESSURE_SPECTRAL_FEATURES + EUS_COMPACT_FEATURES + (COUPLING_FEATURE,)
M5_FEATURES = M3_FEATURES

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "data" / "NVC_V3_2"
DEFAULT_V31_ROOT = ROOT / "data" / "NVC_V3_1"
DEFAULT_V3_ROOT = ROOT / "data" / "NVC_V3"
DEFAULT_338_CYCLES = ROOT / "data" / "DSD_cycles"
DEFAULT_338_REFERENCE = ROOT / "data" / "DSD_nvc_results"
DEFAULT_164_CYCLES = ROOT / "data" / "SPARC164_cycles"
DEFAULT_164_LABELS = ROOT / "data" / "SPARC164_nvc_results" / "sparc164_teacher_labels.csv"

FORBIDDEN_FEATURE_TOKENS = {
    "teacher_label", "target", "subject", "dataset", "cycle_id", "event_id", "event_uid",
    "matched_urine_event_id", "still_active", "actionable", "recovery", "urine", "volume",
    "outer_fold", "future", "eval_only",
}

def common_eus_bands(native_rates, reliable_high=None):
    """Return the fixed common bands and an audit record, never animal-specific bands."""
    rates = tuple(float(x) for x in native_rates if x is not None)
    high = min([COMMON_EUS_HIGH_HZ, *(r / 2.0 for r in rates)] if rates else [COMMON_EUS_HIGH_HZ])
    if reliable_high is not None:
        high = min(high, float(reliable_high))
    included = tuple((lo, hi) for lo, hi in EUS_COMPACT_BANDS if hi <= high + 1e-9)
    excluded = tuple((lo, hi) for lo, hi in EUS_COMPACT_BANDS if hi > high + 1e-9)
    return high, included, excluded
