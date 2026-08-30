"""V4 preregistered NVC-vs-stable feature-learning configuration."""
from pathlib import Path
import numpy as np
SUBJECTS_338 = ("STxF26", "STxF27", "STxF29")
SUBJECTS_164 = ("STxF31", "STxF33", "STxF34", "STxF35", "STxF37")
SUBJECTS = SUBJECTS_338 + SUBJECTS_164

PRIMARY_SCORE_STAGE = "confirmation_point"
DIAGNOSTIC_DELAYS_S = (0.5, 1.0, 2.0)
DP_FS_HZ = 100.0
BASELINE_WINDOW_S = 25.0
PRESSURE_SPEC_WINDOW_S = 5.0
EUS_SLOW_HISTORY_S = 60.0
COMMON_EUS_HIGH_HZ = 1500.0
EUS_STFT_WINDOW_S = 0.25
EUS_STFT_LOW_HZ = 20.0
EUS_BIN_HZ = 4.0
EUS_FAST_BANDS = ((20.0, 60.0), (60.0, 120.0), (120.0, 250.0), (250.0, 500.0))
EPSILON = 1e-9
RANDOM_STATE = 20260828

P0_FEATURES = ("p_current_delta", "p_peak_delta", "p_threshold_above_duration")
P1_FEATURES = P0_FEATURES + (
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt", "p_positive_dpdt_occupancy",
    "p_auc", "p_auc_growth", "pressure_curvature", "peak_to_current_drop", "p_local_variability",
)
P2_FEATURES = P1_FEATURES + (
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel", "pressure_spectral_entropy",
)
PRESSURE_SPECTRAL_FEATURES = (
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
)

E0_FEATURES = (
    "eus_relative_rms", "eus_relative_amplitude", "eus_envelope_slope", "eus_tonic_occupancy",
    "eus_burst_occupancy", "eus_short_term_variability",
)
E_FAST_FEATURES = tuple(f"eus_relative_log_bandpower_{int(lo)}_{int(hi)}" for lo, hi in EUS_FAST_BANDS)
E1_FEATURES = E0_FEATURES + E_FAST_FEATURES
E_SLOW_FEATURES = ("eus_slow_modulation_power_0p017_0p133", "eus_slow_modulation_depth", "eus_slow_periodicity")
E2_FEATURES = E1_FEATURES + E_SLOW_FEATURES

COUPLING_FEATURES = ("causal_pressure_eus_corr", "eus_activation_latency_s", "pressure_eus_coactivation")
M1_FEATURES = P2_FEATURES
M2_FEATURES = E1_FEATURES
FUSION_FEATURES = ("S_P", "S_E") + COUPLING_FEATURES
M3_FEATURES = FUSION_FEATURES
M4_FEATURES = FUSION_FEATURES

ROOT = Path(__file__).resolve().parents[2]
V31_ROOT = ROOT / "data" / "NVC_V3_1"
OUTPUT_ROOT = ROOT / "data" / "NVC_V4"

FORBIDDEN_FEATURE_TOKENS = ("teacher", "target", "subject", "dataset", "cycle", "event", "urine", "volume", "future", "void", "recovery", "outer")

def assert_safe_feature_schema(names):
    bad = [str(n) for n in names if any(tok in str(n).casefold() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise AssertionError(f"forbidden evaluation/identity feature fields: {bad}")

def stable_feature_names():
    return tuple(dict.fromkeys(P2_FEATURES + E2_FEATURES + COUPLING_FEATURES))
