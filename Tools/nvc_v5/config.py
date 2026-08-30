"""Frozen V5 individualized prospective experiment configuration.

V5 intentionally changes the validation unit (within-animal prospective
calibration -> future-cycle replay).  The registered feature schema is frozen
locally so V5 never imports executable code from another model version.
"""
from pathlib import Path

SUBJECTS = ("STxF37", "STxF26")
SPLITS = {
    "STxF37": {"train": ("B01", "B02", "B03", "B04"),
               "test": ("B05", "B06", "B07")},
    "STxF26": {"train": tuple(f"B{i:02d}" for i in range(1, 14)),
               "test": ("B14", "B15", "B16")},
}
CALIBRATION_LENGTH = {
    "STxF37": (
        {"name": "C1", "train": ("B01", "B02"), "test": ("B03", "B04", "B05", "B06", "B07")},
        {"name": "C2", "train": ("B01", "B02", "B03"), "test": ("B04", "B05", "B06", "B07")},
        {"name": "C3", "train": ("B01", "B02", "B03", "B04"), "test": ("B05", "B06", "B07")},
    )
}

PRIMARY_TRAIN_LABELS = ("NVC_CORE", "STABLE_FILLING")
CHALLENGE_LABELS = ("PREVOID_PROGRESSIVE", "VOID_CONFIRMED")
RANDOM_STATE = 20260828
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
PRESSURE_SPECTRAL_FEATURES = (
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
)
STREAM_UPDATE_S = 0.25
STREAM_DELAYS_S = (0.0, 0.5, 1.0, 2.0)

# V5 parallel five-model registration.  These are deliberately independent
# candidates: a failure in one model must not gate the others.
PARALLEL_MODELS = ("R0", "M1", "M2", "M3", "M4")
PARALLEL_MODEL_DESCRIPTIONS = {
    "R0": "I-M1-HC frozen high-coverage pressure reference",
    "M1": "P-EARLY causal pressure redesign without event-age variability",
    "M2": "E-EARLY EUS time-domain plus compact fast spectrum",
    "M3": "PE-EARLY low-dimensional pressure/EUS fusion",
    "M4": "EUS-SP-LASSO causal EUS-to-pressure sparse regression",
}
# V4 schemas are reproduced verbatim without importing V4 code.
P0_FEATURES = ("p_current_delta", "p_peak_delta", "p_threshold_above_duration")
P1_FEATURES = P0_FEATURES + (
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt", "p_positive_dpdt_occupancy",
    "p_auc", "p_auc_growth", "pressure_curvature", "peak_to_current_drop", "p_local_variability",
)
M1_FEATURES = P1_FEATURES + (
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel", "pressure_spectral_entropy",
)
P2_FEATURES = M1_FEATURES
E0_FEATURES = (
    "eus_relative_rms", "eus_relative_amplitude", "eus_envelope_slope", "eus_tonic_occupancy",
    "eus_burst_occupancy", "eus_short_term_variability",
)
E_FAST_FEATURES = tuple(
    f"eus_relative_log_bandpower_{int(lo)}_{int(hi)}" for lo, hi in EUS_FAST_BANDS)
M2_FEATURES = E0_FEATURES + E_FAST_FEATURES
E_SLOW_FEATURES = (
    "eus_slow_modulation_power_0p017_0p133", "eus_slow_modulation_depth", "eus_slow_periodicity",
)
E2_FEATURES = M2_FEATURES + E_SLOW_FEATURES
COUPLING_FEATURES = ("causal_pressure_eus_corr", "eus_activation_latency_s", "pressure_eus_coactivation")
FUSION_FEATURES = ("S_P", "S_E") + COUPLING_FEATURES

# P-EARLY is a registered replacement for V4 P2.  In particular it uses a
# fixed trailing one-second variability window and never uses the event-age
# dependent ``p_local_variability`` field.
P_EARLY_FEATURES = P0_FEATURES + (
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt",
    "p_positive_dpdt_occupancy", "p_auc", "p_auc_growth",
    "pressure_curvature", "peak_to_current_drop",
    "p_trailing_variability_1s",
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
    "pressure_spectral_entropy",
)
E_EARLY_FEATURES = M2_FEATURES
M3_FEATURES = ("S_P", "S_E") + COUPLING_FEATURES
M4_FEATURES = E_FAST_FEATURES

ROOT = Path(__file__).resolve().parents[2]
V4_ROOT = ROOT / "data" / "NVC_V4"
V31_ROOT = ROOT / "data" / "NVC_V3_1"
REFERENCE_338_ROOT = ROOT / "data" / "DSD_nvc_results"
OUTPUT_ROOT = ROOT / "data" / "NVC_V5"

MODEL_FEATURES = {
    "I-P0": P0_FEATURES,
    "I-P1": P1_FEATURES,
    "I-M1": M1_FEATURES,
    "I-E0": E0_FEATURES,
    "I-M2": M2_FEATURES,
    "I-M3": FUSION_FEATURES,
    "I-M4": FUSION_FEATURES,
}

MODEL_DESCRIPTIONS = {
    "I-P0": "individual pressure anchor",
    "I-P1": "individual pressure morphology/dynamics",
    "I-M1": "individual V4-compatible pressure NVC",
    "I-E0": "individual EUS time-domain",
    "I-M2": "individual EUS fast-spectrum",
    "I-M3": "individual PE late-fusion logistic regression",
    "I-M4": "individual PE shrinkage LDA",
    "I-M1-HC": "individual high-coverage pressure model",
}

FORBIDDEN_FEATURE_TOKENS = (
    "teacher", "target", "subject", "dataset", "cycle", "event", "urine",
    "volume", "future", "void", "recovery", "outer",
)


def assert_safe_feature_schema(names):
    bad = [str(n) for n in names if any(tok in str(n).casefold() for tok in FORBIDDEN_FEATURE_TOKENS)]
    if bad:
        raise AssertionError(f"forbidden evaluation/identity feature fields: {bad}")
