"""Frozen scientific and streaming configuration.  Values are never LOSO-tuned."""

SUBJECT_CYCLES = {"STxF26": 16, "STxF27": 8, "STxF29": 8}

DP_FS_HZ = 100.0
BASELINE_WINDOW_S = 25.0
FEATURE_WINDOW_S = 2.0
UPDATE_STEP_S = 0.25
DECISION_DELAYS_S = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
V2_DECISION_DELAYS_S = [0.00, 0.50, 1.00]
CANDIDATE_THRESHOLD_MMHG = 2.21
CONFIRM_THRESHOLD_MMHG = 3.68
RECOVERY_THRESHOLD_MMHG = 1.47
CANDIDATE_HOLD_S = 0.20
CONFIRM_HOLD_S = 0.5
RECOVERY_HOLD_S = 1.0
POST_EVENT_LOCKOUT_S = 15.0
QUIET_EXCLUSION_S = 5.0
QUIET_TO_NVC_RATIO = 2.0
RANDOM_SEED = 338

# Unified subject-adaptive pressure formula (never subject-tuned).
ADAPTIVE_SIGMA_MULTIPLIER = 4.0
ADAPTIVE_Q_PERCENTILE = 99.5
ADAPTIVE_CONFIRM_BOUNDS_MMHG = (1.47, 3.68)
ADAPTIVE_START_RATIO = 0.60
ADAPTIVE_RECOVERY_RATIO = 0.40
LOCAL_RECOVERY_FRACTION = 0.60
EXPLORATORY_SIGMA_MULTIPLIER = 3.0
EXPLORATORY_Q_PERCENTILE = 99.0
ADAPTIVE_HISTORY_MAX_S = 300.0
PRESSURE_JUMP_LIMIT_MMHG_S = 50.0

# Fixed native-Volume engineering parameters.  These govern teacher evidence only.
URINE_SMOOTH_S = 0.25
URINE_STEP_LOOKBACK_S = 10.0
URINE_STEP_HOLD_S = 0.50
URINE_STEP_SUSTAIN_RATIO = 0.50
URINE_ONSET_BACKTRACK_RATIO = 0.10
URINE_STEP_MERGE_S = 8.0
URINE_EVENT_TAIL_S = 1.0

# Existing project preprocessing used a causal 50-500 Hz EUS band and 20 Hz envelope.
EUS_BANDPASS_HZ = (50.0, 500.0)
EUS_NOTCH_HZ = 50.0
EUS_NOTCH_Q = 30.0
EUS_ENVELOPE_LP_HZ = 20.0
EUS_FILTER_ORDER = 4
EUS_LOCAL_BASELINE_S = 25.0
EUS_MIN_BASELINE_S = 10.0
TONIC_MAD_MULTIPLIER = 3.0

CMG_VALID_RANGE_MMHG = (-50.0, 100.0)
GUARD_PROGRESSIVE_S = 2.0

PRESSURE_FEATURES = [
    "current_delta_p_norm", "peak_delta_p_to_now_norm", "mean_dpdt_0p25s_norm",
    "mean_dpdt_0p5s_norm", "dpdt_change", "peak_to_current_drop_norm",
    "negative_dpdt_occupancy_0p5s", "pressure_auc_growth_0p5s_norm",
]
EUS_FEATURES = [
    "eus_tonic_occupancy", "eus_envelope_slope",
]

COMPLETE_EVENT_FEATURES = [
    "local_prominence_mmHg", "peak_delta_p_mmHg", "candidate_to_recovery_s",
    "confirm_to_recovery_s", "rise_to_peak_s", "peak_to_recovery_s",
    "pressure_auc", "recovery_fraction", "fall_from_peak_mmHg",
    "nvc_frequency_per_hour", "nvc_count_per_valid_cycle",
]
CAUSAL_FEATURE_ORDER = list(PRESSURE_FEATURES)
EXPLORATORY_EUS_FEATURES = ["eus_tonic_occupancy", "eus_envelope_slope"]

V2_TIME_FREQUENCY_FEATURES = [
    "pressure_bandpower_0p2_20_ratio_5s",
    "pressure_low_band_fraction_0p2_0p6_ratio_5s",
    "pressure_high_band_fraction_5_20_ratio_5s",
    "eus_burst_band_fraction_3_9_ratio_5s",
    "eus_dpdt_correlation_1s",
]
V2_CONFIG_FEATURES = {
    "C0": list(PRESSURE_FEATURES),
    "C1": list(PRESSURE_FEATURES),
    "C2": list(PRESSURE_FEATURES) + list(EUS_FEATURES),
    "C3": list(PRESSURE_FEATURES) + list(EUS_FEATURES) + list(V2_TIME_FREQUENCY_FEATURES),
}
V2_CONFIG_COMPLEXITY_ORDER = {"C1": 0, "C2": 1, "C3": 2}

MODEL_CONFIG = {
    "M0": {"kind": "fixed_pressure_rule", "confirm_threshold_mmhg": CONFIRM_THRESHOLD_MMHG},
    "M0A": {"kind": "adaptive_pressure_rule", "formula": "max(4*MAD_sigma,q99.5), clipped 1.47-3.68"},
    "M1": {"kind": "standard_scaler_l2_logistic", "C": 1.0, "features": PRESSURE_FEATURES},
    "M2": {"kind": "standard_scaler_l2_logistic", "C": 1.0, "features": PRESSURE_FEATURES + EUS_FEATURES},
}
