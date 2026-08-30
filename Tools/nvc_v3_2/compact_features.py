"""Pre-registered low-dimensional multimodal feature schema.

M3 and M5 intentionally import the same tuple from this module; no output of
M1 or M2 is consulted when constructing it.
"""
from .config import (
    P_FEATURES, TRAJECTORY_COMPACT_FEATURES, PRESSURE_SPECTRAL_FEATURES,
    EUS_COMPACT_FEATURES, COUPLING_FEATURE, M3_FEATURES, M5_FEATURES,
)

PREREGISTERED_M3_FEATURES = tuple(M3_FEATURES)
PREREGISTERED_M5_FEATURES = tuple(M5_FEATURES)

def compact_feature_names():
    return tuple(PREREGISTERED_M3_FEATURES)

def assert_m3_m5_same_schema():
    assert tuple(PREREGISTERED_M3_FEATURES) == tuple(PREREGISTERED_M5_FEATURES)
    return True
