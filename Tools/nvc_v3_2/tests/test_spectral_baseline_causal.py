import numpy as np
from ..spectral_features import causal_pressure_spectral_features

def test_spectral_baseline_causal():
    x = np.random.default_rng(1).normal(size=6000)
    out, reason = causal_pressure_spectral_features(x, 4000, {"start_index": 3600})
    assert reason == "" and out["pressure_spec_baseline_end_s"] <= 36.0
