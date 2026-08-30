import numpy as np
from ..spectral_features import causal_pressure_spectral_features

def test_no_future_spectral_feature():
    x = np.sin(np.arange(5000) / 17.0)
    event = {"start_index": 3000, "confirm_index": 3100}
    a, _ = causal_pressure_spectral_features(x, 3200, event)
    x[3201:] = 1e9
    b, _ = causal_pressure_spectral_features(x, 3200, event)
    assert a == b
