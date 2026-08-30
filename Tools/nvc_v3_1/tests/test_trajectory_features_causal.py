import numpy as np

from ..features import assert_trajectory_causal


def test_trajectory_features_causal():
    n, index, confirm = 5000, 4000, 3800
    delta = np.sin(np.arange(n) / 80.0) + np.arange(n) * 0.001
    eus = 2.0 + 0.1 * np.sin(np.arange(n) / 30.0)
    valid = np.ones(n, dtype=bool)
    adaptive = {"sigma_dpdt": np.ones(n), "adaptive_confirm": np.full(n, 3.68)}
    event = {"confirm_index": confirm, "start_index": 3700, "local_trough_index": 3650}
    assert assert_trajectory_causal(delta, eus, valid, index, event, adaptive)
