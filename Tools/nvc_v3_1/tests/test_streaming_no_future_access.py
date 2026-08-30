import pandas as pd

from .conftest import OUTPUT


def test_streaming_no_future_access():
    replay = pd.read_csv(OUTPUT / "streaming_replay_v31.csv")
    finite = replay[replay["feature_max_time_s"].notna()]
    assert (finite["feature_max_time_s"] <= finite["decision_time_s"] + 1e-9).all()
    assert not replay["lockout_policy"].isna().any()
