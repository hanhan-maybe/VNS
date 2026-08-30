import pandas as pd

from .conftest import OUTPUT


def test_no_future_feature():
    frame = pd.read_csv(OUTPUT / "event_features_delayed_v31.csv")
    finite = frame[frame["feature_max_time_s"].notna()]
    assert (finite["feature_max_time_s"] <= finite["decision_time_s"] + 1e-9).all()
