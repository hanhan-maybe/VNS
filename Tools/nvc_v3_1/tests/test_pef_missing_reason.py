import pandas as pd

from .conftest import OUTPUT


def test_pef_missing_reason():
    frame = pd.read_csv(OUTPUT / "pef_missingness_audit.csv")
    allowed = {"INSUFFICIENT_HISTORY", "WINDOW_EDGE", "MISSING_DP", "MISSING_EUS",
               "FILTER_EDGE", "SPECTRAL_WINDOW_TOO_SHORT", "NONFINITE_FEATURE", "OTHER"}
    observed = set(frame.loc[~frame["PEF_scorable"].astype(bool), "missing_reason"])
    assert observed <= allowed
