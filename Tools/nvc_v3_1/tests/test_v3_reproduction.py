import pandas as pd

from .conftest import OUTPUT


def test_v3_reproduction():
    frame = pd.read_csv(OUTPUT / "baseline_v3_reproduction.csv")
    assert frame["match"].astype(bool).all()
    assert set(frame["model"]) == {"C0", "P", "PE", "PE_SPECTRAL_COMMON", "PEF"}
