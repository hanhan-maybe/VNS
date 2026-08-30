import pandas as pd

def test_v31_frozen_reproduction_still_passes():
    d=pd.read_csv("data/NVC_V3_1/baseline_v3_reproduction.csv")
    assert d.match.astype(bool).all()
