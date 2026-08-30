from pathlib import Path
import pandas as pd

def test_v31_reproduction_unchanged():
    p = Path(__file__).resolve().parents[3] / "data" / "NVC_V3_1" / "baseline_v3_reproduction.csv"
    d = pd.read_csv(p)
    assert not d.empty and d["match"].astype(bool).all()
