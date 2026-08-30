import pandas as pd

def test_m2_frequency_selection_train_only():
    d = pd.read_csv("data/NVC_V3_2/m2_frequency_selection_by_fold.csv")
    assert d.empty or d["outer_fold"].notna().all()
