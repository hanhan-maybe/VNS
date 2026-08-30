import pandas as pd

def test_thresholds_fit_on_training_animals_only():
    d=pd.read_csv("data/NVC_V4/v4_outer_fold_audit.csv")
    for r in d.itertuples():
        assert str(r.outer_held_out_animal) not in str(r.threshold_fit_animals).split("+")
