import pandas as pd

def test_v4_outer_loso_no_animal_leakage():
    d=pd.read_csv("data/NVC_V4/v4_outer_fold_audit.csv")
    for r in d.itertuples():
        assert str(r.outer_held_out_animal) not in str(r.outer_training_animals).split("+")
        assert bool(r.leakage) is False
