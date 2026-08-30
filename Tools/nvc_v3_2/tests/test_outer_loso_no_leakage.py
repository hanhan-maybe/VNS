import pandas as pd
from ..config import SUBJECTS

def test_outer_loso_no_leakage():
    audit = pd.read_csv("data/NVC_V3_2/v32_outer_fold_audit.csv")
    for r in audit.itertuples():
        assert str(r.outer_held_out_animal) not in str(r.outer_training_animals).split("+")
        assert r.leakage is False
