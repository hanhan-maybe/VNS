import pandas as pd

def test_inner_selection_train_only():
    audit = pd.read_csv("data/NVC_V3_2/v32_outer_fold_audit.csv")
    assert audit["threshold_fit_animals"].notna().all()
    for r in audit.itertuples():
        assert str(r.outer_held_out_animal) not in str(r.threshold_fit_animals).split("+")
