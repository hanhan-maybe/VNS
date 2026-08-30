import pandas as pd

from .conftest import OUTPUT


def test_threshold_train_only():
    audit = pd.read_csv(OUTPUT / "nested_loso_audit.csv")
    for row in audit.itertuples(index=False):
        assert row.outer_held_out_animal not in row.threshold_fit_animals.split("+")
