import pandas as pd

from .conftest import OUTPUT


def test_delay_selection_train_only():
    audit = pd.read_csv(OUTPUT / "nested_loso_audit.csv")
    for row in audit.itertuples(index=False):
        assert row.outer_held_out_animal not in row.inner_delay_fit_animals.split("+")
