import pandas as pd

from .conftest import OUTPUT


def test_positive_class_mapping():
    audit = pd.read_csv(OUTPUT / "class_mapping_audit.csv")
    assert (audit["positive_class"] == 1).all()
    assert (audit["positive_index"] == 1).all()
