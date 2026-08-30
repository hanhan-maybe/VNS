import pandas as pd

from .conftest import OUTPUT


def test_teacher_label_frozen():
    events = pd.read_csv(OUTPUT / "source_events_v31.csv")
    counts = events["teacher_label"].value_counts().to_dict()
    assert counts["NVC_CORE"] == 27
    assert counts["PREVOID_PROGRESSIVE"] == 63
    assert counts["GREY_ZONE"] == 2
    assert counts["INVALID"] == 8
