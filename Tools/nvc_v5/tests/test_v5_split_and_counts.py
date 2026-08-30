from .. import config as C
from ..data_adapter import build_v5_dataset


def test_fixed_splits_and_nvc_counts():
    train, challenges, _, cycles, _, _ = build_v5_dataset()
    assert C.SPLITS["STxF37"]["train"] == ("B01", "B02", "B03", "B04")
    assert C.SPLITS["STxF37"]["test"] == ("B05", "B06", "B07")
    for subject, expected_train, expected_test in (("STxF37", 5, 6), ("STxF26", 3, 3)):
        g = train[train.subject == subject]
        assert int(g[g.cycle_id.astype(str).isin(C.SPLITS[subject]["train"])].teacher_label.eq("NVC_CORE").sum()) == expected_train
        assert int(g[g.cycle_id.astype(str).isin(C.SPLITS[subject]["test"])].teacher_label.eq("NVC_CORE").sum()) == expected_test
    assert "B15" in set(train[train.subject == "STxF26"].cycle_id)
    assert int((challenges.challenge_type == "PREVOID_CHALLENGE").sum()) > 0
