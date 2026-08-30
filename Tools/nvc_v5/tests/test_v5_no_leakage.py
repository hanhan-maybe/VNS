from .. import config as C
from ..data_adapter import build_v5_dataset


def test_individual_rows_are_disjoint_by_cycle():
    train, _, _, _, _, _ = build_v5_dataset()
    for subject in C.SUBJECTS:
        tr = set(C.SPLITS[subject]["train"])
        te = set(C.SPLITS[subject]["test"])
        assert tr.isdisjoint(te)
        assert not set(train[(train.subject == subject) & train.cycle_id.astype(str).isin(te) & train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].cycle_id).intersection(tr)

