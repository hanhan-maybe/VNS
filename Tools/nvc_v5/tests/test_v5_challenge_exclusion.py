from .. import config as C
from ..data_adapter import build_v5_dataset


def test_challenges_are_not_primary_training_rows():
    train, challenges, _, _, _, _ = build_v5_dataset()
    assert set(train.teacher_label.unique()).issubset(set(C.PRIMARY_TRAIN_LABELS))
    assert set(challenges.challenge_type.unique()).issubset({"PREVOID_CHALLENGE", "VOID_CHALLENGE"})

