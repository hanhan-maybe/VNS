import pandas as pd

def test_challenges_are_separate_from_training():
    tr=pd.read_csv("data/NVC_V4/v4_training_samples.csv")
    ch=pd.read_csv("data/NVC_V4/v4_challenge_samples.csv")
    assert "PREVOID_PROGRESSIVE" not in set(tr.teacher_label)
    assert set(ch.challenge_type.unique()) <= {"PREVOID_CHALLENGE","VOID_CHALLENGE"}
