import pandas as pd

def test_stable_sampling_ratio_and_audit():
    d=pd.read_csv("data/NVC_V4/v4_training_samples.csv")
    a=pd.read_csv("data/NVC_V4/v4_stable_windows.csv")
    assert len(a)==int((d.teacher_label=="STABLE_FILLING").sum())
    assert a.matched_nvc_event_uid.notna().all()
