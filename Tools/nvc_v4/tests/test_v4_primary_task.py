import pandas as pd

def test_v4_primary_task_excludes_outcomes():
    d=pd.read_csv("data/NVC_V4/v4_training_samples.csv")
    assert set(d.teacher_label.unique()) <= {"NVC_CORE","STABLE_FILLING"}
    assert set(d.target.unique()) <= {0,1}
