import numpy as np
import pandas as pd

from ..validation import per_animal_metrics


def test_missing_metric_is_na():
    frame = pd.DataFrame({
        "subject": ["A"], "dataset": ["x"], "teacher_label": ["PREVOID_PROGRESSIVE"],
        "target": [0], "p_nvc": [0.2], "predicted_nvc": [False], "actionable_hit": [False],
        "actionable": [False],
    })
    result = per_animal_metrics(frame, "X").iloc[0]
    assert np.isnan(result.frozen_sensitivity)
    assert np.isnan(result.scorable_sensitivity)
    assert np.isnan(result.AUROC)
