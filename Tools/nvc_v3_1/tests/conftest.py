from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ..config import DELAYS_S, MODEL_FEATURES, SUBJECTS

OUTPUT = Path(__file__).resolve().parents[3] / "data" / "NVC_V3_1"


@pytest.fixture
def synthetic_features():
    rows = []
    names = sorted(set(sum((list(v) for v in MODEL_FEATURES.values()), [])))
    for delay in DELAYS_S:
        for subject_number, subject in enumerate(SUBJECTS):
            for target, label in ((0, "PREVOID_PROGRESSIVE"), (1, "NVC_CORE")):
                for repeat in range(2):
                    row = {
                        "dataset": "338" if subject_number < 3 else "164", "subject": subject,
                        "cycle_id": f"B{repeat:02d}", "event_id": f"E{target}{repeat}",
                        "event_uid": f"{subject}::{delay}::{target}::{repeat}", "teacher_label": label,
                        "decision_delay_s": delay, "decision_time_s": 20.0 + delay,
                        "feature_max_time_s": 20.0 + delay, "base_eligible": True,
                        "base_failure_reason": "", "spectral_scorable": True,
                        "still_active": target == 1, "actionable": target == 1,
                    }
                    for index, name in enumerate(names):
                        row[name] = target + subject_number * 0.01 + repeat * 0.001 + index * 1e-5
                    rows.append(row)
    return pd.DataFrame(rows)
