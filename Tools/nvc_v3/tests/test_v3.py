from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Tools.nvc_v3.development import (
    ALL_MODEL_FEATURES,
    MODEL_ORDER,
    SUBJECTS,
    SUBJECTS_164,
    assert_v3_paths,
    expanded_animal_class_weights,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "data" / "NVC_V3"


def test_registered_subjects_and_unchanged_feature_counts():
    assert len(SUBJECTS) == 8
    assert SUBJECTS_164 == ("STxF31", "STxF33", "STxF34", "STxF35", "STxF37")
    assert {name: len(columns) for name, columns in ALL_MODEL_FEATURES.items()} == {
        "C0": 8, "P": 6, "PE": 9, "PE_SPECTRAL_COMMON": 9, "PEF": 10}


def test_single_class_animal_keeps_equal_total_weight():
    frame = pd.DataFrame({
        "subject": ["A", "A", "B", "B", "B", "B"],
        "target": [0, 0, 0, 0, 1, 1],
    })
    weights = expanded_animal_class_weights(frame)
    totals = pd.Series(weights).groupby(frame["subject"]).sum()
    assert np.allclose(totals.to_numpy(), [0.5, 0.5])
    b_classes = pd.Series(weights[frame["subject"].eq("B")]).groupby(
        frame.loc[frame["subject"].eq("B"), "target"].reset_index(drop=True)).sum()
    assert np.allclose(b_classes.to_numpy(), [0.25, 0.25])


def test_unregistered_cohort_paths_are_rejected():
    with pytest.raises(RuntimeError):
        assert_v3_paths([Path("data/STxF30_cycles")])


def test_generated_v3_contract_when_output_exists():
    if not OUTPUT.exists():
        pytest.skip("V3 output has not been generated")
    features = pd.read_csv(OUTPUT / "event_features_v3.csv")
    predictions = pd.read_csv(OUTPUT / "event_predictions_v3.csv")
    metrics = pd.read_csv(OUTPUT / "per_animal_metrics_v3.csv")
    audit = pd.read_csv(OUTPUT / "sparc164_reconstruction_audit_v3.csv")
    assert set(features["subject"]) == set(SUBJECTS)
    assert int((features["teacher_label"] == "NVC_CORE").sum()) == 27
    assert int((features["teacher_label"] == "PREVOID_PROGRESSIVE").sum()) == 63
    assert audit["causal"].astype(bool).all()
    scored = predictions[predictions["feature_max_time_s"].notna()]
    assert (scored["feature_max_time_s"] <= scored["decision_time_s"] + 1e-9).all()
    assert len(metrics) == len(MODEL_ORDER) * len(SUBJECTS)
    assert (metrics["held_out_subject"].apply(
        lambda held: all(held != train for train in metrics.loc[
            metrics["held_out_subject"] == held, "train_subjects"].iloc[0].split("+")))).all()
