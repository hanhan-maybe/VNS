import numpy as np

from ..config import MODEL_FEATURES, SUBJECTS
from ..validation import fit_pipeline, prepare_frame


def test_scaler_train_only(synthetic_features):
    train_subjects = SUBJECTS[:-1]
    frame = prepare_frame(synthetic_features, "PE", 0.5, train_subjects)
    model = fit_pipeline(frame, "PE", train_subjects)
    transformed = model.named_steps["imputer"].transform(frame[list(MODEL_FEATURES["PE"])])
    assert np.allclose(model.named_steps["scaler"].mean_, transformed.mean(axis=0))
    assert SUBJECTS[-1] not in model.fit_subjects_
