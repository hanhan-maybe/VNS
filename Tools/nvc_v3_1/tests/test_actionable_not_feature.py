from ..config import MODEL_FEATURES


def test_actionable_not_feature():
    assert all("actionable" not in features for features in MODEL_FEATURES.values())
