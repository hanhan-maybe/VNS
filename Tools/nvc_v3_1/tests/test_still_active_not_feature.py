from ..config import MODEL_FEATURES


def test_still_active_not_feature():
    assert all("still_active" not in features for features in MODEL_FEATURES.values())
