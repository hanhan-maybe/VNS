from ..models import assert_feature_schema_safe
from ..config import M1_FEATURES, M3_FEATURES

def test_eval_only_fields_not_features():
    assert_feature_schema_safe(M1_FEATURES)
    assert_feature_schema_safe(M3_FEATURES)
