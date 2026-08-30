from ..config import FUSION_FEATURES, M3_FEATURES, M4_FEATURES

def test_fusion_schema_same_for_lr_and_lda():
    assert tuple(M3_FEATURES)==tuple(M4_FEATURES)==tuple(FUSION_FEATURES)
