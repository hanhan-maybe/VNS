from ..config import M3_FEATURES, M5_FEATURES

def test_m3_m5_same_features():
    assert tuple(M3_FEATURES) == tuple(M5_FEATURES)
