from ..config import common_eus_bands

def test_common_eus_bandwidth():
    high, included, excluded = common_eus_bands([10000.0, 10000.0])
    assert high == 1500.0 and len(included) == 4 and not excluded
    high, included, excluded = common_eus_bands([1000.0, 10000.0])
    assert high == 500.0 and all(hi <= high for _, hi in included)
