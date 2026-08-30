from ..data_adapter import build_v5_dataset
from ..development import choose_high_coverage_features


def test_high_coverage_pressure_selection_uses_calibration_only():
    train, _, _, _, _, _ = build_v5_dataset()
    for subject in ("STxF37", "STxF26"):
        g = train[(train.subject == subject) & train.cycle_id.astype(str).isin(("B01", "B02", "B03", "B04") if subject == "STxF37" else tuple(f"B{i:02d}" for i in range(1, 14)))]
        selected, audit = choose_high_coverage_features(g)
        assert "p_local_variability" not in selected
        assert audit.iloc[-1].calibration_nvc_coverage >= 0.9

