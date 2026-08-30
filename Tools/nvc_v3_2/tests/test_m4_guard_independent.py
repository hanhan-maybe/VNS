import pandas as pd

def test_m4_guard_independent():
    d = pd.read_csv("data/NVC_V3_2/v32_outer_fold_audit.csv")
    m = d[d.model.eq("M4_EVENT_PROGRESSION_GUARD")]
    assert len(m) > 0 and m["progression_model_independent"].astype(bool).all()
    p = pd.read_csv("data/NVC_V3_2/m4_progression_predictions.csv")
    assert "p_progression" in p
