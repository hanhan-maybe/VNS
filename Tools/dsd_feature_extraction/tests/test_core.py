from pathlib import Path
import json

import numpy as np
import pandas as pd

from Tools.dsd_feature_extraction import config as C
from Tools.dsd_feature_extraction.data_io import validate_manifest
from Tools.dsd_feature_extraction.data_io import load_cycle, load_native_volume
from Tools.dsd_feature_extraction.detectors import (AdaptiveHistory, adaptive_local_pressure_events,
                                                    adaptive_threshold_from_history, associate_adaptive_labels)
from Tools.dsd_feature_extraction.features import decision_feature_at_index, feature_at_index
from Tools.dsd_feature_extraction.models import (cross_predictions, feature_columns, fit_logistic, model_frame,
                                                 replay_trigger_mask, select_safety_threshold)
from Tools.sparc338_pre_stim_extract import resolve_subject_provenance
from Tools.sparc338_smrx_reader import match_channels


ROOT = Path(__file__).resolve().parents[3]


def test_frozen_cohort_counts():
    rows = validate_manifest(ROOT / "data" / "DSD_cycles")
    assert rows.groupby("subject").size().to_dict() == C.SUBJECT_CYCLES
    assert "STxF21" not in set(rows.subject)


def test_current_teacher_labels_are_frozen_and_quiet_is_not_a_training_event():
    teacher = pd.read_csv(ROOT / "data" / "DSD_nvc_results" / "teacher_labels.csv")
    counts = teacher[teacher.sample_type == "EVENT"].teacher_label.value_counts().to_dict()
    assert counts.get("NVC_CORE", 0) == 10 and counts.get("PREVOID_PROGRESSIVE", 0) == 30
    assert counts.get("GREY_ZONE", 0) == 1 and counts.get("INVALID", 0) == 4
    quiet = pd.DataFrame([{"subject": "STxF26", "teacher_label": "QUIET_STORAGE", "x": 1.0}])
    assert model_frame(quiet).empty


def test_feature_whitelist_has_no_identity_or_urine_fields():
    forbidden = ("subject", "cycle", "event", "urine", "volume", "void", "label", "adaptive_confirm")
    for model in ("M1", "M2"):
        assert all(not any(token in col.lower() for token in forbidden) for col in feature_columns(model))


def test_post_confirm_changes_do_not_change_features():
    n, i = 5000, 3500
    t = np.arange(n) / C.DP_FS_HZ
    delta = 0.2 * np.sin(t) + np.maximum(0, (t - 33.0) * 4)
    env = 1.0 + 0.05 * np.sin(2 * t)
    valid = np.ones(n, dtype=bool)
    cycle = {}
    a = feature_at_index(cycle, delta, env, valid, i)
    delta[i + 1:] = 1e9; env[i + 1:] = -1e9
    b = feature_at_index(cycle, delta, env, valid, i)
    assert a == b and a is not None
    model = fit_logistic(_synthetic_training(), "M2", ["STxF26", "STxF27"])
    adaptive = {"adaptive_confirm": np.full(n, 3.68), "adaptive_start": np.full(n, 2.21), "sigma_dpdt": np.ones(n)}
    da = decision_feature_at_index({}, delta, env, valid, adaptive, i, {"start_index": i - 200, "local_trough_index": i - 200})
    delta[i + 1:] = 1e9; env[i + 1:] = -1e9
    db = decision_feature_at_index({}, delta, env, valid, adaptive, i, {"start_index": i - 200, "local_trough_index": i - 200})
    pa = model.predict_proba(pd.DataFrame([{k: da[k] for k in feature_columns("M2")}]))[:, 1]
    pb = model.predict_proba(pd.DataFrame([{k: db[k] for k in feature_columns("M2")}]))[:, 1]
    assert np.array_equal(pa, pb)


def test_adaptive_threshold_bounds_and_frozen_ratios():
    low = adaptive_threshold_from_history(AdaptiveHistory(), .01, 1.0)
    high = adaptive_threshold_from_history(AdaptiveHistory(), 100.0, 1.0)
    assert C.ADAPTIVE_CONFIRM_BOUNDS_MMHG[0] <= low["confirm"] <= C.ADAPTIVE_CONFIRM_BOUNDS_MMHG[1]
    assert C.ADAPTIVE_CONFIRM_BOUNDS_MMHG[0] <= high["confirm"] <= C.ADAPTIVE_CONFIRM_BOUNDS_MMHG[1]
    assert np.isclose(high["start"], 2.208, atol=.01)
    assert np.isclose(high["recovery"], 1.472, atol=.01)


def test_abstain_all_is_not_a_safety_pass_and_delays_do_not_duplicate_events():
    frame = pd.DataFrame([{"subject": "STxF26", "event_id": "E1", "teacher_label": "NVC_CORE", "p_void_risk": 1.0, "decision_eligible": 1},
                          {"subject": "STxF26", "event_id": "E2", "teacher_label": "PREVOID_PROGRESSIVE", "p_void_risk": 1.0, "decision_eligible": 1}])
    threshold, info = select_safety_threshold(frame)
    assert threshold == 0.0 and info["ABSTAIN_ALL"] and not info["admission"]
    delays = pd.DataFrame([{"event_id": "E1", "decision_delay_s": x} for x in C.DECISION_DELAYS_S])
    assert delays.event_id.nunique() == 1 and len(delays) == len(C.DECISION_DELAYS_S)


def test_volume_and_teacher_metadata_cannot_change_causal_thresholds():
    n = 7000; t = np.arange(n) / 100.0
    p = 10 + .05 * np.sin(t) + np.where((t > 55) & (t < 60), 2.0, 0.0)
    base = {"bladder_pressure_mmHg": p, "cmg_valid_100hz": np.ones(n, bool)}
    changed = {**base, "urine_output_auxiliary_100hz": np.full(n, 1e9), "teacher_label": "VOID_CONFIRMED"}
    _, _, a = adaptive_local_pressure_events(base, AdaptiveHistory(), .2, 1.0)
    _, _, b = adaptive_local_pressure_events(changed, AdaptiveHistory(), .2, 1.0)
    assert np.array_equal(a["adaptive_confirm"], b["adaptive_confirm"])


def _synthetic_training():
    rows = []
    for subject, shift in [("STxF26", 0.0), ("STxF27", .1)]:
        for j, label in enumerate(["NVC_CORE", "NVC_CORE", "PREVOID_PROGRESSIVE", "VOID_CONFIRMED"]):
            row = {"subject": subject, "cycle_id": "B01", "event_id": f"E{j}", "teacher_label": label,
                   "sample_type": "EVENT", "data_valid": True, "confirm_time_s": 30 + j * 20,
                   "recovery_confirm_s": 35 + j * 20, "cycle_duration_s": 100.0, "local_prominence_mmHg": 5.0}
            for k, col in enumerate(C.PRESSURE_FEATURES + C.EUS_FEATURES):
                row[col] = float((label == "NVC_CORE") * 2 + shift + k * .01 + j * .001)
            rows.append(row)
    return pd.DataFrame(rows)


def test_held_out_subject_never_enters_fit_scaler_or_model():
    frame = _synthetic_training()
    model = fit_logistic(frame[frame.subject == "STxF26"], "M1", ["STxF26"])
    assert model.fit_subjects_ == ("STxF26",)
    assert "STxF27" not in model.fit_subjects_
    oof = cross_predictions(frame, ["STxF26", "STxF27"], "M1")
    assert set(oof.inner_held_subject) == {"STxF26", "STxF27"}
    assert "STxF29" not in set(oof.subject)


def test_each_event_at_most_one_trigger_and_invalid_never_triggers():
    frame = _synthetic_training().iloc[:2].copy()
    frame["probability"] = 1.0
    frame.iloc[1, frame.columns.get_loc("data_valid")] = False
    trigger = replay_trigger_mask(frame, "probability", .5)
    assert trigger.sum() == 1
    assert not trigger[1]
    assert frame.loc[trigger, "event_id"].nunique() == trigger.sum()


def test_recovered_nvc_is_not_retroactively_matched_to_later_urine():
    t = np.arange(1000) / 100.0
    event = {"start_index": 100, "end_index": 400, "confirm_index": 150, "peak_index": 200,
             "recovery_start_index": 300, "recovery_confirm_index": 400, "recovered": True,
             "locally_recovered": True, "data_invalid": False, "local_prominence_mmHg": 5.0,
             "detection_level": "MAIN"}
    urine = [{"subject": "STxF26", "cycle_id": "B01", "urine_event_id": "U1", "onset_s": 5.0, "offset_s": 6.0}]
    labels, cycle_valid, _ = associate_adaptive_labels("STxF26", "B01", [event], urine, t)
    assert labels[0][1] is None
    assert not cycle_valid


def test_stage_a_catalog_uid_counts_and_cross_subject_duplicates():
    catalog = pd.read_csv(ROOT / "data" / "DSD_nvc_results" / "dsd_reference_event_catalog.csv")
    assert catalog.event_uid.is_unique
    assert catalog.teacher_label.value_counts().to_dict() == {"PREVOID_PROGRESSIVE": 30, "NVC_CORE": 10, "INVALID": 4, "GREY_ZONE": 1}
    assert len(catalog[catalog.teacher_label == "NVC_CORE"]) == 10
    assert len(catalog[catalog.event_id == "B05_L001"]) > 1
    assert catalog[catalog.event_id == "B05_L001"].event_uid.nunique() == len(catalog[catalog.event_id == "B05_L001"])
    assert set(catalog.subject) == set(C.SUBJECT_CYCLES)
    assert "STxF30" not in set(catalog.subject)


def test_stage_a_duration_definitions_and_outlier_flag():
    c = pd.read_csv(ROOT / "data" / "DSD_nvc_results" / "dsd_reference_event_catalog.csv")
    d = pd.read_csv(ROOT / "data" / "DSD_nvc_results" / "nvc_duration_summary.csv")
    n = c[c.teacher_label == "NVC_CORE"].copy()
    expected = {
        "candidate_to_recovery_s": n.recovery_confirm_s - n.candidate_start_s,
        "confirm_to_recovery_s": n.recovery_confirm_s - n.confirm_time_s,
        "rise_to_peak_s": n.local_peak_time_s - n.candidate_start_s,
        "peak_to_recovery_s": n.recovery_confirm_s - n.local_peak_time_s,
    }
    events = d[d.summary_level == "EVENT"].set_index("event_uid")
    for name, values in expected.items():
        assert (values >= 0).all()
        assert np.allclose(events.loc[n.event_uid, name].to_numpy(), values.to_numpy())
        assert name + "_iqr" in d.columns
    assert d.summary_level.isin(["SUBJECT", "MACRO_SUBJECT_BALANCED", "ALL_EVENTS"]).any()
    assert bool(d.duration_outlier_flag.any())


def test_stage_a_schema_ranges_loso_and_external_protocol():
    out = ROOT / "data" / "DSD_nvc_results"
    schema = json.loads((out / "feature_schema.json").read_text(encoding="utf-8"))
    assert schema["causal_feature_order"] == C.PRESSURE_FEATURES
    assert not any(any(x in f.lower() for x in ("subject", "volume", "urine", "label")) for f in schema["causal_feature_order"])
    ranges = pd.read_csv(out / "dsd_reference_feature_ranges.csv")
    assert set(C.SUBJECT_CYCLES) | {"MACRO_SUBJECT_BALANCED"} <= set(ranges.subject)
    metrics = pd.read_csv(out / "decision_time_metrics.csv")
    for _, row in metrics.iterrows():
        assert set(row.train_subjects.split("+")) == set(C.SUBJECT_CYCLES) - {row.held_out_subject}
        assert row.test_event_uid_count == row.n_events
    protocol = json.loads((out / "external_validation_protocol.json").read_text(encoding="utf-8"))
    assert protocol["read_data_in_this_stage"] is False
    readiness = json.loads((out / "pre_f30_readiness.json").read_text(encoding="utf-8"))
    assert readiness["readiness_status"] == "READY_TO_EXTRACT_STxF30_CYCLES"
    assert json.loads((out / "dsd_stage_a_freeze.json").read_text(encoding="utf-8"))["stimulation_enabled"] is False


def test_subject_adaptive_outputs_have_common_contract():
    cycles = ROOT / "data" / "STxF30_cycles"
    results = ROOT / "data" / "STxF30_nvc_results"
    manifest = pd.read_csv(cycles / "cycle_manifest.csv")
    extraction = json.loads((cycles / "cycle_extraction_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((results / "f30_external_validation_summary.json").read_text(encoding="utf-8"))
    assert len(manifest) == 11 and extraction["assigned_volume_event_count"] == 11
    assert extraction["all_samples_strictly_pre_stim"] is True
    assert validation["external_validation_only"] is True and validation["stimulation_enabled"] is False
    assert validation["causal_feature_order"] == C.PRESSURE_FEATURES


def test_global_volume_events_are_conserved_and_unique_assignment():
    cycles = ROOT / "data" / "STxF30_cycles"
    events = pd.read_csv(cycles / "all_volume_events.csv")
    assignment = pd.read_csv(cycles / "volume_event_cycle_assignment.csv")
    assert events.urine_event_id.is_unique
    assert assignment.urine_event_id.is_unique
    assert len(assignment[assignment.cycle_id.notna() & assignment.cycle_id.astype(str).ne("")]) == 11


def test_generic_causal_features_have_no_future_or_label_fields():
    causal = pd.read_csv(ROOT / "data" / "STxF30_nvc_results" / "f30_causal_features.csv")
    assert list(causal.columns[3:]) == C.PRESSURE_FEATURES
    assert not any(any(token in c.lower() for token in ("volume", "urine", "label", "recovery", "future")) for c in causal.columns[3:])


def test_external_dataset_provenance_does_not_extend_frozen_registry():
    external = resolve_subject_provenance("STxF33", "164")
    assert external == {
        "source_dataset_id": "164", "local_subject_id": "STxF33",
        "dsd_confirmed": None, "urine_source": "AUTO_DISCOVER_FROM_SMRX",
        "urine_review_status": "AUTOMATIC_ONLY", "manual_review_used": False,
    }
    with np.testing.assert_raises(KeyError):
        resolve_subject_provenance("STxF33", "338")


def test_equal_score_required_channel_candidates_fail_closed():
    rows = [
        {"subject": "STxF33", "channel": 1, "type": "Adc", "title": "CMG PRES", "units": "mmHg", "sample_rate_hz": 1000, "divide_ticks": 1},
        {"subject": "STxF33", "channel": 2, "type": "Adc", "title": "CMG PRES", "units": "mmHg", "sample_rate_hz": 1000, "divide_ticks": 1},
        {"subject": "STxF33", "channel": 3, "type": "Adc", "title": "EUS", "units": "mV", "sample_rate_hz": 1000, "divide_ticks": 1},
        {"subject": "STxF33", "channel": 4, "type": "EventRise", "title": "STIM", "units": "", "sample_rate_hz": 1, "divide_ticks": 1},
    ]
    selected, warnings = match_channels(rows)
    assert selected["BLADDER"] is None
    assert any("Ambiguous BLADDER" in warning for warning in warnings)
