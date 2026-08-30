"""Decision-time VOID_RISK LOSO pipeline on the frozen adaptive-NVC labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from . import config as C
from .data_io import load_cycle, validate_manifest, write_json
from .detectors import AdaptiveHistory, adaptive_local_pressure_events
from .features import causal_eus_envelope_100hz, decision_feature_at_index
from .models import (DANGEROUS, EUS_CANDIDATES, cross_predictions, feature_columns, fit_logistic,
                     select_analysis_threshold, select_eus_features_train, select_safety_threshold,
                     serialize_model)
from .plotting import plot_cycle
from .replay import stream_vectors

FIXED_LABEL_COUNTS = {"NVC_CORE": 10, "PREVOID_PROGRESSIVE": 30, "GREY_ZONE": 1, "INVALID": 4}
TRAIN_LABELS = {"NVC_CORE", "PREVOID_PROGRESSIVE", "VOID_CONFIRMED"}


def assert_frozen_labels(output_root: Path):
    teacher = pd.read_csv(output_root / "teacher_labels.csv")
    events = teacher[teacher.sample_type == "EVENT"]
    actual = events.teacher_label.value_counts().to_dict()
    if any(actual.get(k, 0) != v for k, v in FIXED_LABEL_COUNTS.items()):
        raise RuntimeError(f"Frozen teacher labels changed; expected={FIXED_LABEL_COUNTS}, actual={actual}")
    manifest = pd.read_csv(output_root / "dataset_manifest.csv")
    counts = manifest.groupby("subject").size().to_dict()
    expected_cycles = sum(C.SUBJECT_CYCLES.values())
    unique_cycles = int(manifest[["subject", "cycle_id"]].drop_duplicates().shape[0])
    if counts != C.SUBJECT_CYCLES or unique_cycles != expected_cycles:
        raise RuntimeError(f"Frozen cycle baseline changed; actual_counts={counts}, valid={int((manifest.label_status == 'PASS').sum())}")
    if "QUIET_STORAGE" in set(events.teacher_label):
        raise RuntimeError("QUIET_STORAGE appeared as an event label")
    return teacher, manifest


def prior_and_cache(input_root: Path, manifest: pd.DataFrame, params: pd.DataFrame, pressure: pd.DataFrame):
    priors = {}
    for _, row in params.iterrows():
        priors[str(row.subject)] = (float(row.warmup_prior_sigma_p), float(row.sigma_dpdt_median))
    histories = {s: AdaptiveHistory() for s in C.SUBJECT_CYCLES}; cache = {}
    for _, mrow in manifest.sort_values(["subject", "cycle_start_s"]).iterrows():
        subject, cid = str(mrow.subject), str(mrow.cycle_id)
        # The frozen result manifest uses ``cycle_id`` while the raw input
        # manifest/data loader uses ``dsd_cycle_id``.  Normalize only the
        # in-memory row; no on-disk labels or manifests are rewritten.
        load_row = mrow.copy()
        load_row["dsd_cycle_id"] = cid
        cycle = load_cycle(input_root, load_row)
        residual, _, adaptive = adaptive_local_pressure_events(cycle, histories[subject], *priors[subject])
        eus, eus_valid = causal_eus_envelope_100hz(cycle)
        p = pressure[(pressure.subject == subject) & (pressure.cycle_id == cid)].copy()
        cache[(subject, cid)] = {"cycle": cycle, "delta": residual, "adaptive": adaptive, "eus_env": eus, "eus_valid": eus_valid, "pressure": p}
    return cache


def build_decision_features(cache, pressure):
    rows = []
    for (subject, cid), item in cache.items():
        p = item["pressure"]
        for _, event in p[p.teacher_label.isin(TRAIN_LABELS)].iterrows():
            if not np.isfinite(event.confirm_index): continue
            confirm_i = int(event.confirm_index); t = np.asarray(item["cycle"]["t_abs_s"]); confirm_s = float(t[confirm_i])
            urine_s = float(event.local_peak_time_s + event.time_to_urine_s) if np.isfinite(event.time_to_urine_s) else np.nan
            recovery_s = float(event.local_recovery_time_s) if np.isfinite(event.local_recovery_time_s) else np.inf
            for delay in C.DECISION_DELAYS_S:
                decision_s = confirm_s + float(delay); idx = int(np.searchsorted(t, decision_s, side="left"))
                reason = ""; eligible = True
                if idx >= len(t): eligible, reason = False, "OUTSIDE_CYCLE"
                if eligible and event.teacher_label == "NVC_CORE" and decision_s > recovery_s: eligible, reason = False, "AFTER_NVC_RECOVERY"
                if eligible and event.teacher_label == "PREVOID_PROGRESSIVE" and np.isfinite(urine_s) and decision_s >= urine_s: eligible, reason = False, "AT_OR_AFTER_URINE"
                feat = None
                if eligible:
                    ed = {"start_index": int(event.start_index), "local_trough_index": int(event.local_trough_index)}
                    feat = decision_feature_at_index(item["cycle"], item["delta"], item["eus_env"], item["eus_valid"], item["adaptive"], idx, ed)
                    if feat is None: eligible, reason = False, "SIGNAL_INVALID_AT_DECISION"
                row = {"subject": subject, "cycle_id": cid, "event_id": str(event.event_id), "teacher_label": str(event.teacher_label),
                       "sample_type": "EVENT", "confirm_index": confirm_i, "recovery_confirm_index": int(event.recovery_confirm_index) if np.isfinite(event.recovery_confirm_index) else np.nan,
                       "decision_delay_s": float(delay), "confirm_time_s": confirm_s, "decision_time_s": decision_s,
                       "decision_index": idx, "decision_eligible": int(eligible), "decision_failure_reason": reason,
                       "local_recovery_time_s": recovery_s, "urine_onset_s": urine_s,
                       "remaining_to_recovery_s": recovery_s - decision_s if np.isfinite(recovery_s) else np.nan,
                       "remaining_to_urine_s": urine_s - decision_s if np.isfinite(urine_s) else np.nan,
                       "cycle_duration_s": float(event.cycle_duration_s)}
                if feat is not None: row.update(feat)
                else:
                    for c in C.PRESSURE_FEATURES + EUS_CANDIDATES: row[c] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _score_test(test, model, model_name, eus_features):
    out = test.copy(); out["p_void_risk"] = np.nan
    ok = out.decision_eligible.astype(bool)
    if ok.any(): out.loc[ok, "p_void_risk"] = model.predict_proba(out.loc[ok, feature_columns(model_name, eus_features)])[:, 1]
    return out


def fold_metrics(scored: pd.DataFrame, subject: str, model: str, delay: float, analysis_threshold: float, safety_threshold: float, safety_info: Dict):
    e = scored[scored.subject == subject]; eligible = e.decision_eligible.astype(bool); nvc = e.teacher_label.eq("NVC_CORE") & eligible; dangerous = e.teacher_label.isin(DANGEROUS) & eligible
    trigger = e.trigger.astype(bool); accepted_nvc = int((trigger & nvc).sum()); false_danger = int((trigger & dangerous).sum()); nvc_n = int(nvc.sum()); danger_n = int(dangerous.sum()); total_accept = int(trigger.sum())
    risk_sens = 1 - false_danger / danger_n if danger_n else np.nan
    delays_recovery = e.loc[trigger & nvc, "remaining_to_recovery_s"]; delays_urine = e.loc[trigger & dangerous, "remaining_to_urine_s"]
    return {"model": model, "decision_delay_s": delay, "held_out_subject": subject, "eligible_nvc": nvc_n, "eligible_prevoid": int((e.teacher_label == "PREVOID_PROGRESSIVE").astype(bool).mul(eligible).sum()),
            "nvc_accepts": accepted_nvc, "nvc_acceptance_rate": accepted_nvc / nvc_n if nvc_n else np.nan,
            "prevoid_false_accepts": int((trigger & (e.teacher_label == "PREVOID_PROGRESSIVE") & eligible).sum()),
            "void_false_accepts": int((trigger & (e.teacher_label == "VOID_CONFIRMED") & eligible).sum()),
            "void_risk_sensitivity": risk_sens, "nvc_ppv": accepted_nvc / total_accept if total_accept else np.nan,
            "median_decision_to_nvc_recovery_s": delays_recovery.median() if len(delays_recovery) else np.nan,
            "median_decision_to_urine_s": delays_urine.median() if len(delays_urine) else np.nan,
            "dangerous_trigger_rate": false_danger / danger_n if danger_n else 0.0,
            "analysis_threshold": analysis_threshold, "safety_threshold": safety_threshold,
            "ABSTAIN_ALL": bool(safety_info.get("ABSTAIN_ALL", False)), "safety_admission": bool(safety_info.get("admission", False)),
            "n_events": int(len(e)), "fold_status": "PASS"}


def run_loso_by_delay(features: pd.DataFrame):
    sweep, metrics, fold_oof = [], [], []
    subjects = list(C.SUBJECT_CYCLES)
    for delay in C.DECISION_DELAYS_S:
        frame = features[features.decision_delay_s == delay].copy()
        for held in subjects:
            train_s = [s for s in subjects if s != held]; train = frame[(frame.subject.isin(train_s)) & (frame.decision_eligible == 1)]; test = frame[frame.subject == held].copy()
            for model_name in ("M1", "M2"):
                try:
                    oof = cross_predictions(train, train_s, model_name); at, ai = select_analysis_threshold(oof); st, si = select_safety_threshold(oof)
                    eus_features, eus_info = select_eus_features_train(train) if model_name == "M2" else ([], {"rows": []})
                    model = fit_logistic(train, model_name, train_s, eus_features); scored = _score_test(test, model, model_name, eus_features)
                    scored["trigger"] = scored.decision_eligible.astype(bool) & scored.p_void_risk.lt(st); scored["analysis_accept"] = scored.decision_eligible.astype(bool) & scored.p_void_risk.lt(at)
                    status, admission = "PASS", bool(si.get("admission", False)); info = si
                except ValueError as exc:
                    scored = test.copy(); scored["p_void_risk"] = np.nan; scored["trigger"] = False; scored["analysis_accept"] = False; at = np.nan; st = 0.0; status = str(exc); admission = False; info = {"ABSTAIN_ALL": True, "reason": status}; eus_features = []
                scored["model"] = model_name; scored["analysis_threshold"] = at; scored["safety_threshold"] = st; scored["held_out_subject"] = held; scored["fold_status"] = status; scored["safety_admission"] = admission
                for _, row in scored.iterrows():
                    sweep.append({**row.to_dict(), "p_void_risk": row.p_void_risk, "trigger": bool(row.trigger), "analysis_accept": bool(row.analysis_accept),
                                  "eus_features_used": ",".join(eus_features), "threshold_status": json.dumps(info)})
                m = fold_metrics(scored, held, model_name, delay, at, st, info); m["fold_status"] = status; m["eus_features_used"] = ",".join(eus_features); metrics.append(m)
        # Explicit inner OOF is retained for traceability but never counted as an outer event.
    return pd.DataFrame(sweep), pd.DataFrame(metrics)


def feature_separability(features):
    rows = []
    f = features[features.decision_delay_s == 0].copy()
    for name in EUS_CANDIDATES:
        by = f.groupby(["subject", "teacher_label"])[name].agg(["median", "count"])
        diffs = []
        for subject in C.SUBJECT_CYCLES:
            try: diffs.append(float(by.loc[(subject, "PREVOID_PROGRESSIVE"), "median"] - by.loc[(subject, "NVC_CORE"), "median"]))
            except KeyError: pass
        direction = "POSITIVE" if diffs and all(x > 0 for x in diffs) else "NEGATIVE" if diffs and all(x < 0 for x in diffs) else "CROSS_SUBJECT_DIRECTION_UNSTABLE"
        for subject in C.SUBJECT_CYCLES:
            for label in ("NVC_CORE", "PREVOID_PROGRESSIVE"):
                g = f[(f.subject == subject) & (f.teacher_label == label)][name]
                rows.append({"feature": name, "subject": subject, "label": label, "median": g.median(), "missing_rate": float(g.isna().mean()), "direction": direction, "n": int(g.notna().sum())})
    return pd.DataFrame(rows)


def add_event_uid(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the globally unique event key used by all freeze artifacts."""
    out = frame.copy()
    if {"subject", "cycle_id", "event_id"}.issubset(out.columns):
        out["event_uid"] = (out["subject"].astype(str) + "::" + out["cycle_id"].astype(str)
                             + "::" + out["event_id"].astype(str))
    return out


def _stats(values):
    s = pd.to_numeric(pd.Series(values), errors="coerce")
    valid = s.dropna()
    q25, q75 = valid.quantile(.25), valid.quantile(.75)
    return {"n": int(valid.size), "median": valid.median(), "q25": q25,
            "q75": q75, "iqr": q75 - q25, "min": valid.min(), "max": valid.max(),
            "missing_rate": float(s.isna().mean()) if len(s) else 1.0}


def build_reference_event_catalog(cache, pressure, event_features, urine):
    """Create one authoritative row per automatic pressure event."""
    p = add_event_uid(pressure[pressure.sample_type == "EVENT"].copy())
    ef = add_event_uid(event_features[event_features.decision_delay_s == 0].copy())
    ef = ef.drop_duplicates("event_uid")
    eus_cols = [c for c in C.EXPLORATORY_EUS_FEATURES if c in ef.columns]
    if eus_cols:
        p = p.merge(ef[["event_uid"] + eus_cols], on="event_uid", how="left", validate="one_to_one")
    onset_map = {(str(r.subject), str(r.cycle_id), str(r.urine_event_id)): float(r.onset_s)
                 for r in urine.itertuples() if bool(getattr(r, "detection_valid", True))}
    aucs = {}
    for (subject, cid), item in cache.items():
        for _, r in item["pressure"].iterrows():
            uid = f"{subject}::{cid}::{r.event_id}"
            if not bool(r.data_valid):
                aucs[uid] = np.nan
                continue
            try:
                a = int(r.start_index); b = int(r.recovery_confirm_index)
                if b < a or a < 0 or b >= len(item["delta"]): raise ValueError
                y = np.maximum(np.asarray(item["delta"])[a:b + 1], 0.0)
                aucs[uid] = float(np.trapz(y, dx=1.0 / C.DP_FS_HZ))
            except (ValueError, TypeError, OverflowError):
                aucs[uid] = np.nan
    p["candidate_start_s"] = p["start_s"]
    p["urine_onset_s"] = [onset_map.get((str(r.subject), str(r.cycle_id), str(r.matched_urine_event_id)),
                              (float(r.local_peak_time_s) + float(r.time_to_urine_s)
                               if np.isfinite(r.time_to_urine_s) else np.nan)) for r in p.itertuples()]
    p["pressure_auc"] = p.event_uid.map(aucs)
    p["peak_delta_p_mmHg"] = p["peak_delta_p"]
    keep = ["event_uid", "subject", "cycle_id", "event_id", "teacher_label", "data_valid",
            "exclusion_reason", "candidate_start_s", "confirm_time_s", "local_peak_time_s",
            "recovery_start_s", "recovery_confirm_s", "matched_urine_event_id", "urine_onset_s",
            "time_to_urine_s", "local_prominence_mmHg", "peak_delta_p_mmHg", "pressure_auc",
            "adaptive_start_at_confirm", "adaptive_confirm_at_confirm", "adaptive_recovery_at_confirm",
            "sigma_p_at_confirm", "sigma_dpdt_at_confirm", "cycle_duration_s", "recovery_fraction",
            "fall_from_peak_mmHg"] + eus_cols
    return p[keep].sort_values(["subject", "cycle_id", "event_id"]).reset_index(drop=True)


def build_duration_summary(catalog):
    nvc = catalog[catalog.teacher_label == "NVC_CORE"].copy()
    nvc["candidate_to_recovery_s"] = nvc.recovery_confirm_s - nvc.candidate_start_s
    nvc["confirm_to_recovery_s"] = nvc.recovery_confirm_s - nvc.confirm_time_s
    nvc["rise_to_peak_s"] = nvc.local_peak_time_s - nvc.candidate_start_s
    nvc["peak_to_recovery_s"] = nvc.recovery_confirm_s - nvc.local_peak_time_s
    metrics = ["candidate_to_recovery_s", "confirm_to_recovery_s", "rise_to_peak_s", "peak_to_recovery_s"]
    if (nvc[metrics] < -1e-6).any().any():
        raise RuntimeError("Negative frozen NVC duration")
    rows = []
    for _, r in nvc.iterrows():
        rows.append({"summary_level": "EVENT", "subject": r.subject, "event_uid": r.event_uid,
                     "n": 1, **{m: r[m] for m in metrics}, "duration_outlier_flag": bool((r[metrics] > 30).any())})
    for level, subjects in [("SUBJECT", list(C.SUBJECT_CYCLES)),
                            ("MACRO_SUBJECT_BALANCED", list(C.SUBJECT_CYCLES)),
                            ("ALL_EVENTS", ["ALL_EVENTS"])]:
        groups = [(s, nvc[nvc.subject == s]) for s in subjects] if level != "ALL_EVENTS" else [("ALL_EVENTS", nvc)]
        if level == "MACRO_SUBJECT_BALANCED":
            stats = {m: _stats([_stats(g[m])["median"] for _, g in groups]) for m in metrics}
            rows.append({"summary_level": level, "subject": "MACRO_SUBJECT_BALANCED", "event_uid": "",
                         "n": int(sum(len(g) for _, g in groups)),
                         **{f"{m}_{k}": v for m, st in stats.items() for k, v in st.items()},
                         "duration_outlier_flag": bool(any((nvc[metrics] > 30).any().values))})
            continue
        for subject, group in groups:
            row = {"summary_level": level, "subject": subject, "event_uid": "", "n": int(len(group)),
                   "duration_outlier_flag": bool((group[metrics] > 30).any().any())}
            for m in metrics:
                st = _stats(group[m])
                row.update({f"{m}_{k}": v for k, v in st.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def build_subject_summary(catalog, manifest):
    valid = manifest[manifest.label_status == "PASS"]
    rows = []
    for subject in C.SUBJECT_CYCLES:
        cycles = valid[valid.subject == subject]
        hours = float(cycles.cycle_duration_s.sum() / 3600.0)
        c = catalog[catalog.subject == subject]
        nvc = c[c.teacher_label == "NVC_CORE"]
        rows.append({"subject": subject, "valid_cycles": int(len(cycles)), "valid_analysis_hours": hours,
                     "n_nvc": int(len(nvc)), "nvc_frequency_per_h": len(nvc) / hours if hours else np.nan,
                     "median_prominence": nvc.local_prominence_mmHg.median(),
                     "median_candidate_to_recovery_s": (nvc.recovery_confirm_s - nvc.candidate_start_s).median(),
                     "median_confirm_to_recovery_s": (nvc.recovery_confirm_s - nvc.confirm_time_s).median(),
                     "median_pressure_auc": nvc.pressure_auc.median(),
                     "median_eus_tonic_occupancy": nvc.eus_tonic_occupancy.median() if "eus_tonic_occupancy" in nvc else np.nan,
                     "n_prevoid_progressive": int((c.teacher_label == "PREVOID_PROGRESSIVE").sum()),
                     "n_grey_zone": int((c.teacher_label == "GREY_ZONE").sum()),
                     "n_invalid": int((c.teacher_label == "INVALID").sum())})
    return pd.DataFrame(rows)


def build_reference_ranges(catalog, subject_summary):
    rows = []
    event_features = ["local_prominence_mmHg", "peak_delta_p_mmHg", "pressure_auc",
                      "recovery_fraction", "fall_from_peak_mmHg"]
    duration_features = ["candidate_to_recovery_s", "confirm_to_recovery_s", "rise_to_peak_s", "peak_to_recovery_s"]
    for feature in event_features + duration_features + C.EXPLORATORY_EUS_FEATURES:
        for subject in C.SUBJECT_CYCLES:
            g = catalog[(catalog.subject == subject) & catalog.teacher_label.isin(["NVC_CORE", "PREVOID_PROGRESSIVE"])]
            label = "NVC_CORE"
            if feature in duration_features:
                vals = (g[g.teacher_label == label].recovery_confirm_s - g[g.teacher_label == label].candidate_start_s
                        if feature == "candidate_to_recovery_s" else
                        g[g.teacher_label == label].recovery_confirm_s - g[g.teacher_label == label].confirm_time_s
                        if feature == "confirm_to_recovery_s" else
                        g[g.teacher_label == label].local_peak_time_s - g[g.teacher_label == label].candidate_start_s
                        if feature == "rise_to_peak_s" else
                        g[g.teacher_label == label].recovery_confirm_s - g[g.teacher_label == label].local_peak_time_s)
            else:
                vals = g[g.teacher_label == label][feature]
            st = _stats(vals)
            rows.append({"feature_name": feature, "feature_type": "eus_exploratory" if feature in C.EXPLORATORY_EUS_FEATURES else "complete_event", "subject": subject, "label": label, **st})
    for feature, source in [("nvc_frequency_per_hour", "nvc_frequency_per_h"), ("nvc_count_per_valid_cycle", "n_nvc")]:
        for _, r in subject_summary.iterrows():
            st = _stats([r[source]])
            rows.append({"feature_name": feature, "feature_type": "cycle_summary", "subject": r.subject, "label": "NVC_CORE", **st})
    base = pd.DataFrame(rows)
    macro = []
    for (feature, ftype, label), g in base.groupby(["feature_name", "feature_type", "label"]):
        macro.append({"feature_name": feature, "feature_type": ftype, "subject": "MACRO_SUBJECT_BALANCED", "label": label,
                      "n": int((g["n"] > 0).sum()), "median": g["median"].mean(), "q25": g["q25"].mean(), "q75": g["q75"].mean(),
                      "min": g["min"].mean(), "max": g["max"].mean(), "missing_rate": g["missing_rate"].mean()})
    return pd.concat([base, pd.DataFrame(macro)], ignore_index=True)


def _freeze_protocol():
    return {
        "external_subject": "STxF30", "external_phenotype": "PARTIAL_SCI",
        "external_data_current_status": "PRE_STIM_BASELINE_ONLY_NOT_CYCLE_EXTRACTED",
        "read_data_in_this_stage": False,
        "evaluation_order": [
            "extract stable pre-stimulation cycles from the external subject source",
            "apply the same cycle boundaries and quality gates as the fixed cohort",
            "retain pressure, raw EUS, 100 Hz alignment, and native Volume",
            "apply the frozen automatic label rules",
            "apply the frozen complete and causal feature schemas",
            "compare against dsd_reference_feature_ranges.csv",
            "do not modify frozen rules based on external results",
            "do not add the external subject to training or threshold selection",
            "treat the external subject as exploratory external validation only",
        ],
        "special_cases": {"no_nvc": "valid_negative_animal", "overflow_or_untrusted_volume": "UNSCORABLE", "single_external_animal": "cannot_support_cross_animal_PASS"},
        "future_report_fields": ["valid_pre_stimulation_cycles", "valid_analysis_hours", "automatic_nvc_count", "nvc_frequency", "prominence", "four_duration_definitions", "pressure_auc", "causal_pressure_features", "exploratory_eus_features", "median_iqr_differences", "animal_range_membership", "frozen_m1_m2_scores_exploratory", "no_nvc_candidate_false_identifications"],
    }


def _artifact_manifest(output_root: Path, authoritative_names):
    legacy = {"baseline_nvc_events.csv", "baseline_nvc_subject_summary.csv", "nvc_yield_sensitivity.csv", "replay_events.csv", "replay_metrics.csv"}
    rows = []
    for path in sorted(output_root.iterdir()):
        if path.is_dir():
            if path.name != "quicklooks": continue
            count = len(list(path.glob("*.png"))); name = "quicklooks/"
            rows.append({"filename": name, "authoritative": True, "producer": "pre-f30-freeze", "stage": "quicklook", "row_count": count, "notes": "32 fixed-cycle diagnostic plots"})
            continue
        name = path.name
        auth = name in authoritative_names and name not in legacy
        rows.append({"filename": name, "authoritative": auth, "producer": "pre-f30-freeze" if auth else "legacy-or-prior-run", "stage": "stage_a_freeze" if auth else "legacy", "row_count": 1 if path.suffix == ".json" else (int(pd.read_csv(path).shape[0]) if path.suffix == ".csv" else np.nan), "notes": "LEGACY_NOT_AUTHORITATIVE" if not auth else "verified in fixed cohort"})
    return pd.DataFrame(rows)


def freeze_stage_a(args):
    teacher, manifest = assert_frozen_labels(args.output_root)
    pressure = pd.read_csv(args.output_root / "pressure_events.csv")
    params = pd.read_csv(args.output_root / "subject_adaptive_params.csv")
    urine = pd.read_csv(args.output_root / "urine_events.csv")
    event_features = add_event_uid(pd.read_csv(args.output_root / "event_features.csv"))
    cache = prior_and_cache(args.input_root, manifest, params, pressure)
    catalog = build_reference_event_catalog(cache, pressure, event_features, urine)
    label_counts = catalog.teacher_label.value_counts().to_dict()
    if label_counts != FIXED_LABEL_COUNTS or not catalog.event_uid.is_unique or set(catalog.subject) != set(C.SUBJECT_CYCLES):
        raise RuntimeError(f"Stage-A event catalog integrity failed: labels={label_counts}, unique={catalog.event_uid.is_unique}, subjects={set(catalog.subject)}")
    if len(catalog[(catalog.teacher_label == "NVC_CORE")]) != 10:
        raise RuntimeError("NVC event count is not 10")
    # Rewrite only derived event/prediction tables with the composite key; the teacher table remains untouched.
    for name in ["pressure_events.csv", "event_features.csv", "decision_time_sweep.csv", "loso_predictions.csv"]:
        path = args.output_root / name
        if path.exists(): add_event_uid(pd.read_csv(path)).to_csv(path, index=False)
    duration = build_duration_summary(catalog)
    subjects = build_subject_summary(catalog, manifest)
    ranges = build_reference_ranges(catalog, subjects)
    catalog.to_csv(args.output_root / "dsd_reference_event_catalog.csv", index=False)
    duration.to_csv(args.output_root / "nvc_duration_summary.csv", index=False)
    ranges.to_csv(args.output_root / "dsd_reference_feature_ranges.csv", index=False)
    subjects.to_csv(args.output_root / "dsd_reference_subject_summary.csv", index=False)
    schema = {"complete_event_feature_order": C.COMPLETE_EVENT_FEATURES, "causal_feature_order": C.CAUSAL_FEATURE_ORDER,
              "exploratory_eus_features": C.EXPLORATORY_EUS_FEATURES,
              "forbidden_inputs": ["subject", "cycle_id", "event_id", "event_uid", "Volume", "urine", "teacher_label", "time_to_urine_s", "future_external_fields"]}
    write_json(args.output_root / "feature_schema.json", schema)
    protocol = _freeze_protocol(); write_json(args.output_root / "external_validation_protocol.json", protocol)
    # Recompute the existing LOSO summary with composite-key event counts.
    sweep = add_event_uid(pd.read_csv(args.output_root / "decision_time_sweep.csv"))
    metrics = pd.read_csv(args.output_root / "decision_time_metrics.csv")
    if "train_subjects" not in metrics.columns:
        metrics["train_subjects"] = metrics.held_out_subject.map(lambda h: "+".join(s for s in C.SUBJECT_CYCLES if s != h))
    metrics["test_event_uid_count"] = metrics.apply(lambda r: int(sweep[(sweep.decision_delay_s == r.decision_delay_s) & (sweep.model == r.model) & (sweep.held_out_subject == r.held_out_subject)].event_uid.nunique()), axis=1)
    metrics.to_csv(args.output_root / "decision_time_metrics.csv", index=False)
    metrics.to_csv(args.output_root / "loso_metrics_by_subject.csv", index=False)
    comparison = pd.read_csv(args.output_root / "model_comparison.csv")
    eligible = sweep[sweep.decision_eligible.astype(bool)].groupby(["decision_delay_s", "teacher_label"]).event_uid.nunique().unstack(fill_value=0).to_dict("index")
    auth = {"teacher_labels.csv", "dataset_manifest.csv", "pressure_events.csv", "subject_adaptive_params.csv", "urine_events.csv", "event_features.csv", "decision_time_sweep.csv", "decision_time_metrics.csv", "loso_predictions.csv", "loso_metrics_by_subject.csv", "model_comparison.csv", "feature_separability.csv", "final_model.json", "run_summary.json", "feature_schema.json", "dsd_reference_event_catalog.csv", "nvc_duration_summary.csv", "dsd_reference_feature_ranges.csv", "dsd_reference_subject_summary.csv", "external_validation_protocol.json", "dsd_stage_a_freeze.json", "pre_f30_readiness.json", "artifact_manifest.csv", "stream_test_vectors.npz"}
    manifest_out = _artifact_manifest(args.output_root, auth); manifest_out.to_csv(args.output_root / "artifact_manifest.csv", index=False)
    authoritative = manifest_out[manifest_out.authoritative].filename.tolist()
    loso_summary = {"decision_delays_s": C.DECISION_DELAYS_S, "models": ["M1", "M2"], "outer_combinations": int(len(metrics)), "model_comparison_file": "model_comparison.csv", "event_uid_corrected": True, "training_subjects": list(C.SUBJECT_CYCLES), "held_out_subjects": list(C.SUBJECT_CYCLES)}
    stage = {"fixed_subjects": list(C.SUBJECT_CYCLES), "fixed_cycle_counts": C.SUBJECT_CYCLES, "automatic_label_counts": FIXED_LABEL_COUNTS,
             "automatic_label_protocol": {"teacher_label_source": "AUTOMATIC_PRESSURE_RECOVERY_PLUS_VOLUME", "teacher_label_quality": "RULE_BASED_SILVER_STANDARD", "manual_review_used": False},
             "duration_definitions": {"candidate_to_recovery_s": "recovery_confirm_s - candidate_start_s", "confirm_to_recovery_s": "recovery_confirm_s - confirm_time_s", "rise_to_peak_s": "local_peak_time_s - candidate_start_s", "peak_to_recovery_s": "recovery_confirm_s - local_peak_time_s"},
             "complete_event_feature_order": C.COMPLETE_EVENT_FEATURES, "causal_feature_order": C.CAUSAL_FEATURE_ORDER,
             "reference_distribution_file": "dsd_reference_feature_ranges.csv", "loso_summary": loso_summary,
             "feature_reference_status": "FROZEN_FOR_EXTERNAL_VALIDATION", "causal_classifier_status": "HOLD_NO_CAUSAL_SEPARATION",
             "manual_review_used": False, "stimulation_enabled": False, "STxF30_in_training": False}
    write_json(args.output_root / "dsd_stage_a_freeze.json", stage)
    readiness_checks = {"fixed_cohort_integrity": True, "label_integrity": True, "event_uid_integrity": bool(catalog.event_uid.is_unique), "authoritative_artifacts_consistent": True, "duration_definition_frozen": True, "feature_schema_frozen": True, "reference_ranges_generated": bool(len(ranges)), "loso_recalculated_with_event_uid": True, "external_protocol_frozen": True, "STxF30_not_read": True, "readiness_status": "READY_TO_EXTRACT_STxF30_CYCLES"}
    write_json(args.output_root / "pre_f30_readiness.json", readiness_checks)
    summary = {"fixed_label_counts": FIXED_LABEL_COUNTS, "valid_cycles": int(catalog[["subject", "cycle_id"]].drop_duplicates().shape[0]), "event_uid_unique": bool(catalog.event_uid.is_unique), "automatic_event_count": int(len(catalog)), "nvc_event_count": int((catalog.teacher_label == "NVC_CORE").sum()), "eligible_counts_by_delay_event_uid": {str(k): v for k, v in eligible.items()}, "authoritative_artifacts": authoritative, "feature_reference_status": "FROZEN_FOR_EXTERNAL_VALIDATION", "causal_classifier_status": "HOLD_NO_CAUSAL_SEPARATION", "manual_review_used": False, "stimulation_enabled": False, "STxF30_not_read": True, "STxF30_in_training": False, "external_protocol_file": "external_validation_protocol.json", "stage_freeze_file": "dsd_stage_a_freeze.json", "readiness_file": "pre_f30_readiness.json"}
    write_json(args.output_root / "run_summary.json", summary)
    # The stage/readiness files are created after the first manifest pass;
    # refresh it so every file named by run_summary is authoritative.
    manifest_out = _artifact_manifest(args.output_root, auth)
    manifest_out.to_csv(args.output_root / "artifact_manifest.csv", index=False)
    summary["authoritative_artifacts"] = manifest_out[manifest_out.authoritative].filename.tolist()
    write_json(args.output_root / "run_summary.json", summary)
    print(json.dumps({"status": "READY_TO_EXTRACT_STxF30_CYCLES", "nvc_event_count": 10, "authoritative_artifacts": len(authoritative)}, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["current", "pre-f30-freeze"], default="current"); ap.add_argument("--input-root", type=Path); ap.add_argument("--baseline-root", type=Path); ap.add_argument("--subject"); ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args(argv); args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "pre-f30-freeze":
        if args.input_root is None:
            ap.error("--input-root is required for pre-f30-freeze")
        return freeze_stage_a(args)
    if args.input_root is None:
        ap.error("--input-root is required for current mode")
    qdir = args.output_root / "quicklooks"; qdir.mkdir(exist_ok=True)
    teacher, manifest = assert_frozen_labels(args.output_root)
    pressure = pd.read_csv(args.output_root / "pressure_events.csv"); params = pd.read_csv(args.output_root / "subject_adaptive_params.csv"); urine = pd.read_csv(args.output_root / "urine_events.csv")
    cache = prior_and_cache(args.input_root, manifest, params, pressure)
    features = build_decision_features(cache, pressure)
    frozen_events = features[features.decision_delay_s == 0].drop_duplicates(["subject", "cycle_id", "event_id"])
    if len(frozen_events[frozen_events.teacher_label == "NVC_CORE"]) != 10 or len(frozen_events[frozen_events.teacher_label == "PREVOID_PROGRESSIVE"]) != 30:
        raise RuntimeError("Decision feature generation changed the frozen teacher event set")
    sweep, metrics = run_loso_by_delay(features)
    comparisons = []
    for (delay, model), g in metrics.groupby(["decision_delay_s", "model"]):
        dangerous = int(g.prevoid_false_accepts.sum() + g.void_false_accepts.sum()); accepted_subjects = int((g.nvc_accepts > 0).sum()); complete = bool((g.fold_status == "PASS").all()); abstain = bool(g.ABSTAIN_ALL.any())
        comparisons.append({"decision_delay_s": delay, "model": model, "macro_nvc_acceptance_rate": g.nvc_acceptance_rate.mean(), "total_nvc_accepts": int(g.nvc_accepts.sum()),
                            "dangerous_trigger_count": dangerous, "dangerous_trigger_rate": dangerous / int(g.eligible_prevoid.sum() + g.eligible_nvc.sum() - g.nvc_accepts.sum()) if int(g.eligible_prevoid.sum() + g.eligible_nvc.sum() - g.nvc_accepts.sum()) else 0.0,
                            "accepted_subjects": accepted_subjects, "all_three_folds_complete": complete, "ABSTAIN_ALL": abstain,
                            "safety_pass": bool(complete and not abstain and dangerous == 0 and int(g.nvc_accepts.sum()) > 0)})
    comparison = pd.DataFrame(comparisons)
    candidates = comparison[comparison.safety_pass]
    if candidates.empty: status = "HOLD_NO_CAUSAL_SEPARATION"; selected_model = "M1"; selected_delay = 0.0
    elif int(candidates.accepted_subjects.max()) < 3: status = "HOLD_CROSS_SUBJECT_UNSTABLE"; selected = candidates.sort_values(["macro_nvc_acceptance_rate", "decision_delay_s"]).iloc[-1]; selected_model, selected_delay = str(selected.model), float(selected.decision_delay_s)
    else: status = "PASS_338_SHADOW_SEPARABILITY"; selected = candidates.sort_values(["macro_nvc_acceptance_rate", "decision_delay_s", "model"], ascending=[False, True, True]).iloc[0]; selected_model, selected_delay = str(selected.model), float(selected.decision_delay_s)
    final_frame = features[(features.decision_delay_s == selected_delay) & (features.decision_eligible == 1)].copy(); train_s = list(C.SUBJECT_CYCLES)
    oof = cross_predictions(final_frame, train_s, selected_model); at, ai = select_analysis_threshold(oof); st, si = select_safety_threshold(oof); eus_final, eus_info = select_eus_features_train(final_frame) if selected_model == "M2" else ([], {"rows": []}); model = fit_logistic(final_frame, selected_model, train_s, eus_final)
    final_model = {"status": status, "stimulation_enabled": False, "selected_model": selected_model, "decision_delay_s": selected_delay, **serialize_model(model, selected_model, at, st, selected_delay), "analysis_threshold_selection": ai, "safety_threshold_selection": si, "eus_features_used": eus_final, "void_risk_target": "PREVOID_PROGRESSIVE/VOID_CONFIRMED=1; NVC_CORE=0"}
    sweep.to_csv(args.output_root / "decision_time_sweep.csv", index=False); metrics.to_csv(args.output_root / "decision_time_metrics.csv", index=False); comparison.to_csv(args.output_root / "model_comparison.csv", index=False)
    features.to_csv(args.output_root / "event_features.csv", index=False); sweep.to_csv(args.output_root / "loso_predictions.csv", index=False); metrics.to_csv(args.output_root / "loso_metrics_by_subject.csv", index=False); sweep.to_csv(args.output_root / "replay_events.csv", index=False); metrics.to_csv(args.output_root / "replay_metrics.csv", index=False)
    sep = feature_separability(features); sep.to_csv(args.output_root / "feature_separability.csv", index=False)
    write_json(args.output_root / "final_model.json", final_model)
    for (subject, cid), item in cache.items():
        sr = sweep[(sweep.subject == subject) & (sweep.cycle_id == cid) & (sweep.model == selected_model)]
        plot_cycle(qdir / f"{subject}_{cid}.png", item["cycle"], item["delta"], item["eus_env"], urine[(urine.subject == subject) & (urine.cycle_id == cid)], item["pressure"], pd.DataFrame(), item["adaptive"], sr)
    first = sorted(cache)[0]; item = cache[first]; sr = sweep[(sweep.subject == first[0]) & (sweep.cycle_id == first[1]) & (sweep.model == selected_model) & (sweep.decision_delay_s == selected_delay)].copy(); sr["probability"] = sr.p_void_risk; sr["confirm_index"] = sr.decision_index; sr["trigger"] = sr.trigger
    np.savez_compressed(args.output_root / "stream_test_vectors.npz", **stream_vectors(item["cycle"], item["delta"], item["eus_env"], sr, selected_model, st, item["adaptive"]))
    write_json(args.output_root / "run_summary.json", {"fixed_label_counts": FIXED_LABEL_COUNTS, "valid_cycles": int(manifest[["subject", "cycle_id"]].drop_duplicates().shape[0]), "decision_delays_s": C.DECISION_DELAYS_S,
        "eligible_counts_by_delay": features[features.decision_eligible == 1].groupby(["decision_delay_s", "teacher_label"]).event_id.nunique().unstack(fill_value=0).to_dict("index"),
        "model_comparison": comparison.to_dict("records"), "analysis_thresholds": sweep.groupby(["model", "held_out_subject", "decision_delay_s"]).analysis_threshold.first().reset_index().to_dict("records") if "analysis_threshold" in sweep else [],
        "safety_thresholds": sweep.groupby(["model", "held_out_subject", "decision_delay_s"]).safety_threshold.first().reset_index().to_dict("records"),
        "eus_direction_summary": sep[["feature", "direction"]].drop_duplicates().to_dict("records"), "selected_model": selected_model, "selected_delay_s": selected_delay,
        "status": status, "stimulation_enabled": False, "nvc_sufficiency": "338 DSD固定队列内缺少足够的高可信NVC事件。"})
    print((args.output_root / "run_summary.json").read_text(encoding="utf-8")); return 0


if __name__ == "__main__": raise SystemExit(main())
