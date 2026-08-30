"""Execute the V5 within-animal prospective NVC experiment."""
from __future__ import annotations

from pathlib import Path
import json
import traceback
import numpy as np
import pandas as pd

from .feature_extraction import extract_v4_features
from .source_adapter import _load_pair
from . import config as C
from .data_adapter import build_v5_dataset, event_confirm_time
from .modeling import (
    apply_fusion, apply_model, fit_fusion, fit_individual, metrics, prepare,
    score_model,
)


BASE_MODEL_MAP = {
    "I-P0": (C.P0_FEATURES, "lr"),
    "I-P1": (C.P1_FEATURES, "lr"),
    "I-M1": (C.M1_FEATURES, "lr"),
    "I-E0": (C.E0_FEATURES, "lr"),
    "I-M2": (C.M2_FEATURES, "lr"),
}
FUSION_MODEL_MAP = {"I-M3": "lr", "I-M4": "lda"}
GLOBAL_NAME_MAP = {
    "I-P0": "P0_anchor", "I-P1": "P1_morphology_dynamics", "I-M1": "M1_P_NVC",
    "I-E0": "E0_time", "I-M2": "M2_E_NVC", "I-M3": "M3_PE_NVC_LATE_FUSION",
    "I-M4": "M4_PE_NVC_SHRINKAGE_LDA",
}


def _safe_write_csv(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False)


def choose_high_coverage_features(train: pd.DataFrame):
    """Choose a reduced pressure set using calibration coverage only."""
    work = train[train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy()
    selected = list(C.M1_FEATURES)
    rows = []

    def coverage(cols):
        n = work[work.teacher_label.eq("NVC_CORE")]
        s = work[work.teacher_label.eq("STABLE_FILLING")]
        nc = float(np.isfinite(n[cols].to_numpy(float)).all(axis=1).mean()) if len(n) else np.nan
        sc = float(np.isfinite(s[cols].to_numpy(float)).all(axis=1).mean()) if len(s) else np.nan
        return nc, sc

    before, sbefore = coverage(selected)
    rows.append({"step": 0, "dropped_feature": "", "features": "|".join(selected),
                 "calibration_nvc_coverage": before, "calibration_stable_coverage": sbefore,
                 "selected": True})
    step = 0
    # The objective is high coverage, with no test information.  Keep at least
    # the three anchor features and remove only features that improve NVC
    # completeness.  In the registered data this identifies local variability.
    while before < 0.90 and len(selected) > len(C.P0_FEATURES):
        candidates = []
        for f in selected:
            if f in C.P0_FEATURES:
                continue
            cols = [x for x in selected if x != f]
            nc, sc = coverage(cols)
            candidates.append((nc, sc, f, cols))
        if not candidates:
            break
        best = max(candidates, key=lambda x: (x[0], x[1], -selected.index(x[2])))
        nc, sc, dropped, cols = best
        if nc <= before + 1e-12:
            break
        selected = cols; before, sbefore = nc, sc; step += 1
        rows.append({"step": step, "dropped_feature": dropped, "features": "|".join(selected),
                     "calibration_nvc_coverage": before, "calibration_stable_coverage": sbefore,
                     "selected": True})
    return tuple(selected), pd.DataFrame(rows)


def pressure_coverage_audit(train, test, features=C.M1_FEATURES, selected_hc=()):
    rows = []
    for split_name, frame in (("calibration", train), ("prospective_test", test)):
        for role in C.PRIMARY_TRAIN_LABELS:
            x = frame[frame.teacher_label.eq(role)]
            for f in features:
                miss = int(x[f].isna().sum())
                rows.append({"split": split_name, "teacher_label": role, "feature": f,
                             "n": int(len(x)), "missing": miss,
                             "coverage": float((len(x)-miss)/len(x)) if len(x) else np.nan,
                             "selected_in_I_M1_HC": bool(f in selected_hc)})
    return pd.DataFrame(rows)


def _empty_prediction(frame, name, reason):
    out = frame.copy(); out["model"] = name; out["score"] = np.nan; out["threshold"] = np.nan
    out["predicted_nvc"] = False; out["model_failure_reason"] = reason
    return out


def run_animal_models(all_rows, cycles, subject, train_cycles, test_cycles, hc_features):
    train = all_rows[(all_rows.subject == subject) & all_rows.cycle_id.astype(str).isin(tuple(train_cycles)) & all_rows.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy()
    test = all_rows[(all_rows.subject == subject) & all_rows.cycle_id.astype(str).isin(tuple(test_cycles)) & all_rows.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy()
    result_rows, per_cycle_rows, bundles, model_audit = [], [], {}, []
    specs = list(BASE_MODEL_MAP.items()) + [("I-M1-HC", (tuple(hc_features), "lr"))]
    for name, (features, classifier) in specs:
        try:
            model, threshold, source, tr_scored, oof = fit_individual(train, features, classifier)
            pred = apply_model(model, threshold, test, name)
            met = metrics(pred, test_cycles, cycles, name)
            result_rows.append({k: v for k, v in met.items() if k != "per_cycle"} | {"animal": subject, "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "train_nvc": int((train.teacher_label == "NVC_CORE").sum()), "threshold_source": source, "status": "OK"})
            per_cycle_rows.extend([{**r, "animal": subject, "train_cycles": "|".join(train_cycles)} for r in met["per_cycle"]])
            bundles[name] = {"kind": "base", "model": model, "threshold": threshold, "features": tuple(features), "train": train, "train_scored": tr_scored, "oof": oof}
            model_audit.append({"animal": subject, "model": name, "classifier": classifier, "status": "OK", "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "threshold": threshold, "threshold_source": source, "features": "|".join(features), "fit_cycles": "|".join(model.fit_cycles_)})
            result_rows[-1]["_pred"] = pred
        except Exception as exc:
            result_rows.append({"model": name, "animal": subject, "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "train_nvc": int((train.teacher_label == "NVC_CORE").sum()), "status": f"FAILED_{type(exc).__name__}", "failure_reason": str(exc)})
            bundles[name] = {"kind": "failed", "threshold": np.nan, "features": tuple(features), "reason": str(exc)}
            model_audit.append({"animal": subject, "model": name, "classifier": classifier, "status": f"FAILED_{type(exc).__name__}", "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "threshold": np.nan, "threshold_source": "", "features": "|".join(features), "failure_reason": str(exc)})
            result_rows[-1]["_pred"] = _empty_prediction(test, name, str(exc))
    # Late fusion uses the same individual train rows and the same causal
    # coupling fields, but a distinct classifier for I-M4.
    try:
        fb = fit_fusion(train, C.M1_FEATURES, C.M2_FEATURES, "lr")
        pred = apply_fusion(fb, test, "I-M3")
        met = metrics(pred, test_cycles, cycles, "I-M3")
        result_rows.append({k: v for k, v in met.items() if k != "per_cycle"} | {"animal": subject, "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "train_nvc": int((train.teacher_label == "NVC_CORE").sum()), "threshold_source": fb["threshold_source"], "status": "OK" if fb["fusion_model"] is not None else "FAILED_NO_FUSION_ROWS"})
        per_cycle_rows.extend([{**r, "animal": subject, "train_cycles": "|".join(train_cycles)} for r in met["per_cycle"]])
        bundles["I-M3"] = {"kind": "fusion", **fb}
        model_audit.append({"animal": subject, "model": "I-M3", "classifier": "lr", "status": "OK" if fb["fusion_model"] is not None else "FAILED_NO_FUSION_ROWS", "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "threshold": fb["threshold"], "threshold_source": fb["threshold_source"], "features": "|".join(C.FUSION_FEATURES)})
        result_rows[-1]["_pred"] = pred
    except Exception as exc:
        result_rows.append({"model": "I-M3", "animal": subject, "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "train_nvc": int((train.teacher_label == "NVC_CORE").sum()), "status": f"FAILED_{type(exc).__name__}", "failure_reason": str(exc)})
        bundles["I-M3"] = {"kind": "failed", "threshold": np.nan, "reason": str(exc), "features": tuple(C.FUSION_FEATURES)}
        result_rows[-1]["_pred"] = _empty_prediction(test, "I-M3", str(exc))
    try:
        fb = fit_fusion(train, C.M1_FEATURES, C.M2_FEATURES, "lda")
        pred = apply_fusion(fb, test, "I-M4")
        met = metrics(pred, test_cycles, cycles, "I-M4")
        result_rows.append({k: v for k, v in met.items() if k != "per_cycle"} | {"animal": subject, "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "train_nvc": int((train.teacher_label == "NVC_CORE").sum()), "threshold_source": fb["threshold_source"], "status": "OK" if fb["fusion_model"] is not None else "FAILED_NO_FUSION_ROWS"})
        per_cycle_rows.extend([{**r, "animal": subject, "train_cycles": "|".join(train_cycles)} for r in met["per_cycle"]])
        bundles["I-M4"] = {"kind": "fusion", **fb}
        model_audit.append({"animal": subject, "model": "I-M4", "classifier": "lda", "status": "OK" if fb["fusion_model"] is not None else "FAILED_NO_FUSION_ROWS", "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "threshold": fb["threshold"], "threshold_source": fb["threshold_source"], "features": "|".join(C.FUSION_FEATURES)})
        result_rows[-1]["_pred"] = pred
    except Exception as exc:
        result_rows.append({"model": "I-M4", "animal": subject, "train_cycles": "|".join(train_cycles), "test_cycles": "|".join(test_cycles), "train_nvc": int((train.teacher_label == "NVC_CORE").sum()), "status": f"FAILED_{type(exc).__name__}", "failure_reason": str(exc)})
        bundles["I-M4"] = {"kind": "failed", "threshold": np.nan, "reason": str(exc), "features": tuple(C.FUSION_FEATURES)}
        result_rows[-1]["_pred"] = _empty_prediction(test, "I-M4", str(exc))
    preds = pd.concat([x.pop("_pred") for x in result_rows], ignore_index=True)
    # `result_rows` may have been popped from; the scalar metric rows remain.
    return pd.DataFrame(result_rows), preds, pd.DataFrame(per_cycle_rows), bundles, pd.DataFrame(model_audit), train, test


def _score_bundle(bundle, row):
    """Score one causal feature row using an already-frozen individual bundle."""
    if bundle.get("kind") == "base":
        f = pd.DataFrame([row]); p = prepare(f, bundle["features"])
        if not bool(p.model_scorable.iloc[0]): return np.nan
        return float(score_model(bundle["model"], p)[0])
    if bundle.get("kind") == "fusion" and bundle.get("fusion_model") is not None:
        f = pd.DataFrame([row]); p = prepare(f, bundle["p_features"]); e = prepare(f, bundle["e_features"])
        out = f.copy(); out["S_P"] = np.nan; out["S_E"] = np.nan
        for col in C.COUPLING_FEATURES:
            if col not in out.columns:
                out[col] = np.nan
        if bool(p.model_scorable.iloc[0]): out.loc[0, "S_P"] = score_model(bundle["pmodel"], p)[0]
        if bool(e.model_scorable.iloc[0]): out.loc[0, "S_E"] = score_model(bundle["emodel"], e)[0]
        if not np.isfinite(out[["S_P", "S_E"] + list(C.COUPLING_FEATURES)].to_numpy(float)).all(): return np.nan
        return float(score_model(bundle["fusion_model"], out)[0])
    return np.nan


def streaming_replay(subject, test_cycles, bundles, events, paths):
    """Causal event trajectories at 0.25 s updates and fixed delay probes."""
    rows, summaries = [], []
    ev = events[(events.subject == subject) & events.cycle_id.astype(str).isin(tuple(test_cycles)) & events.teacher_label.eq("NVC_CORE")].copy()
    for er in ev.itertuples(index=False):
        key = (str(er.subject), str(er.cycle_id)); item = _load_pair(paths[key]); cycle = item["cycle"]
        idx0 = int(float(er.start_index)); idx_confirm = int(float(er.confirm_index)); t = np.asarray(cycle["t_abs_s"], dtype=float)
        if idx0 < 0 or idx_confirm < idx0 or idx_confirm >= len(t): continue
        end_idx = min(len(t) - 1, idx_confirm + int(round(2.0 * C.DP_FS_HZ)))
        grid = list(range(idx0, end_idx + 1, max(1, int(round(C.STREAM_UPDATE_S * C.DP_FS_HZ)))))
        if idx_confirm not in grid: grid.append(idx_confirm)
        # Explicitly include the registered delay probes even when the event
        # onset is not aligned to the 0.25 s grid.
        for delay in C.STREAM_DELAYS_S:
            j = idx_confirm + int(round(float(delay) * C.DP_FS_HZ))
            if idx0 <= j <= end_idx:
                grid.append(j)
        grid = sorted(set(grid))
        for name, bundle in bundles.items():
            threshold = float(bundle.get("threshold", np.nan))
            scores = []
            for idx in grid:
                tt = float(t[idx]); f, reason = extract_v4_features(cycle, idx, idx0, tt)
                row = {"subject": subject, "cycle_id": str(er.cycle_id), "teacher_label": "NVC_CORE", **f}
                score = _score_bundle(bundle, row)
                scores.append((idx, tt, score, reason))
                rows.append({"animal": subject, "model": name, "sample_uid": str(er.event_uid), "cycle_id": str(er.cycle_id), "update_time_s": tt, "relative_to_confirm_s": tt - float(t[idx_confirm]), "score": score, "threshold": threshold, "predicted_nvc": bool(np.isfinite(score) and np.isfinite(threshold) and score >= threshold), "feature_failure_reason": reason or "", "replay_type": "NVC_EVENT_CAUSAL_0P25S"})
            cross = [x for x in scores if np.isfinite(x[2]) and np.isfinite(threshold) and x[2] >= threshold]
            at = {round(x[1] - float(t[idx_confirm]), 2): x[2] for x in scores}
            summaries.append({"animal": subject, "model": name, "sample_uid": str(er.event_uid), "cycle_id": str(er.cycle_id), "candidate_onset_s": float(t[idx0]), "confirm_time_s": float(t[idx_confirm]), "first_valid_score_s": float(next((x[1] for x in scores if np.isfinite(x[2])), np.nan)), "first_threshold_crossing_s": float(cross[0][1]) if cross else np.nan, "trigger_latency_s": float(cross[0][1] - t[idx_confirm]) if cross else np.nan, "score_at_confirm": float(next((x[2] for x in scores if x[0] == idx_confirm and np.isfinite(x[2])), np.nan)), "score_at_plus_0p5s": float(next((v for k,v in at.items() if abs(k-0.5)<0.01), np.nan)), "score_at_plus_1s": float(next((v for k,v in at.items() if abs(k-1.0)<0.01), np.nan)), "score_at_plus_2s": float(next((v for k,v in at.items() if abs(k-2.0)<0.01), np.nan)), "threshold": threshold, "replay_type": "NVC_EVENT_SUMMARY"})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def challenge_replay(subject, challenges, bundles, test_cycles):
    """Score PREVOID/VOID only after individual models are frozen."""
    ch = challenges[(challenges.subject == subject) & challenges.cycle_id.astype(str).isin(tuple(test_cycles))].copy()
    parts, summaries = [], []
    for name, bundle in bundles.items():
        if bundle.get("kind") == "base":
            out = apply_model(bundle["model"], bundle["threshold"], ch, name)
        elif bundle.get("kind") == "fusion":
            out = apply_fusion(bundle, ch, name)
        else:
            out = _empty_prediction(ch, name, bundle.get("reason", "MODEL_NOT_FIT"))
        out["animal"] = subject; out["replay_type"] = "FROZEN_CHALLENGE"
        parts.append(out)
        for typ, g in out.groupby("challenge_type", dropna=False):
            scored = g[g.score.notna()]
            trig = scored.score >= scored.threshold
            summaries.append({
                "animal": subject, "model": name, "challenge_type": typ,
                "n": int(len(g)), "scorable": int(len(scored)),
                "triggered": int(trig.sum()),
                "trigger_fraction": float(trig.mean()) if len(scored) else np.nan,
                "median_score": float(scored.score.median()) if len(scored) else np.nan,
                "first_trigger_time_s": float(scored.loc[trig, "decision_time_s"].min()) if trig.any() else np.nan,
            })
    return (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(), pd.DataFrame(summaries))


def matched_global_comparison(bundles_by_animal, test_cycles_by_animal, cycles):
    """Compare frozen V4 global OOF scores on the exact matched V4 rows.

    V4 stable sampling is retained for this table so the comparison is truly
    paired.  The separately generated V5 stable rows (including F26/B15) are
    still used for individual prospective false-trigger metrics.
    """
    path = C.V4_ROOT / "v4_event_predictions.csv"
    if not path.exists(): return pd.DataFrame()
    global_pred = pd.read_csv(path)
    v4_rows_path = C.V4_ROOT / "v4_training_samples.csv"
    if not v4_rows_path.exists(): return pd.DataFrame()
    v4_rows = pd.read_csv(v4_rows_path)
    rows = []
    for subject, test_cycles in test_cycles_by_animal.items():
        raw = v4_rows[(v4_rows.subject == subject) & v4_rows.cycle_id.astype(str).isin(tuple(test_cycles)) & v4_rows.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy()
        for ind, glob in GLOBAL_NAME_MAP.items():
            bundle = bundles_by_animal.get(subject, {}).get(ind, {})
            if not raw.empty and bundle.get("kind") == "base":
                a = apply_model(bundle["model"], bundle["threshold"], raw, ind)
            elif not raw.empty and bundle.get("kind") == "fusion":
                a = apply_fusion(bundle, raw, ind)
            else:
                a = pd.DataFrame()
            b = global_pred[(global_pred.subject == subject) & global_pred.model.eq(glob) & global_pred.cycle_id.astype(str).isin(tuple(test_cycles))].copy()
            if b.empty or a.empty: continue
            joined = a[["sample_uid", "subject", "cycle_id", "teacher_label", "target", "score", "threshold"]].merge(b[["sample_uid", "score", "threshold"]], on="sample_uid", how="inner", suffixes=("_individual", "_global"))
            if joined.empty: continue
            ip = joined[["sample_uid", "subject", "cycle_id", "teacher_label", "target", "score_individual", "threshold_individual"]].rename(columns={"score_individual": "score", "threshold_individual": "threshold"})
            gp = joined[["sample_uid", "subject", "cycle_id", "teacher_label", "target", "score_global", "threshold_global"]].rename(columns={"score_global": "score", "threshold_global": "threshold"})
            for label, frame in (("V4_GLOBAL", gp), ("V5_INDIVIDUAL", ip)):
                mm = metrics(frame, test_cycles, cycles, f"{label}_{ind}")
                rows.append({k: v for k,v in mm.items() if k != "per_cycle"} | {"animal": subject, "model": ind, "comparison": label, "matched_rows": int(len(joined)), "matched_nvc": int((joined.teacher_label == "NVC_CORE").sum()), "matched_stable": int((joined.teacher_label == "STABLE_FILLING").sum())})
    return pd.DataFrame(rows)


def run(output_root: Path = C.OUTPUT_ROOT):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    train_all, challenges, stable_audit, cycles, paths, events = build_v5_dataset()
    train_all.to_csv(output_root / "v5_training_samples.csv", index=False)
    challenges.to_csv(output_root / "v5_challenge_samples.csv", index=False)
    stable_audit.to_csv(output_root / "v5_stable_windows.csv", index=False)
    cycles.to_csv(output_root / "v5_cycle_manifest.csv", index=False)

    all_results, all_preds, all_per_cycle, all_audits = [], [], [], []
    bundles_by_animal = {}; splits_by_animal = {}
    hc_audits = []
    coverage_parts = []
    for subject in C.SUBJECTS:
        split = C.SPLITS[subject]; splits_by_animal[subject] = split
        tr = train_all[(train_all.subject == subject) & train_all.cycle_id.astype(str).isin(split["train"])].copy()
        te = train_all[(train_all.subject == subject) & train_all.cycle_id.astype(str).isin(split["test"])].copy()
        hc, hca = choose_high_coverage_features(tr); hca["animal"] = subject; hc_audits.append(hca)
        coverage_parts.append(pressure_coverage_audit(tr, te, C.M1_FEATURES, hc))
        result, pred, per, bundles, audit, _, _ = run_animal_models(train_all, cycles, subject, split["train"], split["test"], hc)
        result["hc_features"] = "|".join(hc); pred["split"] = "prospective_test"; pred["animal"] = subject
        all_results.append(result); all_preds.append(pred); all_per_cycle.append(per); all_audits.append(audit)
        bundles_by_animal[subject] = bundles
    results = pd.concat(all_results, ignore_index=True)
    preds = pd.concat(all_preds, ignore_index=True)
    per_cycle = pd.concat(all_per_cycle, ignore_index=True)
    audits = pd.concat(all_audits, ignore_index=True)
    results.drop(columns=[c for c in results.columns if c.startswith("_")], errors="ignore").to_csv(output_root / "v5_individual_model_results.csv", index=False)
    preds.to_csv(output_root / "v5_test_predictions.csv", index=False)
    per_cycle.to_csv(output_root / "v5_per_cycle_metrics.csv", index=False)
    audits.to_csv(output_root / "v5_model_audit.csv", index=False)
    pd.concat(hc_audits, ignore_index=True).to_csv(output_root / "v5_high_coverage_selection.csv", index=False)
    pd.concat(coverage_parts, ignore_index=True).to_csv(output_root / "v5_pressure_coverage_audit.csv", index=False)

    # F37 calibration-length diagnostic (C3 duplicates the main fixed split).
    cal_rows = []
    for spec in C.CALIBRATION_LENGTH["STxF37"]:
        tr = train_all[(train_all.subject == "STxF37") & train_all.cycle_id.astype(str).isin(spec["train"])].copy()
        te = train_all[(train_all.subject == "STxF37") & train_all.cycle_id.astype(str).isin(spec["test"])].copy()
        hc, _ = choose_high_coverage_features(tr)
        try:
            rr, pp, _, _, _, _, _ = run_animal_models(train_all, cycles, "STxF37", spec["train"], spec["test"], hc)
            rr["calibration"] = spec["name"]; rr["calibration_nvc"] = int((tr.teacher_label == "NVC_CORE").sum()); rr["future_nvc"] = int((te.teacher_label == "NVC_CORE").sum()); cal_rows.append(rr)
        except Exception as exc:
            cal_rows.append(pd.DataFrame([{"animal": "STxF37", "calibration": spec["name"], "status": f"FAILED_{type(exc).__name__}", "failure_reason": str(exc)}]))
    cal = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()
    cal.to_csv(output_root / "v5_calibration_length.csv", index=False)

    # Prospective causal trajectories for all frozen individual bundles.
    stream_long, stream_summary = [], []
    for subject in C.SUBJECTS:
        long, summ = streaming_replay(subject, C.SPLITS[subject]["test"], bundles_by_animal[subject], events, paths)
        stream_long.append(long); stream_summary.append(summ)
    stream_long = pd.concat(stream_long, ignore_index=True) if stream_long else pd.DataFrame()
    stream_summary = pd.concat(stream_summary, ignore_index=True) if stream_summary else pd.DataFrame()
    stream_long.to_csv(output_root / "v5_streaming_replay.csv", index=False)
    stream_summary.to_csv(output_root / "v5_detection_timing.csv", index=False)

    # PREVOID/VOID are challenge-only: scoring happens after all individual
    # models and thresholds are frozen and is never fed back into selection.
    ch_parts, ch_summaries = [], []
    for subject in C.SUBJECTS:
        cp, cs = challenge_replay(subject, challenges, bundles_by_animal[subject], C.SPLITS[subject]["test"])
        if len(cp): ch_parts.append(cp)
        if len(cs): ch_summaries.append(cs)
    challenge_preds = pd.concat(ch_parts, ignore_index=True) if ch_parts else pd.DataFrame()
    challenge_summary = pd.concat(ch_summaries, ignore_index=True) if ch_summaries else pd.DataFrame()
    challenge_preds.to_csv(output_root / "v5_challenge_predictions.csv", index=False)
    challenge_summary.to_csv(output_root / "v5_challenge_summary.csv", index=False)

    matched = matched_global_comparison(bundles_by_animal, {s: C.SPLITS[s]["test"] for s in C.SUBJECTS}, cycles)
    matched.to_csv(output_root / "v5_global_vs_individual_matched.csv", index=False)

    # V5-B is deliberately not forced: calibration has only 3 (F26) and 5
    # (F37) NVC events, insufficient for stable rat-specific frequency selection.
    spectral_status = pd.DataFrame([{"analysis": "V5-B_Individual_Spectral_Adaptation", "status": "NOT_RUN_INSUFFICIENT_CALIBRATION_EVENTS", "F26_calibration_nvc": int(((train_all.subject == "STxF26") & train_all.cycle_id.astype(str).isin(C.SPLITS["STxF26"]["train"]) & train_all.teacher_label.eq("NVC_CORE")).sum()), "F37_calibration_nvc": int(((train_all.subject == "STxF37") & train_all.cycle_id.astype(str).isin(C.SPLITS["STxF37"]["train"]) & train_all.teacher_label.eq("NVC_CORE")).sum()), "reason": "No independent within-animal calibration validation remains for frequency selection."}])
    spectral_status.to_csv(output_root / "v5_spectral_adaptation_status.csv", index=False)

    # F26 stopping-rule diagnostics.
    f26_cycles = cycles[cycles.subject.eq("STxF26")].sort_values("cycle_start_s").copy()
    f26_nvc = train_all[(train_all.subject == "STxF26") & train_all.teacher_label.eq("NVC_CORE")].groupby("cycle_id").size().to_dict()
    first = next((r.cycle_id for r in f26_cycles.itertuples() if f26_nvc.get(r.cycle_id, 0) > 0), "")
    cumulative = 0; third = ""
    for r in f26_cycles.itertuples():
        cumulative += int(f26_nvc.get(r.cycle_id, 0))
        if cumulative >= 3: third = r.cycle_id; break
    def cycle_end(c):
        x = f26_cycles[f26_cycles.cycle_id.eq(c)]
        return float(x.cycle_end_s.iloc[0]) if len(x) else np.nan
    stopping = pd.DataFrame([{"animal": "STxF26", "first_nvc_cycle": first, "first_nvc_cycle_end_s": cycle_end(first), "cycle_at_3_cumulative_nvc": third, "cycle_at_3_cumulative_nvc_end_s": cycle_end(third), "calibration_elapsed_to_first_nvc_s": cycle_end(first) - float(f26_cycles.cycle_start_s.iloc[0]) if first else np.nan, "calibration_elapsed_to_3_nvc_s": cycle_end(third) - float(f26_cycles.cycle_start_s.iloc[0]) if third else np.nan}])
    stopping.to_csv(output_root / "v5_f26_calibration_stopping.csv", index=False)

    # Aggregate failure reasons and a concise machine-readable status.
    failures = train_all.assign(split=np.where(train_all.apply(lambda r: str(r.cycle_id) in C.SPLITS.get(str(r.subject), {}).get("train", ()), axis=1), "calibration", "prospective_test"))
    failure_summary = failures.groupby(["split", "subject", "teacher_label", "feature_failure_reason"], dropna=False).size().reset_index(name="n")
    failure_summary.to_csv(output_root / "v5_failure_reasons.csv", index=False)
    def status_for(name):
        x = results[(results.model == name) & results.status.eq("OK")]
        return bool(len(x) and x.nvc_detected.sum() > 0 and x.sensitivity.notna().any())
    statuses = {
        "INDIVIDUAL_PRESSURE_SUPPORTED": bool(status_for("I-M1") or status_for("I-M1-HC")),
        "INDIVIDUAL_EUS_SUPPORTED": status_for("I-E0"),
        "INDIVIDUAL_EUS_SPECTRAL_INCREMENT_SUPPORTED": False,
        "INDIVIDUAL_MULTIMODAL_INCREMENT_SUPPORTED": False,
        "HIGH_COVERAGE_PRESSURE_SUPPORTED": bool((results[(results.model == "I-M1-HC") & results.status.eq("OK")].coverage >= 0.9).any()),
        "CALIBRATION_REQUIREMENT_ESTIMATED": bool(len(cal)),
    }
    summary = {
        "version": "V5.0.0", "primary_task": "NVC_CORE vs STABLE_FILLING",
        "subjects": list(C.SUBJECTS), "splits": C.SPLITS,
        "train_nvc": {s: int(((train_all.subject == s) & train_all.cycle_id.astype(str).isin(C.SPLITS[s]["train"]) & train_all.teacher_label.eq("NVC_CORE")).sum()) for s in C.SUBJECTS},
        "test_nvc": {s: int(((train_all.subject == s) & train_all.cycle_id.astype(str).isin(C.SPLITS[s]["test"]) & train_all.teacher_label.eq("NVC_CORE")).sum()) for s in C.SUBJECTS},
        "stable_rows": int((train_all.teacher_label == "STABLE_FILLING").sum()),
        "models": list(C.MODEL_FEATURES) + ["I-M1-HC"], "v5_b_status": spectral_status.to_dict(orient="records"),
        "statuses": statuses, "development_only": True, "real_VNS_enabled": False,
        "deployment_ready": False, "model_results": results.drop(columns=[c for c in results.columns if c.startswith("_")], errors="ignore").to_dict(orient="records"),
    }
    (output_root / "v5_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    report = ["# V5 Individualized Prospective NVC Detection", "", "## Task", "- NVC_CORE vs STABLE_FILLING; PREVOID/VOID are challenge-only.", "- F26 and F37 are fitted separately; no cross-animal pooling.", "", "## Fixed prospective splits", pd.DataFrame([{"animal": s, "train_cycles": "|".join(C.SPLITS[s]["train"]), "test_cycles": "|".join(C.SPLITS[s]["test"]), "train_nvc": summary["train_nvc"][s], "test_nvc": summary["test_nvc"][s]} for s in C.SUBJECTS]).to_markdown(index=False), "", "## Individual model results", results.drop(columns=[c for c in results.columns if c.startswith("_")], errors="ignore").to_markdown(index=False), "", "## Calibration length", cal.to_markdown(index=False) if len(cal) else "No calibration-length results.", "", "## Status", json.dumps(statuses, ensure_ascii=False, indent=2), "", "development_only=true; deployment_ready=false; real_VNS_enabled=false."]
    (output_root / "V5_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    from .visualization import generate_plots
    generate_plots(output_root)
    return summary


if __name__ == "__main__":
    run()
