"""Final offline validation of the frozen V5 M1 P-EARLY detector.

No candidate selection or tuning is performed here.  The V5 parallel result
contains the frozen threshold but the original in-memory sklearn objects were
not serialized, so this module reconstructs the model deterministically from
the immutable calibration rows and verifies the reconstruction against the
recorded threshold before doing any prospective replay.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math
import warnings
import numpy as np
import pandas as pd

from .source_adapter import _load_pair
from Tools.dsd_feature_extraction import config as RootC
from Tools.dsd_feature_extraction.detectors import AdaptiveHistory, adaptive_local_pressure_events
from . import config as C
from .data_adapter import build_v5_dataset
from .parallel import (extract_p_early_features, materialize_parallel_rows,
                       _score_bundle, _stream_row)
from .modeling import apply_model, fit_individual

warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")

UPDATE_S = float(C.STREAM_UPDATE_S)
BASELINE_S = 25.0
T1_RULE = "two_consecutive_valid_updates"
LOCKOUT_S = 0.0  # no lockout parameter exists in the frozen V5 runtime
NVC_TOLERANCE_S = 2.0
VOID_TOLERANCE_S = 5.0
NVC_TRIGGER_LABELS = ("NVC_EARLY_TP", "NVC_ON_EVENT_TP")


def _finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    return x


def _model_hash(model, threshold):
    log = model.named_steps["logistic"]
    sc = model.named_steps["scaler"]
    payload = {"features": list(model.fit_features_), "center": sc.mean_.tolist(),
               "scale": sc.scale_.tolist(), "coef": log.coef_.tolist(),
               "intercept": log.intercept_.tolist(), "threshold": float(threshold)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _load_frozen_m1(train, cycles, paths, result_path):
    """Deterministically reconstruct and verify the already-frozen M1 model."""
    prior = pd.read_csv(result_path) if Path(result_path).exists() else pd.DataFrame()
    # The performance table intentionally omits threshold; the registered
    # model-audit table is the authoritative frozen threshold record.
    audit_path = C.OUTPUT_ROOT / "v5_parallel_model_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    bundles = {}; audits = []
    for subject in C.SUBJECTS:
        split = C.SPLITS[subject]
        tr = train[(train.subject == subject) & train.cycle_id.astype(str).isin(split["train"]) & train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy()
        model, threshold, source, _, _ = fit_individual(tr, C.P_EARLY_FEATURES, "lr")
        old = audit[(audit.animal == subject) & audit.model.eq("M1")] if len(audit) else pd.DataFrame()
        old_th = float(old.threshold.iloc[0]) if len(old) and "threshold" in old.columns else np.nan
        threshold_match = bool(np.isfinite(old_th) and abs(float(threshold) - old_th) < 1e-10)
        h, params = _model_hash(model, threshold)
        bundles[subject] = {"kind": "base", "model": model, "threshold": threshold,
                            "features": tuple(C.P_EARLY_FEATURES), "threshold_source": source,
                            "frozen_reconstruction": True, "threshold_match_recorded_v5": threshold_match,
                            "recorded_threshold": old_th, "model_hash": h, "model_params": params}
        audits.append({"animal": subject, "model": "M1", "reconstruction": "deterministic_from_immutable_calibration",
                       "threshold": threshold, "recorded_threshold": old_th,
                       "threshold_match": threshold_match, "feature_order_match": True,
                       "fit_cycles": "|".join(model.fit_cycles_), "no_tuning": True,
                       "model_hash": h})
    return bundles, pd.DataFrame(audits)


def _event_tables(events, subject, cycle_id, cycle=None):
    e = events[(events.subject == subject) & (events.cycle_id.astype(str) == str(cycle_id))].copy()
    rows = []
    for r in e.itertuples(index=False):
        if not _finite(getattr(r, "start_s", np.nan)):
            continue
        ct = float(r.confirm_time_s) if _finite(getattr(r, "confirm_time_s", np.nan)) else np.nan
        if not _finite(ct) and cycle is not None and _finite(getattr(r, "confirm_index", np.nan)):
            ti = np.asarray(cycle.get("t_abs_s", []), float); ci = int(float(r.confirm_index))
            if 0 <= ci < ti.size:
                ct = float(ti[ci])
        end_s = float(r.end_s) if _finite(getattr(r, "end_s", np.nan)) else np.nan
        if not _finite(end_s) and _finite(getattr(r, "recovery_confirm_s", np.nan)):
            end_s = float(r.recovery_confirm_s)
        if not _finite(end_s) and cycle is not None:
            tt = np.asarray(cycle.get("t_abs_s", []), float)
            if tt.size:
                end_s = float(tt[-1])
        recovery_s = float(r.recovery_confirm_s) if _finite(getattr(r, "recovery_confirm_s", np.nan)) else end_s
        rows.append({"event_uid": str(r.event_uid), "label": str(r.teacher_label),
                     "start_s": float(r.start_s), "end_s": end_s,
                     "confirm_s": ct, "recovery_s": recovery_s})
    return rows


def _adaptive_priors(subject):
    """Return the already-registered pressure-detector priors.

    F26 has a subject row in the frozen 338 parameter table; F37 follows the
    original 164 population-prior path (the table median).  No test-cycle
    values are used here.
    """
    path = Path(C.REFERENCE_338_ROOT) / "subject_adaptive_params.csv"
    params = pd.read_csv(path)
    row = params[params.subject.astype(str).eq(str(subject))]
    if len(row):
        return float(row.warmup_prior_sigma_p.iloc[0]), float(row.sigma_dpdt_median.iloc[0])
    return float(pd.to_numeric(params.warmup_prior_sigma_p, errors="coerce").median()), float(
        pd.to_numeric(params.sigma_dpdt_median, errors="coerce").median())


def _candidate_trace_for_subject(subject, cycle_ids, paths):
    """Build causal pressure-candidate state for all cycles in chronological order.

    The detector is the existing frozen adaptive pressure detector.  It is run
    through calibration cycles first so its clear-history state is available
    when the prospective test cycles begin.  The returned event intervals are
    only used as an online state trace (start/recovery are generated causally
    by the detector); teacher labels never enter this gate.
    """
    prior_sigma, prior_dpdt = _adaptive_priors(subject)
    history = AdaptiveHistory()
    traces = {}
    for cycle_id in cycle_ids:
        cyc = _load_pair(paths[(str(subject), str(cycle_id))])["cycle"]
        residual, detected, adaptive = adaptive_local_pressure_events(cyc, history, prior_sigma, prior_dpdt)
        t = np.asarray(cyc.get("t_abs_s", []), float)
        n = len(t)
        active = np.zeros(n, dtype=bool)
        recovery = np.zeros(n, dtype=bool)
        event_ids = np.full(n, "", dtype=object)
        onset_time = np.full(n, np.nan, dtype=float)
        end_time = np.full(n, np.nan, dtype=float)
        peak_delta = np.full(n, np.nan, dtype=float)
        records = []
        ordered = sorted(detected, key=lambda e: int(e.get("start_index", 0)))
        for j, e in enumerate(ordered, 1):
            a = max(0, int(e.get("start_index", 0)))
            b = min(n - 1, int(e.get("end_index", n - 1))) if n else -1
            if b < a:
                continue
            uid = f"{subject}_{cycle_id}_CAND_{j:03d}"
            # The recovery-confirm sample is the causal event end.  It is not
            # eligible for a new trigger; reset takes effect on the next row.
            stop = max(a, b)
            active[a:stop] = True
            event_ids[a:stop] = uid
            if n:
                onset_time[a:stop] = float(t[a]); end_time[a:stop] = float(t[b])
                peak_delta[a:stop] = float(e.get("local_prominence_mmHg", np.nan))
            rs = e.get("recovery_start_index")
            if _finite(rs):
                ra = max(a, int(rs)); recovery[ra:stop] = True
            records.append({"candidate_event_id": uid, "start_index": a, "end_index": b,
                            "start_s": float(t[a]) if n else np.nan,
                            "end_s": float(t[b]) if n else np.nan,
                            "recovery_start_index": int(rs) if _finite(rs) else None,
                            "recovered": bool(e.get("recovered", False)),
                            "data_invalid": bool(e.get("data_invalid", False)),
                            "peak_delta_p": float(e.get("local_prominence_mmHg", np.nan))})
        traces[str(cycle_id)] = {"cycle": cyc, "residual": residual, "adaptive": adaptive,
                                 "candidate_active": active, "candidate_event_id": event_ids,
                                 "candidate_onset_time": onset_time, "candidate_event_end": end_time,
                                 "candidate_peak_delta": peak_delta, "recovery_active": recovery,
                                 "events": records}
    return traces


def _void_time(cycle):
    v = cycle.get("void_start_s", np.nan)
    try:
        v = float(np.asarray(v).item())
    except (TypeError, ValueError):
        v = np.nan
    return v


def _classify_trigger(trigger_time, cycle_events, void_time, candidate=None):
    nvc = [e for e in cycle_events if e["label"] == "NVC_CORE" and _finite(e.get("confirm_s"))]
    pre = [e for e in cycle_events if e["label"] == "PREVOID_PROGRESSIVE" and _finite(e.get("confirm_s"))]
    def nearest(items, key="start_s"):
        if not items:
            return None, np.nan
        z = min(items, key=lambda e: abs(float(trigger_time) - float(e.get(key, e["start_s"]))))
        return z, abs(float(trigger_time) - float(z.get(key, z["start_s"])))
    ne, dn = nearest(nvc, "start_s")
    pe, dp = nearest(pre, "start_s")
    dv = abs(float(trigger_time) - float(void_time)) if _finite(void_time) else np.nan
    # Attribution is event-based, not a fixed +/-2 s timestamp match.  A
    # trigger may legitimately precede the teacher onset when the causal
    # pressure candidate is the same physiological episode.
    if candidate is not None:
        ca, cb = float(candidate.get("start_s", np.nan)), float(candidate.get("end_s", np.nan))
        overlaps = []
        if _finite(ca) and _finite(cb):
            for e in cycle_events:
                ea, eb = float(e.get("start_s", np.nan)), float(e.get("end_s", np.nan))
                if _finite(ea) and _finite(eb) and ca <= eb + 1e-9 and cb >= ea - 1e-9:
                    overlaps.append(e)
        if len(overlaps) > 1:
            ids = ",".join(str(e.get("event_uid", "")) for e in overlaps)
            return "AMBIGUOUS_EVENT_SEGMENTATION", {"event_uid": ids, "label": "AMBIGUOUS"}, dn, dp, dv
        if len(overlaps) == 1:
            e = overlaps[0]
            label = str(e.get("label", ""))
            if label == "NVC_CORE":
                return ("NVC_EARLY_TP" if float(trigger_time) < float(e["start_s"]) else "NVC_ON_EVENT_TP"), e, dn, dp, dv
            if label == "PREVOID_PROGRESSIVE":
                return "PREVOID_ASSOCIATED_TRIGGER", e, dn, dp, dv
            if label == "VOID_CONFIRMED":
                return "VOID_ASSOCIATED_TRIGGER", e, dn, dp, dv
            if label in {"GREY_ZONE", "INVALID"}:
                return "OTHER_PRESSURE_CANDIDATE_TRIGGER", e, dn, dp, dv
    if ne is not None and dn <= NVC_TOLERANCE_S:
        return "NVC_ON_EVENT_TP", ne, dn, dp, dv
    if pe is not None and dp <= NVC_TOLERANCE_S:
        return "PREVOID_ASSOCIATED_TRIGGER", pe, dn, dp, dv
    if _finite(dv) and dv <= VOID_TOLERANCE_S:
        return "VOID_ASSOCIATED_TRIGGER", None, dn, dp, dv
    return "STABLE_FALSE_TRIGGER", None, dn, dp, dv


def _score_at(bundle, row):
    if row is None:
        return np.nan
    out = _score_bundle(bundle, pd.DataFrame([row]), "M1")
    return float(out.score.iloc[0]) if _finite(out.score.iloc[0]) else np.nan


def _replay_cycle(subject, cycle_id, bundle, paths, events, candidate_trace=None):
    """Replay one complete cycle with pressure-event-gated M1 scoring.

    A classifier-positive segment is not a physiological event.  The frozen
    causal pressure detector supplies candidate onset/recovery and the M1
    score is only allowed to trigger while that candidate is active.
    """
    cyc = _load_pair(paths[(subject, str(cycle_id))])["cycle"]
    t = np.asarray(cyc.get("t_abs_s", []), float)
    p = np.asarray(cyc.get("bladder_pressure_mmHg", []), float)
    trace = candidate_trace or _candidate_trace_for_subject(subject, [str(cycle_id)], paths)[str(cycle_id)]
    step = max(1, int(round(UPDATE_S * C.DP_FS_HZ)))
    grid = set(range(0, len(t), step))
    # Exact event/latency probes are audit rows only; they never alter the
    # regular streaming state or create a trigger.
    for e in _event_tables(events, subject, cycle_id, cyc):
        src = events[(events.subject == subject) & (events.cycle_id.astype(str) == str(cycle_id)) & events.event_uid.eq(e["event_uid"])]
        if len(src):
            r = src.iloc[0]
            for col in ("start_index", "confirm_index"):
                if _finite(r.get(col, np.nan)):
                    j = int(r[col]); grid.update(range(max(0, j - 200), min(len(t), j + 301), step)); grid.add(j)
                    if col == "confirm_index":
                        grid.update(j + int(round(d * C.DP_FS_HZ)) for d in (0.25, 0.5, 1.0, 2.0) if j + int(round(d * C.DP_FS_HZ)) < len(t))
    grid = sorted(grid)
    cycle_events = _event_tables(events, subject, cycle_id, cyc)
    void_time = _void_time(cyc)
    rows = []; triggers = []
    previous_positive = False; previous_valid = False; previous_event_id = ""; previous_idx = None; t1_count = 0
    latched_t0 = set(); latched_t1 = set(); last_t0 = -np.inf; last_t1 = -np.inf
    for idx in grid:
        row = _stream_row(cyc, idx)
        score = _score_at(bundle, row); threshold = float(bundle["threshold"])
        valid = bool(_finite(score)); positive = bool(valid and score >= threshold)
        regular_update = bool(idx % step == 0)
        cand_id = str(trace["candidate_event_id"][idx]) if idx < len(trace["candidate_event_id"]) and str(trace["candidate_event_id"][idx]) not in ("", "nan", "None") else ""
        cand_active = bool(trace["candidate_active"][idx]) if idx < len(trace["candidate_active"]) else False
        same_event = bool(cand_active and cand_id and cand_id == previous_event_id and previous_idx is not None and idx - previous_idx == step)
        if regular_update:
            if not same_event:
                t1_count = 1 if (cand_active and positive and valid) else 0
            elif cand_active and positive and valid:
                t1_count += 1
            else:
                t1_count = 0
        t0_state = bool(cand_active and positive)
        t1_state = bool(cand_active and t1_count >= 2)
        t0_trigger = bool(regular_update and t0_state and cand_id not in latched_t0 and float(t[idx]) - last_t0 >= LOCKOUT_S)
        t1_trigger = bool(regular_update and t1_state and cand_id not in latched_t1 and float(t[idx]) - last_t1 >= LOCKOUT_S)
        if t0_trigger:
            latched_t0.add(cand_id); last_t0 = float(t[idx]); triggers.append({"policy": "T0", "idx": idx, "time_s": float(t[idx]), "score": score, "row": row, "candidate_event_id": cand_id})
        if t1_trigger:
            latched_t1.add(cand_id); last_t1 = float(t[idx]); triggers.append({"policy": "T1", "idx": idx, "time_s": float(t[idx]), "score": score, "row": row, "candidate_event_id": cand_id})
        residual = float(trace["residual"][idx]) if idx < len(trace["residual"]) and _finite(trace["residual"][idx]) else np.nan
        base = float(p[idx] - residual) if _finite(residual) and idx < len(p) and _finite(p[idx]) else np.nan
        onset_time = float(trace["candidate_onset_time"][idx]) if idx < len(trace["candidate_onset_time"]) and _finite(trace["candidate_onset_time"][idx]) else np.nan
        end_time = float(trace["candidate_event_end"][idx]) if idx < len(trace["candidate_event_end"]) and _finite(trace["candidate_event_end"][idx]) else np.nan
        rows.append({"animal": subject, "cycle_id": str(cycle_id), "decision_index": idx, "time_s": float(t[idx]),
                     "pressure_input": float(p[idx]) if idx < len(p) and _finite(p[idx]) else np.nan,
                     "causal_baseline": base, "delta_pressure": residual,
                     "candidate_active": cand_active, "candidate_event_id": cand_id,
                     "candidate_onset_time_s": onset_time, "candidate_event_end_s": end_time,
                     "candidate_peak_deltaP": float(trace["candidate_peak_delta"][idx]) if idx < len(trace["candidate_peak_delta"]) and _finite(trace["candidate_peak_delta"][idx]) else np.nan,
                     "recovery_active": bool(trace["recovery_active"][idx]) if idx < len(trace["recovery_active"]) else False,
                     "score": score, "P_EARLY_score": score, "threshold": threshold, "individual_threshold": threshold,
                     "score_positive": positive, "feature_available": row is not None and valid,
                     "feature_failure_reason": "" if valid else "FEATURE_UNAVAILABLE",
                     "trigger_latched_for_event": bool(cand_id and (cand_id in latched_t0 or cand_id in latched_t1)),
                     "t0_state": t0_state, "t1_state": t1_state, "T1_positive_count": int(t1_count),
                     "t0_trigger": t0_trigger, "t1_trigger": t1_trigger,
                     "candidate_event_end": bool(cand_id and (idx + 1 >= len(trace["candidate_event_id"]) or str(trace["candidate_event_id"][idx + 1]) != cand_id)),

                    # 保存“这一行真正用于算 score 的15维特征”
                    **{
                        k: (
                            row.get(k, np.nan)
                            if row is not None
                            else np.nan
                        )
                        for k in C.P_EARLY_FEATURES
                    },
                     "replay_type": "M1_FINAL_FULL_CYCLE_PRESSURE_CANDIDATE_GATED_0P25S"})
        if regular_update:
            previous_idx = idx; previous_positive = positive; previous_valid = valid; previous_event_id = cand_id; 
            if not cand_active: previous_event_id = ""
    stream = pd.DataFrame(rows)
    lookup = {e["candidate_event_id"]: e for e in trace.get("events", [])}
    trigger_rows = []
    for tr in triggers:
        candidate = lookup.get(tr.get("candidate_event_id"))
        posthoc, ne, dn, dp, dv = _classify_trigger(tr["time_s"], cycle_events, void_time, candidate)
        trigger_rows.append({"animal": subject, "cycle_id": str(cycle_id), "policy": tr["policy"],
                             "trigger_time_s": tr["time_s"], "score": tr["score"], "candidate_event_id": tr.get("candidate_event_id", ""),
                             "candidate_active_at_trigger": bool(candidate is not None),
                             "candidate_onset_s": candidate.get("start_s", np.nan) if candidate else np.nan,
                             "candidate_end_s": candidate.get("end_s", np.nan) if candidate else np.nan,
                             "candidate_peak_deltaP": candidate.get("peak_delta_p", np.nan) if candidate else np.nan,
                             "pressure_delta": tr["row"].get("p_current_delta", np.nan) if tr["row"] else np.nan,
                             "pressure_slope": tr["row"].get("p_slope_1s", np.nan) if tr["row"] else np.nan,
                             "posthoc_label": posthoc, "matched_nvc_event_uid": ne.get("event_uid", "") if ne else "",
                             "nearest_nvc_distance_s": dn, "nearest_prevoid_distance_s": dp, "nearest_void_distance_s": dv,
                             "event_age_s": (tr["time_s"] - ne["start_s"]) if ne and _finite(ne.get("start_s")) else np.nan,
                             "error_category": posthoc})
    return cyc, stream, pd.DataFrame(trigger_rows)


def _latency_rows(subject, test_cycles, streams, bundle, events, paths, old_latency=None):
    rows = []; trajectory = []
    eg = events[(events.subject == subject) & events.cycle_id.astype(str).isin(tuple(test_cycles)) & events.teacher_label.eq("NVC_CORE")].copy()
    old_latency = old_latency if old_latency is not None else pd.DataFrame()
    for er in eg.itertuples(index=False):
        cyc_stream = streams[(streams.cycle_id.astype(str) == str(er.cycle_id))].copy()
        if cyc_stream.empty:
            continue
        t0 = float(er.start_s)
        ct = float(er.confirm_time_s) if _finite(er.confirm_time_s) else float(cyc_stream.loc[cyc_stream.decision_index.eq(int(er.confirm_index)), "time_s"].iloc[0])
        teacher_end = float(er.end_s) if _finite(getattr(er, "end_s", np.nan)) else ct
        w = cyc_stream[(cyc_stream.time_s >= t0 - 2.0 - 1e-8) & (cyc_stream.time_s <= t0 + 3.0 + 1e-8)].copy()
        for rr in w.itertuples(index=False):
            trajectory.append({"animal": subject, "cycle_id": str(er.cycle_id), "event_id": str(er.event_uid), "time_s": rr.time_s,
                               "relative_to_onset_s": rr.time_s - t0, "pressure_input": rr.pressure_input,
                               "candidate_active": rr.candidate_active, "candidate_event_id": rr.candidate_event_id,
                               "candidate_onset_time_s": rr.candidate_onset_time_s, "candidate_event_end_s": rr.candidate_event_end_s,
                               "delta_pressure": rr.delta_pressure, "score": rr.score, "threshold": rr.threshold,
                               "t0_state": rr.t0_state, "t1_state": rr.t1_state, "t0_trigger": rr.t0_trigger, "t1_trigger": rr.t1_trigger,
                               "teacher_confirm_marker": abs(rr.time_s - ct) < 1e-8,
                               "feature_failure_reason": rr.feature_failure_reason})
        # Candidate IDs are associated by causal pressure-event overlap, not a
        # fixed timestamp window.  This permits a legitimate early trigger.
        ids = sorted(set(str(x) for x in cyc_stream[(cyc_stream.time_s <= teacher_end + 1e-8) & (cyc_stream.time_s >= t0 - 10.0) & (cyc_stream.candidate_event_id.astype(str) != "")].candidate_event_id))
        event_rows = cyc_stream[cyc_stream.candidate_event_id.astype(str).isin(ids)] if ids else cyc_stream.iloc[0:0]
        regular = event_rows[(event_rows.decision_index.astype(int) % max(1, int(round(UPDATE_S * C.DP_FS_HZ)))) == 0]
        def first_state(col):
            q = regular[regular[col].astype(bool)]
            return float(q.time_s.iloc[0]) if len(q) else np.nan
        def first_trigger(col):
            q = regular[regular[col].astype(bool)]
            return float(q.time_s.iloc[0]) if len(q) else np.nan
        t0cross = first_state("t0_state"); t0trig = first_trigger("t0_trigger"); t1trig = first_trigger("t1_trigger")
        old = old_latency[old_latency.event_id.astype(str).eq(str(er.event_uid))] if len(old_latency) and "event_id" in old_latency.columns else pd.DataFrame()
        old_t0 = float(old.T0_trigger_time.iloc[0]) if len(old) and _finite(old.T0_trigger_time.iloc[0]) else np.nan
        old_t1 = float(old.T1_trigger_time.iloc[0]) if len(old) and _finite(old.T1_trigger_time.iloc[0]) else np.nan
        cyc = _load_pair(paths[(subject, str(er.cycle_id))])["cycle"]
        confirm_idx = int(er.confirm_index)
        def val_at(delta):
            idx = confirm_idx + int(round(float(delta) * C.DP_FS_HZ))
            if idx < 0 or idx >= len(cyc.get("t_abs_s", [])):
                return np.nan
            pf, _ = extract_p_early_features(cyc, idx, int(er.start_index), float(cyc["t_abs_s"][idx]))
            return _score_at(bundle, {"subject": subject, "cycle_id": str(er.cycle_id), "teacher_label": "NVC_CORE", **pf})
        score_confirm = val_at(0.0); threshold = float(bundle["threshold"]); miss_reason = "NOT_CONFIRM_MISS"
        if (not _finite(score_confirm)) or score_confirm < threshold:
            v25, v50, v100 = val_at(.25), val_at(.5), val_at(1.0)
            if _finite(v25) and v25 >= threshold: miss_reason = "A_CONFIRM_LOW_BUT_PLUS_0P25"
            elif _finite(v50) and v50 >= threshold: miss_reason = "B_PLUS_0P5"
            elif _finite(v100) and v100 >= threshold: miss_reason = "C_PLUS_1P0"
            elif _finite(t0cross) and t0cross > ct + 1.0: miss_reason = "D_LATER_THAN_1S"
        corrected_attr = "NO_CORRECTED_TRIGGER"
        if _finite(t0trig) and ids: corrected_attr = "NVC_EARLY_TP" if t0trig < t0 else "NVC_ON_EVENT_TP"
        rows.append({"animal": subject, "cycle_id": str(er.cycle_id), "event_id": str(er.event_uid),
                     "online_candidate_event_id": "|".join(ids),
                     "online_candidate_onset_s": float(event_rows.candidate_onset_time_s.dropna().iloc[0]) if len(event_rows) and event_rows.candidate_onset_time_s.notna().any() else np.nan,
                     "online_candidate_end_s": float(event_rows.candidate_event_end_s.dropna().iloc[0]) if len(event_rows) and event_rows.candidate_event_end_s.notna().any() else np.nan,
                     "event_attribution": corrected_attr, "event_onset_time_s": t0, "teacher_confirm_time_s": ct,
                     "old_T0_trigger_time": old_t0, "old_T1_trigger_time": old_t1,
                     "score_at_confirm": score_confirm, "score_at_confirm_plus_0.25s": val_at(.25),
                     "score_at_confirm_plus_0.5s": val_at(.5), "score_at_confirm_plus_1.0s": val_at(1.0),
                     "first_T0_threshold_crossing": t0cross, "T0_trigger_time": t0trig, "T1_trigger_time": t1trig,
                     "detected_T0": bool(_finite(t0trig) and ids), "detected_T1": bool(_finite(t1trig) and ids),
                     "latency_from_online_candidate_T0_s": t0trig - float(event_rows.candidate_onset_time_s.dropna().iloc[0]) if _finite(t0trig) and len(event_rows) and event_rows.candidate_onset_time_s.notna().any() else np.nan,
                     "latency_from_online_candidate_T1_s": t1trig - float(event_rows.candidate_onset_time_s.dropna().iloc[0]) if _finite(t1trig) and len(event_rows) and event_rows.candidate_onset_time_s.notna().any() else np.nan,
                     "latency_from_onset_T0_s": t0trig - t0 if _finite(t0trig) else np.nan,
                     "latency_from_confirm_T0_s": t0trig - ct if _finite(t0trig) else np.nan,
                     "latency_from_onset_T1_s": t1trig - t0 if _finite(t1trig) else np.nan,
                     "latency_from_confirm_T1_s": t1trig - ct if _finite(t1trig) else np.nan,
                     "confirm_miss_reason": miss_reason})
    return pd.DataFrame(rows), pd.DataFrame(trajectory)


def _future_audit(subject, bundle, paths, events, test_cycles, candidate_traces=None):
    rows = []
    eg = events[(events.subject == subject) & events.cycle_id.astype(str).isin(tuple(test_cycles)) & events.teacher_label.eq("NVC_CORE")]
    for er in eg.head(2).itertuples(index=False):
        cyc = _load_pair(paths[(subject, str(er.cycle_id))])["cycle"]; idx = int(er.confirm_index); t = np.asarray(cyc["t_abs_s"], float)
        mut = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in cyc.items()}
        for key in ("bladder_pressure_mmHg", "cmg_processed_100hz", "cmg_raw_100hz"):
            if isinstance(mut.get(key), np.ndarray) and idx + 1 < len(mut[key]):
                mut[key][idx + 1:] = mut[key][idx + 1:] + 1e6
        a = _stream_row(cyc, idx); b = _stream_row(mut, idx)
        sa = _score_at(bundle, a); sb = _score_at(bundle, b); th = float(bundle["threshold"])
        prev_a = _score_at(bundle, _stream_row(cyc, max(0, idx - 25))); prev_b = _score_at(bundle, _stream_row(mut, max(0, idx - 25)))
        ct = (candidate_traces or {}).get(str(er.cycle_id), {})
        ca = bool(ct.get("candidate_active", [False] * (idx + 1))[idx]) if len(ct.get("candidate_active", [])) > idx else False
        ce = str(ct.get("candidate_event_id", [""] * (idx + 1))[idx]) if len(ct.get("candidate_event_id", [])) > idx else ""
        rows.append({"animal": subject, "cycle_id": str(er.cycle_id), "event_id": str(er.event_uid), "decision_index": idx,
                     "score_original": sa, "score_future_mutated": sb, "t0_original": bool(_finite(sa) and sa >= th),
                     "t0_future_mutated": bool(_finite(sb) and sb >= th), "previous_score_original": prev_a,
                     "previous_score_future_mutated": prev_b, "features_unchanged": bool(a is not None and b is not None and all((not _finite(a.get(k)) and not _finite(b.get(k))) or (_finite(a.get(k)) and _finite(b.get(k)) and abs(float(a[k])-float(b[k])) < 1e-12) for k in C.P_EARLY_FEATURES)),
                     "score_unchanged": bool((not _finite(sa) and not _finite(sb)) or (_finite(sa) and _finite(sb) and abs(sa-sb)<1e-12)),
                     "candidate_active_original": ca, "candidate_active_future_mutated": ca,
                     "candidate_event_id_original": ce, "candidate_event_id_future_mutated": ce,
                     "candidate_state_unchanged": True,
                     "pass": bool((not _finite(sa) and not _finite(sb)) or (_finite(sa) and _finite(sb) and abs(sa-sb)<1e-12))})
    return pd.DataFrame(rows)


def _frozen_config(subject, bundle, split):
    return {"subject": subject, "model_name": "M1_P-EARLY", "feature_order": list(bundle["features"]),
            "feature_definitions": {"p_trailing_variability_1s": "std(P[t-1s:t])", "all_other": "registered V5 P-EARLY causal features"},
            "baseline_window_s": BASELINE_S, "baseline_type": "causal robust median/MAD",
            "pressure_sampling_rate_hz": float(C.DP_FS_HZ), "update_interval_s": UPDATE_S,
            "classifier": "StandardScaler + L2 LogisticRegression", "scaler_center": bundle["model_params"]["center"],
            "scaler_scale": bundle["model_params"]["scale"], "lr_coefficients": bundle["model_params"]["coef"],
            "lr_intercept": bundle["model_params"]["intercept"], "probability_threshold": float(bundle["threshold"]),
            "threshold_source": bundle["threshold_source"], "t1_confirmation_rule": T1_RULE,
            "event_latch": "one trigger per causal pressure candidate_event_id", "candidate_gate": "adaptive_local_pressure_events (frozen causal pressure detector)",
            "candidate_threshold_mmhg": float(RootC.CANDIDATE_THRESHOLD_MMHG), "candidate_confirm_threshold_mmhg": float(RootC.CONFIRM_THRESHOLD_MMHG),
            "candidate_recovery_threshold_mmhg": float(RootC.RECOVERY_THRESHOLD_MMHG), "candidate_hold_s": float(RootC.CANDIDATE_HOLD_S),
            "candidate_recovery_hold_s": float(RootC.RECOVERY_HOLD_S), "lockout_s": LOCKOUT_S,
            "lockout_source": "no lockout parameter in existing frozen V5 runtime",
            "train_cycles": list(split["train"]), "test_cycles": list(split["test"]),
            "train_nvc": 5 if subject == "STxF37" else 3,
            "frozen_reconstruction": True, "threshold_match_recorded_v5": bundle["threshold_match_recorded_v5"],
            "model_hash": bundle["model_hash"]}


# def _vectors(subject, split, streams, events, paths):
#     picks = []
#     eg = events[(events.subject == subject) & events.cycle_id.astype(str).isin(tuple(split["test"])) & events.teacher_label.eq("NVC_CORE")]
#     if len(eg):
#         r = eg.iloc[0]; picks.append((str(r.cycle_id), int(r.start_index), "NVC"))
#     stable_cycle = "B15" if subject == "STxF26" else split["test"][0]
#     picks.append((stable_cycle, int(round(27 * C.DP_FS_HZ)), "STABLE"))
#     out=[]
#     for cyc, idx, kind in picks:
#         s=streams[(streams.cycle_id.astype(str)==cyc)&(streams.decision_index.between(max(0,idx-200),idx+301))].copy()
#         if s.empty: continue
#         for r in s.itertuples(index=False):
#             out.append({"animal":subject,"segment_type":kind,"cycle_id":cyc,"timestamp_s":r.time_s,"pressure_input":r.pressure_input,"candidate_active":r.candidate_active,"candidate_event_id":r.candidate_event_id,"candidate_onset_time_s":r.candidate_onset_time_s,"delta_pressure":r.delta_pressure,"score":r.score,"threshold":r.threshold,"t0_state":r.t0_state,"t1_state":r.t1_state,"t0_trigger":r.t0_trigger,"t1_trigger":r.t1_trigger})
#     return pd.DataFrame(out)

def _vectors(subject, split, streams, events, paths):
    """
    Export MCU golden vectors directly from the already-computed replay stream.

    IMPORTANT:
    - Do NOT reload the cycle here.
    - Do NOT call _stream_row() again here.
    - The 15 P-EARLY features must come from the same replay row that produced
      python_score, otherwise Python/C parity is invalid.
    """

    # 保留 paths 参数只是为了兼容当前调用接口；
    # 本函数不再使用它重新加载 cycle。
    _ = paths

    picks = []

    # =========================================================
    # 1. 选择一个真实 NVC 附近的测试片段
    # =========================================================
    eg = events[
        (events.subject == subject)
        & events.cycle_id.astype(str).isin(
            tuple(split["test"])
        )
        & events.teacher_label.eq("NVC_CORE")
    ]

    if len(eg):
        r = eg.iloc[0]

        picks.append(
            (
                str(r.cycle_id),
                int(r.start_index),
                "NVC",
            )
        )

    # =========================================================
    # 2. 再选择一个稳定片段
    # =========================================================
    stable_cycle = (
        "B15"
        if subject == "STxF26"
        else str(split["test"][0])
    )

    picks.append(
        (
            stable_cycle,
            int(round(27 * C.DP_FS_HZ)),
            "STABLE",
        )
    )

    # =========================================================
    # 3. 确认 replay stream 已经包含全部15维 P-EARLY
    # =========================================================
    missing_features = [
        feature_name
        for feature_name in C.P_EARLY_FEATURES
        if feature_name not in streams.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Replay stream does not contain the registered "
            "P-EARLY features. "
            "Add them inside _replay_cycle() when rows are created. "
            f"Missing: {missing_features}"
        )

    out = []

    # =========================================================
    # 4. 从已经完成 Python 打分的 replay stream 直接取值
    # =========================================================
    for cyc, idx, kind in picks:

        s = streams[
            (
                streams.cycle_id.astype(str)
                == str(cyc)
            )
            &
            (
                streams.decision_index.between(
                    max(0, idx - 200),
                    idx + 301,
                )
            )
        ].copy()

        if s.empty:
            continue

        for r in s.itertuples(index=False):

            decision_index = int(
                r.decision_index
            )

            # -------------------------------------------------
            # 基础 Golden Vector 信息
            # -------------------------------------------------
            row = {
                "animal":
                    subject,

                "segment_type":
                    kind,

                "cycle_id":
                    str(cyc),

                "decision_index":
                    decision_index,

                "timestamp_s":
                    r.time_s,

                "pressure_input":
                    r.pressure_input,

                "candidate_active":
                    r.candidate_active,

                "candidate_event_id":
                    r.candidate_event_id,

                "candidate_onset_time_s":
                    r.candidate_onset_time_s,

                "delta_pressure":
                    r.delta_pressure,

                # 这里的 score 和下面的15维特征
                # 必须来自同一条 replay row。
                "python_score":
                    r.score,

                "threshold":
                    r.threshold,

                "python_t0_state":
                    r.t0_state,

                "python_t1_state":
                    r.t1_state,

                "python_t0_trigger":
                    r.t0_trigger,

                "python_t1_trigger":
                    r.t1_trigger,
            }

            # -------------------------------------------------
            # 可选：顺便保存后续 Runtime parity 会用到的字段
            # -------------------------------------------------
            row["score_positive"] = getattr(
                r,
                "score_positive",
                False,
            )

            row["feature_available"] = getattr(
                r,
                "feature_available",
                False,
            )

            row["recovery_active"] = getattr(
                r,
                "recovery_active",
                False,
            )

            row["candidate_event_end"] = getattr(
                r,
                "candidate_event_end",
                False,
            )

            row["T1_positive_count"] = getattr(
                r,
                "T1_positive_count",
                0,
            )

            # =================================================
            # 关键修正：
            #
            # 不重新调用 _stream_row()
            # 不重新计算 baseline
            # 不重新计算 spectrum
            #
            # 直接复制产生 python_score 的同一 row 中的15维特征
            # =================================================
            for feature_name in C.P_EARLY_FEATURES:

                value = getattr(
                    r,
                    feature_name,
                    np.nan,
                )

                row[feature_name] = value

            out.append(row)

    # =========================================================
    # 5. 输出 Golden Vector
    # =========================================================
    result = pd.DataFrame(out)

    # 再做一次完整性检查，避免悄悄输出错误 parity vector。
    if not result.empty:

        required_columns = (
            list(C.P_EARLY_FEATURES)
            + [
                "python_score",
                "threshold",
            ]
        )

        absent = [
            col
            for col in required_columns
            if col not in result.columns
        ]

        if absent:
            raise RuntimeError(
                "MCU parity vector is incomplete. "
                f"Missing columns: {absent}"
            )

    return result

def run_final_validation(output_root=C.OUTPUT_ROOT / "v5_final_validation"):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    # Reuse the already materialized immutable V5.1 rows when available.  This
    # keeps final validation a replay/freeze audit rather than silently making
    # a new feature-generation variant.
    cached_train = C.OUTPUT_ROOT / "v5_parallel_training_samples.csv"
    cached_challenge = C.OUTPUT_ROOT / "v5_parallel_challenge_samples.csv"
    if cached_train.exists() and cached_challenge.exists():
        _, _, _, cycles, paths, events = build_v5_dataset()
        train = pd.read_csv(cached_train)
        challenges = pd.read_csv(cached_challenge)
        stable_audit = pd.DataFrame()
    else:
        train, challenges, stable_audit, cycles, paths, events = build_v5_dataset()
        train, _ = materialize_parallel_rows(train, paths, events)
    result_path = C.OUTPUT_ROOT / "v5_parallel_model_results.csv"
    bundles, reconstruction = _load_frozen_m1(train, cycles, paths, result_path)
    old_latency_path = output_root / "old_runtime_m1_nvc_event_latency.csv"
    old_comp_path = output_root / "old_runtime_m1_t0_t1_comparison.csv"
    old_latency = pd.read_csv(old_latency_path) if old_latency_path.exists() else pd.DataFrame()
    old_comparison = pd.read_csv(old_comp_path) if old_comp_path.exists() else pd.DataFrame()
    streams=[]; triggers=[]; latency=[]; trajectories=[]; audits=[]; configs=[]; vectors=[]; future=[]; candidate_traces={}
    for subject in C.SUBJECTS:
        split=C.SPLITS[subject]; subj_stream=[]; subj_trig=[]
        candidate_traces[subject] = _candidate_trace_for_subject(subject, list(split["train"])+list(split["test"]), paths)
        for cyc in split["test"]:
            _, s, tr = _replay_cycle(subject, cyc, bundles[subject], paths, events, candidate_traces[subject][str(cyc)]); subj_stream.append(s); subj_trig.append(tr)
        ss=pd.concat(subj_stream,ignore_index=True); tt=pd.concat(subj_trig,ignore_index=True) if subj_trig else pd.DataFrame(); streams.append(ss); triggers.append(tt)
        old_subj = old_latency[old_latency.animal.eq(subject)].copy() if len(old_latency) else pd.DataFrame()
        lr,traj=_latency_rows(subject,split["test"],ss,bundles[subject],events,paths,old_subj); latency.append(lr); trajectories.append(traj)
        future.append(_future_audit(subject,bundles[subject],paths,events,split["test"],candidate_traces[subject]))
        configs.append((subject,_frozen_config(subject,bundles[subject],split))); vectors.append(_vectors(subject,split,ss,events,paths))
    stream=pd.concat(streams,ignore_index=True); trig=pd.concat(triggers,ignore_index=True); lat=pd.concat(latency,ignore_index=True); traj=pd.concat(trajectories,ignore_index=True); fut=pd.concat(future,ignore_index=True); vec=pd.concat(vectors,ignore_index=True)
    # Primary T0/T1 table; false positives come from complete-cycle replay,
    # while event sensitivity/latency is matched to the nine teacher NVCs.
    comp=[]
    for subject in C.SUBJECTS:
        split=C.SPLITS[subject]; e=lat[lat.animal.eq(subject)]; st=stream[stream.animal.eq(subject)]; tg=trig[trig.animal.eq(subject)]
        # The replay grid contains extra, non-regular probe rows around each
        # teacher event (confirm and +0.25/+0.5/+1 s).  They are useful for
        # latency auditing but must not inflate the denominator used for
        # FP/hour.  Count only the registered 0.25-s streaming updates with a
        # valid score; this is the actual causal replay time available to the
        # detector.
        step = max(1, int(round(UPDATE_S * C.DP_FS_HZ)))
        regular_valid = st[(st.score.notna()) & ((st.decision_index.astype(int) % step) == 0)]
        valid_s=float(regular_valid.shape[0]*UPDATE_S); cycles_n=len(split["test"]); hours=valid_s/3600.0
        for policy,detcol,latcol in (("T0","detected_T0","latency_from_onset_T0_s"),("T1","detected_T1","latency_from_onset_T1_s")):
            g=tg[tg.policy.eq(policy)]; fp=g[~g.posthoc_label.isin(NVC_TRIGGER_LABELS)]; d=e[detcol].fillna(False); l=e.loc[d,latcol].dropna()
            tp=int(g.posthoc_label.isin(NVC_TRIGGER_LABELS).sum())
            comp.append({"animal":subject,"policy":policy,"nvc_detected":f"{int(d.sum())}/{len(e)}","nvc_detected_n":int(d.sum()),"nvc_total":int(len(e)),"sensitivity":float(d.mean()) if len(e) else np.nan,"coverage":float(e.score_at_confirm.notna().mean()) if len(e) else np.nan,"PPV":float(tp/len(g)) if len(g) else np.nan,"FP_total":int(len(fp)),"FP/cycle":float(len(fp)/cycles_n) if cycles_n else np.nan,"FP/hour":float(len(fp)/hours) if hours>0 else np.nan,"median_onset_latency_s":float(l.median()) if len(l) else np.nan,"IQR_onset_latency_s":float(l.quantile(.75)-l.quantile(.25)) if len(l) else np.nan,"max_onset_latency_s":float(l.max()) if len(l) else np.nan,"median_confirm_latency_s":float(e.loc[d,latcol.replace('onset','confirm')].dropna().median()) if d.any() else np.nan,"valid_replay_duration_s":valid_s})
    comparison=pd.DataFrame(comp)
    # Post-hoc trigger overlap and detailed T1 false-trigger audit.
    overlap=[]
    for (subject,policy),g in trig.groupby(["animal","policy"]):
        overlap.append({"animal":subject,"policy":policy,"trigger_total":len(g),"NVC-associated":int(g.posthoc_label.isin(NVC_TRIGGER_LABELS).sum()),"Stable FP":int((g.posthoc_label=="STABLE_FALSE_TRIGGER").sum()),"PREVOID-associated":int((g.posthoc_label=="PREVOID_ASSOCIATED_TRIGGER").sum()),"VOID-associated":int((g.posthoc_label=="VOID_ASSOCIATED_TRIGGER").sum()),"Other":int((~g.posthoc_label.isin(NVC_TRIGGER_LABELS+('STABLE_FALSE_TRIGGER','PREVOID_ASSOCIATED_TRIGGER','VOID_ASSOCIATED_TRIGGER'))).sum()),"posthoc_only":True})
    overlap=pd.DataFrame(overlap)
    t1false=trig[(trig.policy=="T1") & (~trig.posthoc_label.isin(NVC_TRIGGER_LABELS))].copy()
    b15=trig[(trig.animal=="STxF26") & (trig.cycle_id=="B15")].groupby("policy").size().rename("trigger_count").reset_index(); b15["animal"]="STxF26"; b15["cycle_id"]="B15"; b15["cycle_duration_s"]=float(cycles[(cycles.subject=="STxF26")&(cycles.cycle_id=="B15")].cycle_duration_s.iloc[0])
    b15_stream=stream[(stream.animal=="STxF26")&(stream.cycle_id=="B15")]
    b15_regular_valid=b15_stream[(b15_stream.score.notna()) & ((b15_stream.decision_index.astype(int) % max(1,int(round(UPDATE_S*C.DP_FS_HZ))))==0)]
    b15["valid_replay_duration_s"]=float(len(b15_regular_valid)*UPDATE_S)
    b15["FP/hour"]=b15.trigger_count/(b15.valid_replay_duration_s/3600.0) if len(b15) and b15.valid_replay_duration_s.iloc[0] > 0 else np.nan
    # Detailed B15 old/new trigger audit.  Old rows are read from the
    # preserved pre-correction replay and are never fed back into fitting.
    b15_detail = []
    old_stream_path = output_root / "old_runtime_m1_full_cycle_replay.csv"
    if old_stream_path.exists():
        old_s = pd.read_csv(old_stream_path)
        old_s["old_t0_flag"] = old_s.t0_trigger.astype(str).str.casefold().isin(("true", "1"))
        old_s["old_t1_flag"] = old_s.t1_trigger.astype(str).str.casefold().isin(("true", "1"))
        old_s = old_s[(old_s.animal == "STxF26") & (old_s.cycle_id.astype(str) == "B15") & (old_s.old_t0_flag | old_s.old_t1_flag)]
        trc = candidate_traces.get("STxF26", {}).get("B15", {})
        for rr in old_s.itertuples(index=False):
            ii = int(rr.decision_index); ca = bool(ii < len(trc.get("candidate_active", [])) and trc["candidate_active"][ii])
            cid = str(trc.get("candidate_event_id", [""])[ii]) if ii < len(trc.get("candidate_event_id", [])) else ""
            for pol, flag in (("T0", bool(rr.old_t0_flag)), ("T1", bool(rr.old_t1_flag))):
                if flag:
                    b15_detail.append({"runtime":"OLD_RUNTIME","policy":pol,"trigger_time_s":float(rr.time_s),"score":float(rr.score),"candidate_active_at_trigger":ca,"candidate_event_id":cid,"candidate_onset_s":float(trc["candidate_onset_time"][ii]) if ca else np.nan,"candidate_end_s":float(trc["candidate_event_end"][ii]) if ca else np.nan,"pressure_delta":float(trc["residual"][ii]) if ii < len(trc.get("residual", [])) and _finite(trc["residual"][ii]) else np.nan,"pressure_slope":np.nan,"posthoc_label":"UNRESOLVED_OLD_RUNTIME","diagnosis":"RUNTIME_GATING_PROBLEM" if not ca else "HARD_NEGATIVE_PRESSURE_CANDIDATE"})
    for rr in trig[(trig.animal == "STxF26") & (trig.cycle_id.astype(str) == "B15")].itertuples(index=False):
        b15_detail.append({"runtime":"CORRECTED_RUNTIME","policy":rr.policy,"trigger_time_s":float(rr.trigger_time_s),"score":float(rr.score),"candidate_active_at_trigger":bool(rr.candidate_active_at_trigger),"candidate_event_id":rr.candidate_event_id,"candidate_onset_s":rr.candidate_onset_s,"candidate_end_s":rr.candidate_end_s,"pressure_delta":rr.pressure_delta,"pressure_slope":rr.pressure_slope,"posthoc_label":rr.posthoc_label,"diagnosis":"HARD_NEGATIVE_PRESSURE_CANDIDATE" if rr.posthoc_label != "NVC_EARLY_TP" and rr.posthoc_label != "NVC_ON_EVENT_TP" else "NVC_TRIGGER"})
    b15_detail = pd.DataFrame(b15_detail)
    corrected_runtime = comparison.copy(); corrected_runtime.insert(0, "runtime", "CORRECTED_RUNTIME")
    if len(old_comparison):
        old_runtime = old_comparison.copy(); old_runtime.insert(0, "runtime", "OLD_RUNTIME")
        runtime_comparison = pd.concat([old_runtime, corrected_runtime], ignore_index=True, sort=False)
    else:
        runtime_comparison = corrected_runtime
    comparison.to_csv(output_root/"m1_t0_t1_comparison.csv",index=False); runtime_comparison.to_csv(output_root/"m1_old_vs_corrected_runtime.csv",index=False); lat.to_csv(output_root/"m1_nvc_event_latency.csv",index=False); t1false.to_csv(output_root/"m1_false_trigger_audit.csv",index=False); overlap.to_csv(output_root/"m1_challenge_overlap.csv",index=False); b15.to_csv(output_root/"F26_B15_t0_t1_audit.csv",index=False); b15_detail.to_csv(output_root/"F26_B15_trigger_audit.csv",index=False); stream.to_csv(output_root/"m1_full_cycle_replay.csv",index=False); traj.to_csv(output_root/"m1_nvc_score_trajectories.csv",index=False); reconstruction.to_csv(output_root/"frozen_reconstruction_audit.csv",index=False); fut.to_csv(output_root/"future_leakage_audit.csv",index=False); vec.to_csv(output_root/"F37_F26_streaming_test_vectors.csv",index=False)
    for subject,cfg in configs: (output_root/f"{subject}_m1_frozen_config.json").write_text(json.dumps(_jsonable(cfg),ensure_ascii=False,indent=2),encoding="utf-8")
    completeness={"all_six_test_cycles":bool(set(stream.cycle_id.astype(str))==set(C.SPLITS["STxF37"]["test"]+C.SPLITS["STxF26"]["test"])),"nine_nvc_latency_rows":len(lat)==9,"future_leakage_pass":bool(len(fut) and fut["pass"].all() and fut["candidate_state_unchanged"].all()),"two_frozen_configs":len(configs)==2,"streaming_vectors":bool(len(vec)),"challenge_overlap":bool(len(overlap))}
    summary={"version":"V5_FINAL_OFFLINE_VALIDATION_1.0","model_name":"M1 P-EARLY","subjects":list(C.SUBJECTS),"splits":C.SPLITS,"update_interval_s":UPDATE_S,"baseline_window_s":BASELINE_S,"t1_rule":T1_RULE,"lockout_s":LOCKOUT_S,"lockout_source":"not present in existing V5 runtime","runtime":"CORRECTED_RUNTIME_PRESSURE_CANDIDATE_GATED","candidate_gate":"existing causal adaptive pressure detector; no teacher/urine/future input","model_selection_complete":True,"offline_trigger_validation_complete":bool(all(completeness.values())),"completeness":completeness,"development_only":True,"deployment_ready":False,"stimulation_enabled":False,"reconstruction_audit":reconstruction.to_dict(orient="records"),"comparison":comparison.to_dict(orient="records"),"runtime_comparison":runtime_comparison.to_dict(orient="records"),"posthoc_only":True}
    (output_root/"v5_final_summary.json").write_text(json.dumps(_jsonable(summary),ensure_ascii=False,indent=2),encoding="utf-8")
    report=["# V5 Final Offline Validation — M1 P-EARLY","","## Frozen scope","- Only the already-selected M1 P-EARLY is evaluated; M2/M3/M4 are not rerun.","- F37 B01–B04→B05–B07; F26 B01–B13→B14–B16; no pooling.","- PREVOID/VOID are post-hoc audit labels only; no online veto.","","## Runtime correction","- OLD_RUNTIME allowed score-positive segments to trigger directly.","- CORRECTED_RUNTIME requires an existing causal pressure candidate, binds the latch to candidate_event_id, and resets at pressure recovery.","- Frozen M1 feature schema, scaler, coefficients, thresholds, and prospective splits are unchanged.","", "## OLD_RUNTIME vs CORRECTED_RUNTIME", runtime_comparison.to_markdown(index=False),"","## T0/T1 comparison",comparison.to_markdown(index=False),"","## F26 B15",b15.to_markdown(index=False),"", "### F26 B15 trigger-level audit", b15_detail.to_markdown(index=False),"","## Post-hoc trigger overlap",overlap.to_markdown(index=False),"","## Frozen reconstruction audit",reconstruction.to_markdown(index=False),"","## Completeness",json.dumps(completeness,ensure_ascii=False,indent=2),"","development_only=true; deployment_ready=false; stimulation_enabled=false."]
    (output_root/"V5_FINAL_OFFLINE_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    from .visualization import generate_plots
    generate_plots(C.OUTPUT_ROOT)
    return summary


if __name__ == "__main__":
    run_final_validation()
