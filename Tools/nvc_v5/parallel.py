"""V5 registered five-model parallel individualized experiment.

This module is intentionally separate from the historic V5-A runner.  It
keeps the fixed F37/F26 prospective split and fits exactly five independent
candidates (R0/M1/M2/M3/M4).  Every model receives the same materialized
teacher rows, and an exception in one candidate is recorded rather than
preventing the other candidates from running.
"""
from __future__ import annotations

from pathlib import Path
import json
import traceback
import warnings
import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from .source_adapter import _load_pair
from .spectral_features import _periodogram
from .feature_extraction import extract_v4_features
from . import config as C
from .data_adapter import build_v5_dataset, event_confirm_time
from .development import choose_high_coverage_features
from .modeling import (apply_fusion, apply_model, fit_fusion, fit_individual,
                       metrics, score_model, select_threshold)

warnings.filterwarnings("ignore", message="X has feature names, but StandardScaler was fitted without feature names")


def _finite(x):
    try:
        return bool(np.isfinite(float(x)))
    except (TypeError, ValueError):
        return False


def _cycle_start_idx(cycle, seconds=25.0):
    return int(round(seconds * C.DP_FS_HZ))


def _trailing_p_feature(cycle, idx, baseline_end):
    p = np.asarray(cycle.get("bladder_pressure_mmHg", []), dtype=float)
    valid = np.asarray(cycle.get("cmg_valid_100hz", np.ones(p.size, bool)), bool)
    a = max(0, int(idx) - int(round(1.0 * C.DP_FS_HZ)) + 1)
    x = p[a:int(idx) + 1]
    ok = valid[a:int(idx) + 1] & np.isfinite(x)
    if ok.sum() < int(round(0.95 * C.DP_FS_HZ)):
        return np.nan
    b0 = max(0, int(baseline_end) - int(round(25 * C.DP_FS_HZ)))
    b = p[b0:int(baseline_end)]
    bok = valid[b0:int(baseline_end)] & np.isfinite(b)
    if bok.sum() < int(round(0.8 * 25 * C.DP_FS_HZ)):
        return np.nan
    med = float(np.median(b[bok])); scale = max(float(1.4826 * np.median(np.abs(b[bok] - med))), np.finfo(float).eps)
    return float(np.std((x[ok] - med) / scale))


def extract_p_early_features(cycle, index, onset_index=None, decision_time_s=None):
    """Materialize M1 P-EARLY without event-age-dependent variability."""
    idx = int(index)
    onset = idx if onset_index is None else max(0, min(idx, int(onset_index)))
    f, reason = extract_v4_features(cycle, idx, onset, decision_time_s)
    out = {k: f.get(k, np.nan) for k in C.P_EARLY_FEATURES}
    # The causal baseline ends at the event onset, but variability is always
    # the fixed trailing [t-1 s, t] window.  It therefore remains defined for
    # early contractions and cannot encode event age.
    out["p_trailing_variability_1s"] = _trailing_p_feature(cycle, idx, min(idx, onset))
    bad = [k for k in C.P_EARLY_FEATURES if not _finite(out.get(k))]
    if bad and not reason:
        reason = "P_EARLY_FEATURE_MISSING:" + "|".join(bad)
    return out, str(reason or "")


def _event_onsets(events):
    out = {}
    for r in events.itertuples(index=False):
        if _finite(getattr(r, "start_index", np.nan)):
            out[str(getattr(r, "event_uid", ""))] = int(float(r.start_index))
    return out


def materialize_parallel_rows(rows, paths, events):
    """Recompute registered V4 rows with P-EARLY and preserve V4 EUS fields."""
    out = rows.copy()
    cycle_cache = {}
    starts = _event_onsets(events)
    for i, r in rows.iterrows():
        key = (str(r.subject), str(r.cycle_id))
        if key not in cycle_cache:
            cycle_cache[key] = _load_pair(paths[key])["cycle"]
        cyc = cycle_cache[key]
        idx = int(r.decision_index)
        uid = str(r.source_event_uid) if _finite(r.source_event_uid) or str(r.source_event_uid) not in ("", "nan", "None") else ""
        onset = starts.get(uid, max(0, idx - int(round(2 * C.DP_FS_HZ))))
        pf, reason = extract_p_early_features(cyc, idx, onset, float(r.decision_time_s))
        for k, v in pf.items():
            out.at[i, k] = v
        # Keep the compact EUS schema synchronized with the same causal row.
        ef, ereason = extract_v4_features(cyc, idx, onset, float(r.decision_time_s))
        for k in C.E_EARLY_FEATURES:
            if k in ef:
                out.at[i, k] = ef[k]
        if reason or ereason:
            old = str(r.get("feature_failure_reason", "") or "")
            out.at[i, "feature_failure_reason"] = "|".join(dict.fromkeys(x for x in (old, reason, ereason) if x))
    return out, cycle_cache


def _band_vector(x, fs, bands):
    f, p = _periodogram(np.asarray(x, float), float(fs))
    if f.size == 0:
        return [np.nan for _ in bands]
    return [float(np.trapz(p[(f >= lo) & (f <= hi)], f[(f >= lo) & (f <= hi)])) if np.any((f >= lo) & (f <= hi)) else np.nan for lo, hi in bands]


def _robust_scale(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan
    med = float(np.median(x)); mad = max(float(1.4826 * np.median(np.abs(x - med))), np.finfo(float).eps)
    return med, mad


def _continuous_lasso_samples(subject, train_cycles, paths, stride_s=5.0):
    """Create label-free calibration samples for EUS -> normalized pressure."""
    rows = []
    for cyc_id in train_cycles:
        key = (str(subject), str(cyc_id))
        if key not in paths:
            continue
        cyc = _load_pair(paths[key])["cycle"]
        p = np.asarray(cyc.get("bladder_pressure_mmHg", []), float)
        t = np.asarray(cyc.get("t_abs_s", []), float)
        if p.size == 0 or t.size != p.size:
            continue
        first = int(round(27 * C.DP_FS_HZ)); step = max(1, int(round(stride_s * C.DP_FS_HZ)))
        for idx in range(first, p.size, step):
            onset = idx - int(round(2 * C.DP_FS_HZ))
            tt = float(t[idx]); ef, reason = _causal_eus_fast(cyc, idx)
            if reason or not all(_finite(ef.get(k)) for k in C.M4_FEATURES):
                continue
            b0 = onset - int(round(25 * C.DP_FS_HZ))
            if b0 < 0:
                continue
            med, mad = _robust_scale(p[b0:onset])
            if not _finite(med) or not _finite(mad):
                continue
            rows.append({"subject": subject, "cycle_id": str(cyc_id), "decision_index": idx,
                         "decision_time_s": tt, "target_pressure_norm": float((p[idx] - med) / mad), **ef})
    return pd.DataFrame(rows)


def _causal_eus_fast(cycle, idx):
    """Fast compact EUS bands using only the current two seconds and a
    cycle-initial 25-s calibration segment (both are strictly historical)."""
    raw = np.asarray(cycle.get("eus_raw_native", []), float)
    times = np.asarray(cycle.get("t_eus_abs_native", []), float)
    fs = float(cycle.get("eus_fs_native", np.nan))
    pt = np.asarray(cycle.get("t_abs_s", []), float)
    if raw.size == 0 or raw.size != times.size or pt.size <= idx or not _finite(fs):
        return {}, "RAW_EUS_UNAVAILABLE"
    now = float(pt[idx]); start = float(pt[0])
    ci = np.flatnonzero((times <= now + 1e-9) & (times >= now - 2.0 - 1e-9))
    bi = np.flatnonzero((times < start + 25.0 - 1e-9) & (times >= start - 1e-9))
    if ci.size < .90 * 2 * fs or bi.size < .90 * 25 * fs:
        return {}, "EUS_HISTORY_INSUFFICIENT"
    cv = _band_vector(raw[ci], fs, C.EUS_FAST_BANDS)
    # The baseline is the first 25 seconds of a cycle.  It is available for
    # every admissible decision point, so cache its FFT-derived bands on the
    # cycle object instead of recomputing a 250-k sample FFT at every update.
    cache = cycle.get("_v5_eus_initial_bandpower")
    if cache is None:
        cache = _band_vector(raw[bi], fs, C.EUS_FAST_BANDS)
        cycle["_v5_eus_initial_bandpower"] = cache
    bv = cache
    if not all(_finite(a) and _finite(b) for a, b in zip(cv, bv)):
        return {}, "EUS_INVALID"
    return {f"eus_relative_log_bandpower_{int(lo)}_{int(hi)}": float(np.log((a + C.EPSILON) / (b + C.EPSILON))) for (lo, hi), a, b in zip(C.EUS_FAST_BANDS, cv, bv)}, ""


def _fit_lasso(subject, train_cycles, paths, train_rows):
    cont = _continuous_lasso_samples(subject, train_cycles, paths)
    if cont.empty or cont.cycle_id.nunique() < 2:
        raise ValueError("INSUFFICIENT_LASSO_CALIBRATION_CYCLE_BLOCKS")
    X = cont[list(C.M4_FEATURES)].to_numpy(float); y = cont.target_pressure_norm.to_numpy(float)
    groups = cont.cycle_id.astype(str).to_numpy()
    alphas = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)
    cv_rows = []
    for alpha in alphas:
        errs = []
        for held in sorted(set(groups)):
            tr = groups != held; va = groups == held
            if tr.sum() < 5 or not va.any():
                continue
            pipe = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=alpha, max_iter=10000, random_state=C.RANDOM_STATE))])
            pipe.fit(X[tr], y[tr]); errs.append(mean_squared_error(y[va], pipe.predict(X[va])))
        cv_rows.append({"alpha": alpha, "cv_mse": float(np.mean(errs)) if errs else np.nan, "n_folds": len(errs)})
    cv = pd.DataFrame(cv_rows)
    chosen = float(cv.sort_values(["cv_mse", "alpha"], na_position="last").iloc[0].alpha)
    model = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso(alpha=chosen, max_iter=10000, random_state=C.RANDOM_STATE))])
    model.fit(X, y); model.fit_features_ = tuple(C.M4_FEATURES); model.fit_cycles_ = tuple(sorted(set(groups)))
    # Threshold selection is supervised only after the regression is fitted;
    # the regression target itself never contains NVC/urine/future labels.
    q = train_rows.copy(); q["m4_score"] = np.nan
    ok = np.isfinite(q[list(C.M4_FEATURES)].to_numpy(float)).all(axis=1)
    if ok.any(): q.loc[ok, "m4_score"] = model.predict(q.loc[ok, list(C.M4_FEATURES)])
    usable = q[q.m4_score.notna() & q.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)]
    if usable.target.nunique() < 2:
        raise ValueError("INSUFFICIENT_LASSO_LABEL_THRESHOLD_ROWS")
    threshold, source = select_threshold(usable.target, usable.m4_score)
    return {"kind": "lasso", "model": model, "threshold": threshold,
            "threshold_source": source, "features": tuple(C.M4_FEATURES),
            "alpha": chosen, "cv": cv, "continuous_samples": cont,
            "regression_target": "normalized_current_pressure"}


def _score_lasso(bundle, frame):
    out = frame.copy(); out["score"] = np.nan
    if bundle.get("kind") != "lasso":
        return out
    ok = np.isfinite(out.reindex(columns=list(bundle["features"]), fill_value=np.nan).to_numpy(float)).all(axis=1)
    if ok.any(): out.loc[ok, "score"] = bundle["model"].predict(out.loc[ok, list(bundle["features"])])
    out["threshold"] = bundle.get("threshold", np.nan); out["model_scorable"] = out.score.notna(); out["model"] = "M4"; out["predicted_nvc"] = out.score >= out.threshold
    return out


def _score_bundle(bundle, frame, name):
    if bundle.get("kind") == "base":
        return apply_model(bundle["model"], bundle["threshold"], frame, name)
    if bundle.get("kind") == "fusion":
        f = frame.copy()
        for col in C.COUPLING_FEATURES:
            if col not in f.columns:
                f[col] = np.nan
        return apply_fusion(bundle, f, name)
    if bundle.get("kind") == "lasso":
        out = _score_lasso(bundle, frame); out["model"] = name; return out
    out = frame.copy(); out["score"] = np.nan; out["threshold"] = np.nan; out["model"] = name; out["predicted_nvc"] = False; out["model_failure_reason"] = bundle.get("reason", "MODEL_NOT_FIT"); return out


def _stream_row(cycle, idx):
    """Low-cost causal row for complete-cycle replay."""
    p = np.asarray(cycle.get("bladder_pressure_mmHg", []), float); t = np.asarray(cycle.get("t_abs_s", []), float)
    if idx < int(round(27 * C.DP_FS_HZ)) or idx >= p.size or t.size != p.size:
        return None
    b_end = idx - int(round(2 * C.DP_FS_HZ)); b0 = b_end - int(round(25 * C.DP_FS_HZ))
    if b0 < 0: return None
    ps_cache = cycle.get("_v5_pressure_initial_scale")
    if ps_cache is None:
        ps_cache = _robust_scale(p[b0:b_end]); cycle["_v5_pressure_initial_scale"] = ps_cache
    med, scale = ps_cache
    if not _finite(med) or not _finite(scale): return None
    h = (p[idx - int(round(2 * C.DP_FS_HZ)) + 1:idx + 1] - med) / scale
    if h.size < 50 or not np.isfinite(h).all(): return None
    d = np.diff((p[idx - int(round(1 * C.DP_FS_HZ)):idx + 1] - med) / scale) * C.DP_FS_HZ
    d05 = d[-50:]
    auc = float(np.trapz(np.maximum(h, 0), dx=1 / C.DP_FS_HZ))
    ps = _band_vector(p[idx - int(round(5 * C.DP_FS_HZ)) + 1:idx + 1] - med, C.DP_FS_HZ, ((.2, .6), (.2, 20)))
    bps = cycle.get("_v5_pressure_initial_bandpower")
    if bps is None:
        bps = _band_vector(p[b0:b_end] - med, C.DP_FS_HZ, ((.2, .6), (.2, 20))); cycle["_v5_pressure_initial_bandpower"] = bps
    cur = p[idx - int(round(5 * C.DP_FS_HZ)) + 1:idx + 1] - med
    f, pp = _periodogram(cur, C.DP_FS_HZ); m = (f >= .2) & (f <= 20)
    ent = np.nan
    if pp.size and np.any(m) and np.nansum(pp[m]) > 0:
        q = pp[m] / np.nansum(pp[m]); ent = float(-np.sum(q * np.log(q + C.EPSILON)) / np.log(max(q.size, 2)))
    row = {"p_current_delta": float((p[idx] - med) / scale), "p_peak_delta": float(np.max(h)), "p_threshold_above_duration": float(np.mean(h > 3.68) * 2),
           "p_slope_0p5s": float(np.mean(d05)), "p_slope_1s": float(np.mean(d)), "p_max_positive_dpdt": float(np.max(d)),
           "p_positive_dpdt_occupancy": float(np.mean(d > 0)), "p_auc": auc, "p_auc_growth": float(auc / max(2.0 * scale, C.EPSILON)),
           "pressure_curvature": float(np.mean(d05) - np.mean(d)), "peak_to_current_drop": float(np.max(h) - h[-1]),
           "p_trailing_variability_1s": float(np.std(((p[idx - 99:idx + 1] - med) / scale))) if idx >= 99 else np.nan,
           "pressure_power_0p2_0p6_rel": float(np.log((ps[0] + C.EPSILON) / (bps[0] + C.EPSILON))) if _finite(ps[0]) and _finite(bps[0]) else np.nan,
           "pressure_auc_0p2_20_rel": float(np.log((ps[1] + C.EPSILON) / (bps[1] + C.EPSILON))) if _finite(ps[1]) and _finite(bps[1]) else np.nan,
           "pressure_spectral_entropy": ent}
    env = np.asarray(cycle.get("eus_envelope_100hz", cycle.get("eus_envelope_mV", [])), float); ev = np.asarray(cycle.get("eus_valid_100hz", np.ones(env.size, bool)), bool)
    if env.size == p.size:
        eb = env[b0:b_end]; okb = ev[b0:b_end] & np.isfinite(eb); ec = env[idx - 199:idx + 1]; oke = ev[idx - 199:idx + 1] & np.isfinite(ec)
        em, es = _robust_scale(eb[okb])
        if _finite(em) and _finite(es) and oke.mean() >= .9:
            z = (ec[oke] - em) / es; row.update({"eus_relative_rms": float(np.sqrt(np.mean(z*z))), "eus_relative_amplitude": float((env[idx]-em)/es), "eus_envelope_slope": float(np.polyfit(np.arange(z.size)/C.DP_FS_HZ, z, 1)[0]) if z.size > 1 else 0., "eus_tonic_occupancy": float(np.mean(z > 3)), "eus_burst_occupancy": float(np.mean(z > 5)), "eus_short_term_variability": float(np.std(z))})
    for k in C.E_EARLY_FEATURES:
        row.setdefault(k, np.nan)
    ef, _ = _causal_eus_fast(cycle, idx)
    row.update(ef)
    if env.size == p.size:
        a = max(1, idx - 199); ppv = p[a:idx + 1]; eev = env[a:idx + 1]; ok = ev[a:idx + 1] & np.isfinite(ppv) & np.isfinite(eev)
        if ok.sum() > 5:
            ez = (eev[ok] - em) / es if _finite(em) and _finite(es) else np.array([]); dp = np.r_[0, np.diff(ppv[ok]) * C.DP_FS_HZ]
            row["causal_pressure_eus_corr"] = float(np.corrcoef(ez, dp)[0, 1]) if ez.size and np.std(ez) and np.std(dp) else np.nan
            row["pressure_eus_coactivation"] = float(np.mean((dp > 0) & (ez > 3))) if ez.size else np.nan
            posp = np.flatnonzero(dp > 0); pose = np.flatnonzero(ez > 3)
            row["eus_activation_latency_s"] = float((pose[0] - posp[0]) / C.DP_FS_HZ) if posp.size and pose.size else np.nan
    row.update({"subject": str(cycle.get("subject", "")), "decision_index": int(idx), "decision_time_s": float(t[idx]), "teacher_label": "STREAM", "target": np.nan})
    return row


def event_replay(subject, test_cycles, bundles, events, paths):
    detail, summary = [], []
    ev = events[(events.subject == subject) & events.cycle_id.astype(str).isin(tuple(test_cycles)) & events.teacher_label.eq("NVC_CORE")]
    for er in ev.itertuples(index=False):
        key = (str(subject), str(er.cycle_id)); cyc = _load_pair(paths[key])["cycle"]; t = np.asarray(cyc["t_abs_s"], float)
        i0 = int(er.start_index); ic = int(er.confirm_index); end = min(len(t)-1, ic + int(round(2*C.DP_FS_HZ)))
        grid = set(range(i0, end+1, int(round(C.STREAM_UPDATE_S*C.DP_FS_HZ))))
        grid.add(ic)
        grid.update(min(end, ic + int(round(d * C.DP_FS_HZ))) for d in C.STREAM_DELAYS_S)
        grid = sorted(grid)
        for name, b in bundles.items():
            vals = []
            for idx in grid:
                # Use the same fast causal materializer as full-cycle replay.
                # It uses a fixed cycle-initial calibration window and a
                # trailing two-second event context, avoiding repeated native
                # EUS STFTs while retaining the registered 0.25-s updates.
                row = _stream_row(cyc, idx)
                rs = "" if row is not None else "PRESSURE_HISTORY_INSUFFICIENT"
                if row is None:
                    row = {"subject": subject, "cycle_id": str(er.cycle_id)}
                row.update({"subject": subject, "cycle_id": str(er.cycle_id), "teacher_label": "NVC_CORE", "target": 1, "sample_uid": str(er.event_uid)})
                sc = float(_score_bundle(b, pd.DataFrame([row]), name).score.iloc[0]) if b.get("kind") != "failed" else np.nan
                vals.append((idx, float(t[idx]), sc)); detail.append({"animal": subject, "model": name, "sample_uid": str(er.event_uid), "cycle_id": str(er.cycle_id), "update_time_s": float(t[idx]), "relative_to_confirm_s": float(t[idx]-t[ic]), "score": sc, "threshold": b.get("threshold", np.nan), "feature_failure_reason": rs, "replay_type": "NVC_EVENT_CAUSAL_0P25S"})
            th = float(b.get("threshold", np.nan)); ge = [i for i,v in enumerate(vals) if _finite(v[2]) and _finite(th) and v[2] >= th]
            pairs = [i for i in range(1,len(vals)) if _finite(vals[i-1][2]) and _finite(vals[i][2]) and _finite(th) and vals[i-1][2] >= th and vals[i][2] >= th]
            def at(rel):
                z = [v for v in vals if abs(v[1]-t[ic]-rel) < .02]
                return z[0][2] if z else np.nan
            summary.append({"animal": subject, "model": name, "sample_uid": str(er.event_uid), "cycle_id": str(er.cycle_id), "candidate_onset_s": float(t[i0]), "confirm_time_s": float(t[ic]), "t0_detected": bool(ge), "t1_detected": bool(pairs), "t0_first_crossing_s": vals[ge[0]][1] if ge else np.nan, "t1_first_crossing_s": vals[pairs[0]][1] if pairs else np.nan, "score_at_confirm": at(0), "score_at_plus_0p25s": at(.25), "score_at_plus_0p5s": at(.5), "score_at_plus_1s": at(1), "score_at_plus_2s": at(2), "threshold": th, "replay_type": "NVC_EVENT_SUMMARY"})
    return pd.DataFrame(detail), pd.DataFrame(summary)


def full_cycle_replay(subject, test_cycles, bundles, paths, events=None):
    rows, summaries = [], []
    for cyc_id in test_cycles:
        key = (str(subject), str(cyc_id)); cyc = _load_pair(paths[key])["cycle"]; t = np.asarray(cyc.get("t_abs_s", []), float)
        step = int(round(C.STREAM_UPDATE_S * C.DP_FS_HZ)); start = int(round(27*C.DP_FS_HZ))
        # Recompute every registered 0.25-s update.  Cycle-level baseline FFTs
        # are cached by _stream_row/_causal_eus_fast, so this remains tractable
        # while preserving an actual causal T1 two-consecutive-update test.
        compute_step = step
        scores = {name: [] for name in bundles}
        compute_grid = list(range(start, len(t), compute_step))
        for idx in compute_grid:
            row = _stream_row(cyc, idx)
            if row is None: continue
            row["subject"] = subject; row["cycle_id"] = str(cyc_id)
            for name, b in bundles.items():
                out = _score_bundle(b, pd.DataFrame([row]), name); sc = float(out.score.iloc[0]) if _finite(out.score.iloc[0]) else np.nan; th = b.get("threshold", np.nan)
                scores[name].append((idx, float(t[idx]), sc))
        # Emit the complete 0.25-s causal stream.
        output_scores = {name: [] for name in bundles}
        score_maps = {name: {v[0]: v[2] for v in vals} for name, vals in scores.items()}
        out_grid = list(range(start, len(t), step))
        for idx in out_grid:
            for name, b in bundles.items():
                sc = score_maps[name].get(idx, np.nan); th = b.get("threshold", np.nan)
                rows.append({"animal": subject, "model": name, "cycle_id": str(cyc_id), "update_time_s": float(t[idx]), "score": sc, "threshold": th, "t0_trigger": bool(_finite(sc) and _finite(th) and sc >= th), "replay_type": "FULL_CYCLE_CAUSAL_0P25S_CARRY_FORWARD_1S_COMPUTE"})
                output_scores[name].append((idx, float(t[idx]), sc))
        nvc_windows = []
        if events is not None:
            eg = events[(events.subject == subject) & events.cycle_id.astype(str).eq(str(cyc_id)) & events.teacher_label.eq("NVC_CORE")]
            for er in eg.itertuples(index=False):
                try:
                    a = float(er.start_s) if _finite(getattr(er, "start_s", np.nan)) else float(t[int(er.start_index)])
                    b = float(er.confirm_time_s) if _finite(getattr(er, "confirm_time_s", np.nan)) else float(t[int(er.confirm_index)])
                    nvc_windows.append((a, b + 2.0))
                except (TypeError, ValueError, IndexError):
                    continue
        duration_h = float((t[-1] - t[0]) / 3600.0) if t.size > 1 else np.nan
        for name, vals in output_scores.items():
            th = bundles[name].get("threshold", np.nan); t0 = [v for v in vals if _finite(v[2]) and _finite(th) and v[2] >= th]; rises = []
            prev = False
            for v in vals:
                cur = _finite(v[2]) and _finite(th) and v[2] >= th
                if cur and not prev: rises.append(v)
                prev = cur
            t1 = []
            prev_t1 = False
            for a, b in zip(vals[:-1], vals[1:]):
                cur_t1 = (abs(b[1] - a[1] - .25) < .02 and _finite(a[2]) and _finite(b[2]) and _finite(th) and a[2] >= th and b[2] >= th)
                if cur_t1 and not prev_t1:
                    t1.append(b)
                prev_t1 = cur_t1
            def in_nvc(x):
                return any(a <= x <= b for a, b in nvc_windows)
            t0_false = int(sum(not in_nvc(v[1]) for v in rises)); t1_false = int(sum(not in_nvc(v[1]) for v in t1))
            summaries.append({"animal": subject, "model": name, "cycle_id": str(cyc_id), "n_updates": len(vals), "scorable_updates": int(sum(_finite(v[2]) for v in vals)), "nvc_events": len(nvc_windows), "t0_event_hits": int(sum(in_nvc(v[1]) for v in rises)), "t1_event_hits": int(sum(in_nvc(v[1]) for v in t1)), "t0_trigger_count": len(rises), "t1_trigger_count": len(t1), "t0_false_triggers": t0_false, "t1_false_triggers": t1_false, "fp_per_cycle_t0": float(t0_false), "fp_per_cycle_t1": float(t1_false), "fp_per_hour_t0": float(t0_false / duration_h) if duration_h > 0 else np.nan, "fp_per_hour_t1": float(t1_false / duration_h) if duration_h > 0 else np.nan, "first_t0_s": rises[0][1] if rises else np.nan, "first_t1_s": t1[0][1] if t1 else np.nan, "threshold": th, "replay_type": "FULL_CYCLE_SUMMARY"})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def future_perturbation_audit(subject, bundles, paths, events):
    rows = []
    e = events[(events.subject == subject) & events.teacher_label.eq("NVC_CORE")]
    for er in e.head(2).itertuples(index=False):
        key = (subject, str(er.cycle_id)); base = _load_pair(paths[key])["cycle"]; idx = int(er.confirm_index); mutated = {k:(np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k,v in base.items()}
        for keyv in ("bladder_pressure_mmHg", "eus_envelope_100hz", "eus_envelope_mV", "eus_raw_native"):
            if isinstance(mutated.get(keyv), np.ndarray): mutated[keyv][idx+1:] = mutated[keyv][idx+1:] + 1e6
        pf1,_ = extract_p_early_features(base, idx, int(er.start_index), float(base["t_abs_s"][idx])); pf2,_ = extract_p_early_features(mutated, idx, int(er.start_index), float(base["t_abs_s"][idx]))
        for name,b in bundles.items():
            row1 = {"subject": subject, "cycle_id": str(er.cycle_id), "teacher_label":"NVC_CORE", **pf1}; row2 = {"subject": subject, "cycle_id": str(er.cycle_id), "teacher_label":"NVC_CORE", **pf2}
            s1 = _score_bundle(b,pd.DataFrame([row1]),name).score.iloc[0]; s2 = _score_bundle(b,pd.DataFrame([row2]),name).score.iloc[0]
            rows.append({"animal":subject,"model":name,"sample_uid":str(er.event_uid),"decision_index":idx,"score_original":s1,"score_future_mutated":s2,"unchanged":bool((not _finite(s1) and not _finite(s2)) or (_finite(s1) and _finite(s2) and abs(float(s1)-float(s2))<1e-10))})
    return pd.DataFrame(rows)


def _failed_bundle(reason):
    return {"kind":"failed","reason":str(reason),"threshold":np.nan}


def run_parallel(output_root=C.OUTPUT_ROOT):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    print("[V5.1] building fixed F37/F26 dataset", flush=True)
    train, challenges, stable_audit, cycles, paths, events = build_v5_dataset()
    print("[V5.1] materializing P-EARLY rows", flush=True)
    train, _cache = materialize_parallel_rows(train, paths, events)
    train.to_csv(output_root / "v5_parallel_training_samples.csv", index=False)
    challenges.to_csv(output_root / "v5_parallel_challenge_samples.csv", index=False)
    cycles.to_csv(output_root / "v5_parallel_cycle_manifest.csv", index=False)
    result_rows=[]; pred_rows=[]; pc_rows=[]; audits=[]; bundles_by={}; event_long=[]; event_sum=[]; full_long=[]; full_sum=[]; future=[]; lasso_audit=[]
    for subject in C.SUBJECTS:
        print(f"[V5.1] fitting parallel models for {subject}", flush=True)
        split=C.SPLITS[subject]; tr=train[(train.subject==subject)&train.cycle_id.astype(str).isin(split["train"])&train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy(); te=train[(train.subject==subject)&train.cycle_id.astype(str).isin(split["test"])&train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)].copy()
        bundles={}; selected_hc,_=choose_high_coverage_features(tr)
        specs={"R0":(tuple(selected_hc),"lr"),"M1":(C.P_EARLY_FEATURES,"lr"),"M2":(C.E_EARLY_FEATURES,"lr")}
        for name,(features,clf) in specs.items():
            print(f"[V5.1] {subject} {name}", flush=True)
            try:
                model,th,source,trsc,oof=fit_individual(tr,features,clf); bundles[name]={"kind":"base","model":model,"threshold":th,"features":tuple(features),"threshold_source":source}; pred=_score_bundle(bundles[name],te,name); met=metrics(pred,split["test"],cycles,name); status="OK"
                result_rows.append({k:v for k,v in met.items() if k!="per_cycle"}|{"animal":subject,"model":name,"status":status,"train_cycles":"|".join(split["train"]),"test_cycles":"|".join(split["test"]),"train_nvc":int((tr.teacher_label=="NVC_CORE").sum()),"threshold_source":source,"features":"|".join(features)})
                pc_rows.extend([{**r,"animal":subject,"policy":"T0"} for r in met["per_cycle"]]); pred["animal"]=subject; pred["policy"]="T0"; pred_rows.append(pred); audits.append({"animal":subject,"model":name,"status":"OK","classifier":clf,"threshold":th,"threshold_source":source,"features":"|".join(features),"fit_cycles":"|".join(model.fit_cycles_)})
            except Exception as exc:
                bundles[name]=_failed_bundle(exc); result_rows.append({"animal":subject,"model":name,"status":"FAILED","failure_reason":str(exc),"train_cycles":"|".join(split["train"]),"test_cycles":"|".join(split["test"])}); audits.append({"animal":subject,"model":name,"status":"FAILED","failure_reason":str(exc)})
        try:
            print(f"[V5.1] {subject} M3", flush=True)
            fb=fit_fusion(tr,C.P_EARLY_FEATURES,C.E_EARLY_FEATURES,"lr"); bundles["M3"]={"kind":"fusion",**fb}; pred=_score_bundle(bundles["M3"],te,"M3"); met=metrics(pred,split["test"],cycles,"M3"); status="OK" if fb.get("fusion_model") is not None else "NOT_RUN_INSUFFICIENT_FUSION_CALIBRATION"; result_rows.append({k:v for k,v in met.items() if k!="per_cycle"}|{"animal":subject,"model":"M3","status":status,"train_cycles":"|".join(split["train"]),"test_cycles":"|".join(split["test"]),"train_nvc":int((tr.teacher_label=="NVC_CORE").sum()),"threshold_source":fb.get("threshold_source","") ,"features":"|".join(C.M3_FEATURES)}); pc_rows.extend([{**r,"animal":subject,"policy":"T0"} for r in met["per_cycle"]]); pred["animal"]=subject; pred["policy"]="T0"; pred_rows.append(pred); audits.append({"animal":subject,"model":"M3","status":status,"classifier":"lr","threshold":fb.get("threshold",np.nan),"threshold_source":fb.get("threshold_source","") ,"features":"|".join(C.M3_FEATURES)})
        except Exception as exc:
            bundles["M3"]=_failed_bundle(exc); result_rows.append({"animal":subject,"model":"M3","status":"FAILED","failure_reason":str(exc)}); audits.append({"animal":subject,"model":"M3","status":"FAILED","failure_reason":str(exc)})
        try:
            print(f"[V5.1] {subject} M4 EUS-SP-LASSO", flush=True)
            lb=_fit_lasso(subject,split["train"],paths,tr); bundles["M4"]=lb; pred=_score_bundle(lb,te,"M4"); met=metrics(pred,split["test"],cycles,"M4"); result_rows.append({k:v for k,v in met.items() if k!="per_cycle"}|{"animal":subject,"model":"M4","status":"OK","train_cycles":"|".join(split["train"]),"test_cycles":"|".join(split["test"]),"train_nvc":int((tr.teacher_label=="NVC_CORE").sum()),"threshold_source":lb["threshold_source"],"alpha":lb["alpha"],"regression_target":lb["regression_target"],"features":"|".join(C.M4_FEATURES)}); pc_rows.extend([{**r,"animal":subject,"policy":"T0"} for r in met["per_cycle"]]); pred["animal"]=subject; pred["policy"]="T0"; pred_rows.append(pred); lasso_audit.append(lb["cv"].assign(animal=subject)); audits.append({"animal":subject,"model":"M4","status":"OK","classifier":"LASSO","threshold":lb["threshold"],"alpha":lb["alpha"],"regression_target":lb["regression_target"],"features":"|".join(C.M4_FEATURES),"n_continuous_samples":len(lb["continuous_samples"])})
        except Exception as exc:
            bundles["M4"]=_failed_bundle(exc); result_rows.append({"animal":subject,"model":"M4","status":"FAILED","failure_reason":str(exc)}); audits.append({"animal":subject,"model":"M4","status":"FAILED","failure_reason":str(exc)})
        bundles_by[subject]=bundles
        el,es=event_replay(subject,split["test"],bundles,events,paths); event_long.append(el); event_sum.append(es)
        fl,fs=full_cycle_replay(subject,split["test"],bundles,paths,events); full_long.append(fl); full_sum.append(fs)
        future.append(future_perturbation_audit(subject,bundles,paths,events))
        if len(es):
            for name,g in es.groupby("model"):
                rr=next((x for x in result_rows if x.get("animal")==subject and x.get("model")==name),None)
                if rr is not None:
                    rr["t0_event_sensitivity"]=float(g.t0_detected.mean()); rr["t1_sensitivity"]=float(g.t1_detected.mean()); rr["test_nvc_events"]=int(len(g))
    results=pd.DataFrame(result_rows); preds=pd.concat(pred_rows,ignore_index=True) if pred_rows else pd.DataFrame(); per=pd.DataFrame(pc_rows); audits=pd.DataFrame(audits); evlong=pd.concat(event_long,ignore_index=True) if event_long else pd.DataFrame(); evsum=pd.concat(event_sum,ignore_index=True) if event_sum else pd.DataFrame(); fulllong=pd.concat(full_long,ignore_index=True) if full_long else pd.DataFrame(); fullsum=pd.concat(full_sum,ignore_index=True) if full_sum else pd.DataFrame(); future=pd.concat(future,ignore_index=True) if future else pd.DataFrame(); la=pd.concat(lasso_audit,ignore_index=True) if lasso_audit else pd.DataFrame()
    results.to_csv(output_root/"v5_parallel_model_results.csv",index=False); preds.to_csv(output_root/"v5_parallel_predictions.csv",index=False); per.to_csv(output_root/"v5_parallel_per_cycle_metrics.csv",index=False); audits.to_csv(output_root/"v5_parallel_model_audit.csv",index=False); evlong.to_csv(output_root/"v5_parallel_event_replay.csv",index=False); evsum.to_csv(output_root/"v5_parallel_timing.csv",index=False); fulllong.to_csv(output_root/"v5_parallel_streaming_replay.csv",index=False); fullsum.to_csv(output_root/"v5_parallel_streaming_summary.csv",index=False); future.to_csv(output_root/"v5_parallel_future_perturbation.csv",index=False); la.to_csv(output_root/"v5_parallel_eus_sp_lasso_cv.csv",index=False)
    # Challenge scoring is deliberately post-freeze and does not feed back.
    chparts=[]; chsumm=[]
    for subject in C.SUBJECTS:
        split=C.SPLITS[subject]; ch=challenges[(challenges.subject==subject)&challenges.cycle_id.astype(str).isin(split["test"])]
        for name,b in bundles_by[subject].items():
            cp=_score_bundle(b,ch,name); cp["animal"]=subject; cp["replay_type"]="FROZEN_CHALLENGE"; chparts.append(cp)
            if len(cp):
                for typ,g in cp.groupby("challenge_type",dropna=False):
                    sc=g[g.score.notna()]; chsumm.append({"animal":subject,"model":name,"challenge_type":typ,"n":len(g),"scorable":len(sc),"triggered":int((sc.score>=sc.threshold).sum()),"trigger_fraction":float((sc.score>=sc.threshold).mean()) if len(sc) else np.nan})
    chp=pd.concat(chparts,ignore_index=True) if chparts else pd.DataFrame(); chs=pd.DataFrame(chsumm); chp.to_csv(output_root/"v5_parallel_challenge_predictions.csv",index=False); chs.to_csv(output_root/"v5_parallel_challenge_summary.csv",index=False)
    # E0/E1 are a within-M2 ablation and never counted as sixth formal model.
    ab=[]
    for subject in C.SUBJECTS:
        split=C.SPLITS[subject]; tr=train[(train.subject==subject)&train.cycle_id.astype(str).isin(split["train"])&train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)]; te=train[(train.subject==subject)&train.cycle_id.astype(str).isin(split["test"])&train.teacher_label.isin(C.PRIMARY_TRAIN_LABELS)]
        for nm,fschema in (("M2-E0",C.E0_FEATURES),("M2-E1",C.E_EARLY_FEATURES)):
            try:
                m,th,src,_,_=fit_individual(tr,fschema,"lr"); pp=apply_model(m,th,te,nm); mm=metrics(pp,split["test"],cycles,nm); ab.append({k:v for k,v in mm.items() if k!="per_cycle"}|{"animal":subject,"ablation":nm,"status":"OK","threshold_source":src})
            except Exception as exc: ab.append({"animal":subject,"ablation":nm,"status":"FAILED","failure_reason":str(exc)})
    ab=pd.DataFrame(ab); ab.to_csv(output_root/"v5_parallel_eus_ablation.csv",index=False)
    def _supported(frame, model):
        return bool(len(frame) and "status" in frame.columns and "nvc_detected" in frame.columns and ((frame.get("model", frame.get("ablation", "")) == model) & frame.status.eq("OK") & frame.nvc_detected.fillna(0).gt(0)).any())
    status={"INDIVIDUAL_PRESSURE_SUPPORTED":_supported(results,"R0") or _supported(results,"M1"),"EARLY_CAUSAL_PRESSURE_REDESIGN_SUPPORTED":_supported(results,"M1"),"TEMPORAL_CONFIRMATION_SUPPORTED":bool(evsum.t1_detected.any()) if len(evsum) and "t1_detected" in evsum.columns else False,"EUS_TIME_SUPPORTED":_supported(ab,"M2-E0"),"EUS_FAST_SPECTRAL_INCREMENT_SUPPORTED":False,"EUS_SP_LASSO_SUPPORTED":_supported(results,"M4"),"PE_INCREMENT_SUPPORTED":_supported(results,"M3")}
    if len(ab) and "ablation" in ab.columns and "nvc_detected" in ab.columns:
        e0 = ab[ab.ablation.eq("M2-E0")].set_index("animal").nvc_detected
        e1 = ab[ab.ablation.eq("M2-E1")].set_index("animal").nvc_detected
        status["EUS_FAST_SPECTRAL_INCREMENT_SUPPORTED"] = bool(len(e0) and len(e1) and (e1.reindex(e0.index).fillna(-1) > e0).any())
    summary={"version":"V5.1.0-parallel","primary_task":"NVC_CORE vs STABLE_FILLING","subjects":list(C.SUBJECTS),"splits":C.SPLITS,"models":list(C.PARALLEL_MODELS),"model_descriptions":C.PARALLEL_MODEL_DESCRIPTIONS,"results":results.to_dict(orient="records"),"statuses":status,"development_only":True,"deployment_ready":False,"stimulation":False,"challenge_only":True,"future_perturbation_all_unchanged":bool(future.unchanged.all()) if len(future) else False}
    (output_root/"v5_parallel_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    report=["# V5 parallel individualized NVC development","","Five independent models: R0, M1 P-EARLY, M2 E-EARLY, M3 PE-EARLY, M4 EUS-SP-LASSO.","","## Fixed prospective splits",pd.DataFrame([{"animal":s,"train":"|".join(C.SPLITS[s]["train"]),"test":"|".join(C.SPLITS[s]["test"])} for s in C.SUBJECTS]).to_markdown(index=False),"","## Results",results.to_markdown(index=False),"","## EUS E0/E1 ablation",ab.to_markdown(index=False),"","## Status",json.dumps(status,ensure_ascii=False,indent=2),"","development_only=true; deployment_ready=false; stimulation=false."]
    (output_root/"V5_PARALLEL_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    from .visualization import generate_plots
    generate_plots(output_root)
    return summary


if __name__ == "__main__":
    run_parallel()
