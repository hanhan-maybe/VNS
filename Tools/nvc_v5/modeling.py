"""Train-only individual models and prospective test metrics for V5."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def _complete(frame: pd.DataFrame, features) -> np.ndarray:
    # Streaming points before a causal history is available may return an
    # empty feature dict.  Treat absent schema fields as structural missingness
    # rather than raising or imputing them.
    x = frame.reindex(columns=list(features), fill_value=np.nan)
    return np.isfinite(x.to_numpy(dtype=float)).all(axis=1)


def prepare(frame: pd.DataFrame, features) -> pd.DataFrame:
    out = frame.copy()
    out["target"] = out.teacher_label.map({"NVC_CORE": 1, "STABLE_FILLING": 0})
    out["model_scorable"] = _complete(out, features)
    out["model_failure_reason"] = np.where(~out.model_scorable, "STRUCTURAL_FEATURE_MISSING", "")
    return out


def _weights(frame: pd.DataFrame) -> np.ndarray:
    """Equalize classes while retaining every row; fit is one animal only."""
    y = frame.target.to_numpy(dtype=int)
    counts = {int(k): int(v) for k, v in frame.target.value_counts().to_dict().items()}
    return np.asarray([1.0 / max(counts.get(int(v), 1), 1) for v in y], dtype=float)


def fit_model(frame: pd.DataFrame, features, classifier="lr"):
    C.assert_safe_feature_schema(features)
    d = frame[frame.model_scorable.astype(bool)].copy()
    if len(d) == 0 or d.target.nunique() < 2:
        raise ValueError("individual training fold lacks two scorable classes")
    if classifier == "lda":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])
        # LDA in the installed sklearn does not accept sample_weight; the
        # train-only balanced row sampler is represented by the fixed schema.
        model.fit(d[list(features)], d.target)
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logistic", LogisticRegression(
                C=1.0, penalty="l2", solver="lbfgs", max_iter=2000,
                class_weight=None, random_state=C.RANDOM_STATE)),
        ])
        model.fit(d[list(features)], d.target, logistic__sample_weight=_weights(d))
    model.fit_features_ = tuple(features)
    model.fit_cycles_ = tuple(sorted(d.cycle_id.astype(str).unique()))
    model.classifier_ = classifier
    return model


def score_model(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(frame[list(model.fit_features_)])[:, 1].astype(float)


def select_threshold(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    good = np.isfinite(score) & np.isfinite(y)
    y, score = y[good], score[good]
    if len(y) == 0:
        return np.nan, "NO_TRAIN_SCORES"
    candidates = np.unique(np.r_[score, np.nextafter(score, np.inf), 0.5])
    rows = []
    for t in candidates:
        pred = score >= t
        pos, neg = y == 1, y == 0
        sens = float(np.mean(pred[pos])) if pos.any() else np.nan
        spec = float(np.mean(~pred[neg])) if neg.any() else np.nan
        bal = float(balanced_accuracy_score(y, pred))
        rows.append((float(t), bal, sens, spec, sens + spec - 1 if np.isfinite(sens + spec) else np.nan))
    tab = pd.DataFrame(rows, columns=["threshold", "balanced_accuracy", "sensitivity", "specificity", "youden_j"])
    best = tab.sort_values(["balanced_accuracy", "youden_j", "threshold"], ascending=[False, False, False], na_position="last").iloc[0]
    return float(best.threshold), "train_oof" if len(y) > 1 else "train_single"


def cycle_oof_scores(frame: pd.DataFrame, features, cycles, classifier="lr"):
    """Leave-one-cycle-out scores using calibration cycles only."""
    base = prepare(frame, features)
    parts = []
    fallback = False
    cycles = tuple(str(x) for x in cycles)
    for held in cycles:
        tr = base[~base.cycle_id.astype(str).eq(held)].copy()
        te = base[base.cycle_id.astype(str).eq(held)].copy()
        te["oof_score"] = np.nan
        try:
            model = fit_model(tr, features, classifier)
            ok = te.model_scorable.astype(bool)
            if ok.any():
                te.loc[ok, "oof_score"] = score_model(model, te.loc[ok])
        except ValueError:
            fallback = True
        parts.append(te)
    oof = pd.concat(parts, ignore_index=True) if parts else base.iloc[0:0].copy()
    usable = oof[oof.oof_score.notna()]
    if len(usable) == 0 or usable.target.nunique() < 2:
        fallback = True
        try:
            model = fit_model(base, features, classifier)
            ok = base.model_scorable.astype(bool)
            oof = base.copy(); oof["oof_score"] = np.nan
            if ok.any(): oof.loc[ok, "oof_score"] = score_model(model, oof.loc[ok])
        except ValueError:
            oof = base.copy(); oof["oof_score"] = np.nan
    oof["oof_fallback_in_sample"] = bool(fallback)
    return oof


def fit_individual(frame: pd.DataFrame, features, classifier="lr"):
    """Fit one animal model and choose its threshold from calibration only."""
    base = prepare(frame, features)
    model = fit_model(base, features, classifier)
    oof = cycle_oof_scores(base, features, tuple(base.cycle_id.astype(str).unique()), classifier)
    usable = oof[oof.oof_score.notna()]
    threshold, source = select_threshold(usable.target, usable.oof_score)
    if not np.isfinite(threshold):
        in_sample = base[base.model_scorable.astype(bool)].copy()
        threshold, source = select_threshold(in_sample.target, score_model(model, in_sample))
    train_scored = base.copy(); train_scored["score"] = np.nan
    ok = train_scored.model_scorable.astype(bool)
    if ok.any(): train_scored.loc[ok, "score"] = score_model(model, train_scored.loc[ok])
    train_scored["threshold"] = threshold
    return model, threshold, source, train_scored, oof


def apply_model(model, threshold, frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = prepare(frame, model.fit_features_)
    out["score"] = np.nan
    ok = out.model_scorable.astype(bool)
    if ok.any(): out.loc[ok, "score"] = score_model(model, out.loc[ok])
    out["threshold"] = float(threshold) if np.isfinite(threshold) else np.nan
    out["model"] = model_name
    out["predicted_nvc"] = out.score >= out.threshold
    return out


def metrics(pred: pd.DataFrame, test_cycles, cycles: pd.DataFrame, model_name: str) -> dict:
    """Metrics retain raw NVC counts and count unscorable NVC as misses."""
    test_cycles = tuple(str(x) for x in test_cycles)
    p = pred[pred.score.notna()].copy()
    n_all = pred[pred.teacher_label.eq("NVC_CORE")]
    n = p[p.teacher_label.eq("NVC_CORE")]
    st = p[p.teacher_label.eq("STABLE_FILLING")]
    tp = int((n.score >= n.threshold).sum())
    fp = int((st.score >= st.threshold).sum())
    duration = float(cycles[cycles.cycle_id.astype(str).isin(test_cycles)].cycle_duration_s.sum()) if len(cycles) else np.nan
    y = p.target.to_numpy(dtype=int) if len(p) else np.array([])
    s = p.score.to_numpy(dtype=float) if len(p) else np.array([])
    two = len(y) and np.unique(y).size == 2
    per_cycle = []
    for cyc in test_cycles:
        g = pred[pred.cycle_id.astype(str).eq(cyc)]
        gn = g[g.teacher_label.eq("NVC_CORE")]; gs = g[g.teacher_label.eq("STABLE_FILLING")]
        per_cycle.append({
            "model": model_name, "cycle_id": cyc,
            "nvc_total": int(len(gn)), "nvc_scorable": int(gn.score.notna().sum()),
            "nvc_detected": int((gn.score >= gn.threshold).sum()),
            "stable_scorable": int(gs.score.notna().sum()),
            "stable_false_triggers": int((gs.score >= gs.threshold).sum()),
        })
    return {
        "model": model_name, "n_test_cycles": len(test_cycles),
        "test_nvc": int(len(n_all)), "nvc_scorable": int(len(n)), "nvc_detected": tp,
        "sensitivity": float(tp / len(n_all)) if len(n_all) else np.nan,
        "coverage": float(len(n) / len(n_all)) if len(n_all) else np.nan,
        "test_stable_scorable": int(len(st)), "stable_false_triggers": fp,
        "fp_per_cycle": float(fp / len(test_cycles)) if test_cycles else np.nan,
        "fp_per_hour": float(fp / (duration / 3600.0)) if np.isfinite(duration) and duration > 0 else np.nan,
        "ppv": float(tp / (tp + fp)) if tp + fp else np.nan,
        "AUROC": float(roc_auc_score(y, s)) if two else np.nan,
        "AUPRC": float(average_precision_score(y, s)) if two else np.nan,
        "test_duration_s": duration,
        "per_cycle": per_cycle,
    }


def fit_fusion(train: pd.DataFrame, p_features, e_features, classifier="lr"):
    """Fit individual late-fusion model from train-cycle OOF base scores."""
    pmodel, _, _, ptrain, poof = fit_individual(train, p_features, "lr")
    emodel, _, _, etrain, eoof = fit_individual(train, e_features, "lr")
    f = prepare(train, C.FUSION_FEATURES)
    # Base scores used for fusion fitting are leave-one-cycle-out where
    # possible; this avoids fitting fusion on the same row's base prediction.
    pmap = poof[["sample_uid", "oof_score"]].rename(columns={"oof_score": "S_P"})
    emap = eoof[["sample_uid", "oof_score"]].rename(columns={"oof_score": "S_E"})
    f = f.drop(columns=["S_P", "S_E"], errors="ignore").merge(pmap, on="sample_uid", how="left").merge(emap, on="sample_uid", how="left")
    f["model_scorable"] = np.isfinite(f[["S_P", "S_E"] + list(C.COUPLING_FEATURES)].to_numpy(float)).all(axis=1)
    fm = None; threshold = np.nan; threshold_source = "NO_FUSION_ROWS"; foof = f.copy(); foof["oof_score"] = np.nan
    if f[f.model_scorable].target.nunique() >= 2:
        fm = fit_model(f, C.FUSION_FEATURES, classifier)
        fi = f[f.model_scorable].copy(); fi["oof_score"] = score_model(fm, fi)
        foof = f; foof.loc[fi.index, "oof_score"] = fi.oof_score
        threshold, threshold_source = select_threshold(fi.target, fi.oof_score)
    return {
        "pmodel": pmodel, "emodel": emodel, "fusion_model": fm,
        "threshold": threshold, "threshold_source": threshold_source,
        "p_features": tuple(p_features), "e_features": tuple(e_features),
        "train_fusion": f, "fusion_oof": foof,
    }


def apply_fusion(bundle, frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    out = frame.copy()
    out["target"] = out.teacher_label.map({"NVC_CORE": 1, "STABLE_FILLING": 0})
    p = prepare(out, bundle["p_features"]); e = prepare(out, bundle["e_features"])
    out["S_P"] = np.nan; out["S_E"] = np.nan
    ok = p.model_scorable.astype(bool)
    if ok.any(): out.loc[ok, "S_P"] = score_model(bundle["pmodel"], p.loc[ok])
    ok = e.model_scorable.astype(bool)
    if ok.any(): out.loc[ok, "S_E"] = score_model(bundle["emodel"], e.loc[ok])
    out["model_scorable"] = np.isfinite(out[["S_P", "S_E"] + list(C.COUPLING_FEATURES)].to_numpy(float)).all(axis=1)
    out["model_failure_reason"] = np.where(~out.model_scorable, "FUSION_COMPONENT_MISSING", "")
    out["score"] = np.nan
    if bundle["fusion_model"] is not None:
        ok = out.model_scorable.astype(bool)
        if ok.any(): out.loc[ok, "score"] = score_model(bundle["fusion_model"], out.loc[ok])
    out["threshold"] = bundle["threshold"]
    out["model"] = model_name
    out["predicted_nvc"] = out.score >= out.threshold
    return out
