"""VOID_RISK models with animal-balanced causal decision-time validation."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

RISK_LABELS = {"NVC_CORE": 0, "PREVOID_PROGRESSIVE": 1, "VOID_CONFIRMED": 1}
DANGEROUS = {"PREVOID_PROGRESSIVE", "VOID_CONFIRMED"}
EUS_CANDIDATES = ["eus_tonic_occupancy", "eus_envelope_slope"]


def replay_trigger_mask(frame: pd.DataFrame, probability_col: str, threshold: float) -> np.ndarray:
    """Compatibility helper retained for the core VoidGuard tests."""
    trigger = np.zeros(len(frame), dtype=bool)
    lockout_until = -np.inf
    for pos, (_, row) in enumerate(frame.iterrows()):
        t = float(row.get("confirm_time_s", 0.0))
        if not bool(row.get("data_valid", True)) or t < lockout_until: continue
        if float(row.get(probability_col, 0.0)) >= threshold:
            trigger[pos] = True; lockout_until = t + C.POST_EVENT_LOCKOUT_S
    return trigger


def feature_columns(model_name: str, eus_features: Optional[Iterable[str]] = None) -> List[str]:
    cols = list(C.PRESSURE_FEATURES)
    if model_name == "M2": cols += list(eus_features if eus_features is not None else EUS_CANDIDATES)
    forbidden = {"subject", "cycle_id", "event_id", "teacher_label", "Volume", "urine", "recovery", "time_to_urine", "label"}
    assert not any(any(token.lower() in c.lower() for token in forbidden) for c in cols)
    return cols


def model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame[frame["teacher_label"].isin(RISK_LABELS)].copy()
    out["y_void_risk"] = out["teacher_label"].map(RISK_LABELS).astype(int)
    return out


def select_eus_features_train(frame: pd.DataFrame) -> Tuple[List[str], Dict]:
    """Direction rule uses only the supplied training animals."""
    train = model_frame(frame); rows = []; keep = []
    for feature in EUS_CANDIDATES:
        diffs = []
        for subject, group in train.groupby("subject"):
            nvc, risk = group[group.y_void_risk == 0][feature], group[group.y_void_risk == 1][feature]
            if nvc.notna().any() and risk.notna().any(): diffs.append(float(risk.median() - nvc.median()))
        direction = "POSITIVE" if diffs and all(x > 0 for x in diffs) else "NEGATIVE" if diffs and all(x < 0 for x in diffs) else "CROSS_SUBJECT_DIRECTION_UNSTABLE"
        rows.append({"feature": feature, "train_subjects": "+".join(sorted(train.subject.unique())), "diffs": ";".join(f"{x:.6g}" for x in diffs), "direction": direction})
        if direction != "CROSS_SUBJECT_DIRECTION_UNSTABLE": keep.append(feature)
    return keep, {"rows": rows}


def balanced_subject_class_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby(["subject", "y_void_risk"]).size().to_dict()
    weights = np.asarray([0.5 / counts[(s, y)] for s, y in zip(frame.subject, frame.y_void_risk)], dtype=float)
    return weights


def fit_logistic(frame: pd.DataFrame, model_name: str, allowed_subjects: Iterable[str], eus_features=None) -> Pipeline:
    allowed = set(allowed_subjects); train = model_frame(frame)
    if set(train.subject) - allowed: raise AssertionError("held-out animal entered model fit")
    cols = feature_columns(model_name, eus_features)
    if train.y_void_risk.nunique() < 2 or not (train.y_void_risk == 0).any() or not (train.y_void_risk == 1).any():
        raise ValueError("HOLD_INSUFFICIENT_LABELS")
    pipe = Pipeline([("scaler", StandardScaler()), ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", random_state=C.RANDOM_SEED, max_iter=2000))])
    pipe.fit(train[cols], train.y_void_risk, logistic__sample_weight=balanced_subject_class_weights(train))
    pipe.fit_subjects_ = tuple(sorted(allowed)); pipe.fit_features_ = tuple(cols)
    return pipe


def cross_predictions(frame: pd.DataFrame, subjects: List[str], model_name: str) -> pd.DataFrame:
    parts = []
    for held in subjects:
        train_subjects = [s for s in subjects if s != held]; train = frame[frame.subject.isin(train_subjects)]; test = model_frame(frame[frame.subject == held]).copy()
        eus_features, _ = select_eus_features_train(train) if model_name == "M2" else ([], {})
        model = fit_logistic(train, model_name, train_subjects, eus_features)
        test["p_void_risk"] = model.predict_proba(test[list(model.fit_features_)])[:, 1]; test["inner_held_subject"] = held
        test["eus_features_used"] = ",".join(eus_features); parts.append(test)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _accepted(oof, threshold):
    return (oof.p_void_risk < threshold) & oof.decision_eligible.astype(bool)


def select_safety_threshold(oof: pd.DataFrame) -> Tuple[float, Dict]:
    if oof.empty: return 0.0, {"admission": False, "reason": "NO_OOF", "ABSTAIN_ALL": True}
    candidates = sorted(set(np.r_[0.0, oof.p_void_risk.to_numpy(), np.nextafter(oof.p_void_risk.to_numpy(), np.inf), 1.0000001]))
    best = None
    for threshold in candidates:
        accepted = _accepted(oof, float(threshold)); dangerous = int((accepted & oof.teacher_label.isin(DANGEROUS)).sum()); nvc = int((accepted & (oof.teacher_label == "NVC_CORE")).sum())
        if dangerous: continue
        key = (nvc, -float(threshold))
        if best is None or key > best[0]: best = (key, float(threshold), nvc)
    if best is None or best[2] == 0: return 0.0, {"admission": False, "reason": "ABSTAIN_ALL", "ABSTAIN_ALL": True}
    return best[1], {"admission": True, "reason": "SAFE_NVC_ACCEPTANCE", "ABSTAIN_ALL": False, "nvc_accepts": best[2]}


def select_analysis_threshold(oof: pd.DataFrame) -> Tuple[float, Dict]:
    if oof.empty: return 0.0, {"macro_f1": None, "ABSTAIN_ALL": True}
    best = None
    for threshold in sorted(set(np.r_[0.0, oof.p_void_risk.to_numpy(), np.nextafter(oof.p_void_risk.to_numpy(), np.inf), 1.0000001])):
        fs = []
        for _, g in oof.groupby("subject"):
            eligible = g.decision_eligible.astype(bool); y = g.teacher_label.eq("NVC_CORE") & eligible; pred = (g.p_void_risk < threshold) & eligible
            tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum()); den = 2 * tp + fp + fn
            fs.append(2 * tp / den if den else 0.0)
        key = (float(np.mean(fs)), -float(threshold))
        if best is None or key > best[0]: best = (key, float(threshold))
    return best[1], {"macro_f1": best[0][0], "ABSTAIN_ALL": best[1] == 0.0}


def serialize_model(pipe: Pipeline, model_name: str, analysis_threshold: float, safety_threshold: float, delay_s: float) -> Dict:
    scaler, lr = pipe.named_steps["scaler"], pipe.named_steps["logistic"]
    return {"model": model_name, "feature_order": list(pipe.fit_features_), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
            "coefficient_void_risk": lr.coef_[0].tolist(), "intercept_void_risk": float(lr.intercept_[0]), "C": 1.0,
            "decision_delay_s": delay_s, "analysis_threshold": analysis_threshold, "safety_threshold": safety_threshold}
