"""Animal-level validation and metrics for V3.1."""
from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .version_support import expanded_animal_class_weights
from .config import LABEL_TO_TARGET, MODEL_FEATURES, RANDOM_STATE, SUBJECTS, TARGET_LABELS


def prepare_frame(features: pd.DataFrame, model_name: str, delay: float,
                  subjects: Iterable[str] | None = None) -> pd.DataFrame:
    columns = list(MODEL_FEATURES[model_name])
    out = features[(features["teacher_label"].isin(TARGET_LABELS))
                   & np.isclose(features["decision_delay_s"], delay)].copy()
    if subjects is not None:
        out = out[out["subject"].isin(tuple(subjects))].copy()
    out["target"] = out["teacher_label"].map(LABEL_TO_TARGET).astype(int)
    pressure_features = list(MODEL_FEATURES["C0"] if model_name == "C0" else MODEL_FEATURES["P"])
    complete = np.isfinite(out[pressure_features].to_numpy(dtype=float)).all(axis=1)
    out["model_scorable"] = out["base_eligible"].astype(bool) & complete
    if model_name in {"PE_SPECTRAL_COMMON", "PEF"}:
        out["model_scorable"] &= out["spectral_scorable"].astype(bool)
    out["model_failure_reason"] = np.where(
        ~out["base_eligible"].astype(bool), out["base_failure_reason"],
        np.where(~complete, "MODEL_FEATURE_UNSCORABLE", ""))
    return out


def fit_pipeline(frame: pd.DataFrame, model_name: str,
                 allowed_subjects: Sequence[str]) -> Pipeline:
    scored = frame[frame["model_scorable"].astype(bool)].copy()
    allowed = tuple(sorted(map(str, allowed_subjects)))
    if set(scored["subject"].astype(str)) - set(allowed):
        raise AssertionError("Held-out animal entered model fit")
    if scored["target"].nunique() != 2:
        raise ValueError("Training fold does not contain both target classes")
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                        max_iter=2000, random_state=RANDOM_STATE)),
    ])
    columns = list(MODEL_FEATURES[model_name])
    model.fit(scored[columns], scored["target"],
              logistic__sample_weight=expanded_animal_class_weights(scored))
    model.fit_subjects_ = allowed
    model.fit_features_ = tuple(columns)
    model.positive_class_ = 1
    model.positive_class_index_ = int(np.flatnonzero(model.named_steps["logistic"].classes_ == 1)[0])
    return model


def transformed_feature_names(model: Pipeline) -> list[str]:
    """Return imputer output names on both old and current scikit-learn."""
    imputer = model.named_steps["imputer"]
    if hasattr(imputer, "get_feature_names_out"):
        return list(imputer.get_feature_names_out(list(model.fit_features_)))
    names = list(model.fit_features_)
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        names.extend(f"missingindicator_{model.fit_features_[int(i)]}"
                     for i in indicator.features_)
    return names


def predict_frame(frame: pd.DataFrame, model: Pipeline, threshold: float) -> pd.DataFrame:
    out = frame.copy(); out["p_nvc"] = np.nan
    scorable = out["model_scorable"].astype(bool)
    if scorable.any():
        out.loc[scorable, "p_nvc"] = model.predict_proba(
            out.loc[scorable, list(model.fit_features_)])[:, model.positive_class_index_]
    out["threshold_train_only"] = float(threshold)
    out["predicted_nvc"] = out["p_nvc"].ge(threshold) & out["p_nvc"].notna()
    out["predicted_label"] = np.where(
        out["p_nvc"].isna(), "UNSCORABLE",
        np.where(out["predicted_nvc"], "NVC_CORE", "PREVOID_PROGRESSIVE"))
    out["actionable_hit"] = (out["predicted_nvc"] & out["teacher_label"].eq("NVC_CORE")
                              & out["actionable"].astype(bool))
    return out


def select_threshold(y: np.ndarray, probabilities: np.ndarray,
                     weights: np.ndarray) -> tuple[float, float]:
    candidates = sorted(set(np.r_[0.0, probabilities, np.nextafter(probabilities, np.inf), 1.0000001]))
    best = None
    for threshold in candidates:
        pred = probabilities >= threshold
        score = float(balanced_accuracy_score(y, pred, sample_weight=weights))
        negative = y == 0
        specificity = float(np.average(~pred[negative], weights=weights[negative])) if negative.any() else 0.0
        key = (score, specificity, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1], best[0][0]


def per_animal_metrics(predictions: pd.DataFrame, model_name: str | None = None) -> pd.DataFrame:
    rows = []
    for subject, frame in predictions.groupby("subject", sort=True):
        scored = frame[frame["p_nvc"].notna()].copy()
        nvc_all = frame[frame["teacher_label"] == "NVC_CORE"]
        nvc_scored = scored[scored["teacher_label"] == "NVC_CORE"]
        prevoid = scored[scored["teacher_label"] == "PREVOID_PROGRESSIVE"]
        tp = int(nvc_scored["predicted_nvc"].sum())
        fp = int(prevoid["predicted_nvc"].sum())
        fn = int(len(nvc_scored) - tp); tn = int(len(prevoid) - fp)
        two_classes = scored["target"].nunique() == 2
        median_nvc = float(nvc_scored["p_nvc"].median()) if len(nvc_scored) else np.nan
        median_prevoid = float(prevoid["p_nvc"].median()) if len(prevoid) else np.nan
        rows.append({
            "model": model_name or str(frame.get("model", pd.Series([""])).iloc[0]),
            "animal": str(subject), "dataset": str(frame["dataset"].iloc[0]),
            "n_frozen_nvc": int(len(nvc_all)), "n_scorable_nvc": int(len(nvc_scored)),
            "coverage": float(len(nvc_scored) / len(nvc_all)) if len(nvc_all) else np.nan,
            "n_prevoid": int(len(prevoid)), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "frozen_sensitivity": float(tp / len(nvc_all)) if len(nvc_all) else np.nan,
            "scorable_sensitivity": float(tp / len(nvc_scored)) if len(nvc_scored) else np.nan,
            "PPV": float(tp / (tp + fp)) if tp + fp else np.nan,
            "specificity": float(tn / len(prevoid)) if len(prevoid) else np.nan,
            "AUROC": float(roc_auc_score(scored["target"], scored["p_nvc"])) if two_classes else np.nan,
            "AUPRC": float(average_precision_score(scored["target"], scored["p_nvc"])) if two_classes else np.nan,
            "median_score_nvc": median_nvc, "median_score_prevoid": median_prevoid,
            "score_separation": median_nvc - median_prevoid if np.isfinite(median_nvc) and np.isfinite(median_prevoid) else np.nan,
            "actionable_hits": int(nvc_scored["actionable_hit"].sum()),
            "actionable_nvc": int(nvc_all["actionable"].astype(bool).sum()),
        })
    return pd.DataFrame(rows)


def aggregate_metrics(predictions: pd.DataFrame, model_name: str) -> dict:
    animal = per_animal_metrics(predictions, model_name)
    scored = predictions[predictions["p_nvc"].notna()].copy()
    nvc = predictions[predictions["teacher_label"] == "NVC_CORE"]
    prevoid = scored[scored["teacher_label"] == "PREVOID_PROGRESSIVE"]
    tp = int((scored["predicted_nvc"] & scored["teacher_label"].eq("NVC_CORE")).sum())
    fp = int(prevoid["predicted_nvc"].sum())
    positive_animals = animal[animal["n_frozen_nvc"] > 0]
    sensitivities = positive_animals["frozen_sensitivity"].dropna()
    two_classes = scored["target"].nunique() == 2
    return {
        "model": model_name,
        "animal_macro_frozen_sensitivity": float(sensitivities.mean()) if len(sensitivities) else np.nan,
        "pooled_frozen_sensitivity": float(tp / len(nvc)) if len(nvc) else np.nan,
        "worst_animal_sensitivity": float(sensitivities.min()) if len(sensitivities) else np.nan,
        "zero_hit_animals": int((positive_animals["TP"] == 0).sum()),
        "PREVOID_FP": fp, "PREVOID_FPR": float(fp / len(prevoid)) if len(prevoid) else np.nan,
        "PPV": float(tp / (tp + fp)) if tp + fp else np.nan,
        "AUROC": float(roc_auc_score(scored["target"], scored["p_nvc"])) if two_classes else np.nan,
        "AUPRC": float(average_precision_score(scored["target"], scored["p_nvc"])) if two_classes else np.nan,
        "actionable_sensitivity": float(scored["actionable_hit"].sum() / len(nvc)) if len(nvc) else np.nan,
        "scorable_coverage": float(scored[scored["teacher_label"] == "NVC_CORE"].shape[0] / len(nvc)) if len(nvc) else np.nan,
        "TP": tp, "frozen_NVC": int(len(nvc)), "scorable_NVC": int((scored["target"] == 1).sum()),
        "scorable_PREVOID": int((scored["target"] == 0).sum()),
    }


def _training_threshold(frame: pd.DataFrame, model: Pipeline) -> tuple[float, float]:
    scored = frame[frame["model_scorable"].astype(bool)].copy()
    p = model.predict_proba(scored[list(model.fit_features_)])[:, model.positive_class_index_]
    return select_threshold(scored["target"].to_numpy(int), p, expanded_animal_class_weights(scored))


def run_outer_loso(features: pd.DataFrame, model_name: str, delay: float,
                   subjects: Sequence[str] = SUBJECTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = prepare_frame(features, model_name, delay, subjects)
    predictions, audit, coefficients = [], [], []
    for held in subjects:
        train_subjects = [s for s in subjects if s != held]
        train = frame[frame["subject"].isin(train_subjects)]
        test = frame[frame["subject"] == held]
        model = fit_pipeline(train, model_name, train_subjects)
        threshold, ba = _training_threshold(train, model)
        pred = predict_frame(test, model, threshold)
        pred["model"] = model_name; pred["outer_held_out_animal"] = held
        pred["training_animals"] = "+".join(train_subjects)
        pred["model_classes"] = "+".join(map(str, model.named_steps["logistic"].classes_))
        pred["positive_class"] = model.positive_class_
        pred["positive_class_index"] = model.positive_class_index_
        predictions.append(pred)
        classes = model.named_steps["logistic"].classes_
        audit.append({
            "candidate_model": model_name, "outer_held_out_animal": held,
            "outer_training_animals": "+".join(train_subjects), "inner_selected_delay": delay,
            "inner_selected_threshold": threshold, "training_balanced_accuracy": ba,
            "model_classes": "+".join(map(str, classes)), "positive_class": 1,
            "positive_class_index": model.positive_class_index_,
            "scaler_fit_animals": "+".join(model.fit_subjects_),
            "scaler_mean": json.dumps(model.named_steps["scaler"].mean_.tolist()),
            "scaler_scale": json.dumps(model.named_steps["scaler"].scale_.tolist()),
            "threshold_fit_animals": "+".join(train_subjects), "leakage": False,
        })
        names = transformed_feature_names(model)
        for name, coef in zip(names, model.named_steps["logistic"].coef_[0]):
            coefficients.append({"model": model_name, "outer_fold": held, "feature": name,
                                 "coefficient": float(coef), "sign": int(np.sign(coef)),
                                 "abs_coefficient": abs(float(coef))})
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit), pd.DataFrame(coefficients)


def inner_oof(features: pd.DataFrame, model_name: str, delay: float,
              subjects: Sequence[str]) -> pd.DataFrame:
    frame = prepare_frame(features, model_name, delay, subjects)
    parts = []
    for held in subjects:
        fit_subjects = [s for s in subjects if s != held]
        model = fit_pipeline(frame[frame["subject"].isin(fit_subjects)], model_name, fit_subjects)
        test = frame[frame["subject"] == held].copy(); test["p_nvc"] = np.nan
        ok = test["model_scorable"].astype(bool)
        if ok.any():
            test.loc[ok, "p_nvc"] = model.predict_proba(
                test.loc[ok, list(model.fit_features_)])[:, model.positive_class_index_]
        test["inner_held_animal"] = held; test["inner_fit_animals"] = "+".join(fit_subjects)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def threshold_inner_oof(oof: pd.DataFrame) -> tuple[float, float]:
    scored = oof[oof["p_nvc"].notna()].copy()
    return select_threshold(scored["target"].to_numpy(int), scored["p_nvc"].to_numpy(float),
                            expanded_animal_class_weights(scored))


def score_oof(oof: pd.DataFrame, threshold: float, model_name: str) -> tuple[dict, pd.DataFrame]:
    scored = oof.copy(); scored["predicted_nvc"] = scored["p_nvc"].ge(threshold) & scored["p_nvc"].notna()
    scored["actionable_hit"] = (scored["predicted_nvc"] & scored["teacher_label"].eq("NVC_CORE")
                                 & scored["actionable"].astype(bool))
    return aggregate_metrics(scored, model_name), scored


def select_delay_inner(features: pd.DataFrame, subjects: Sequence[str]) -> tuple[float, float, pd.DataFrame]:
    rows = []
    for delay in sorted(features["decision_delay_s"].unique()):
        oof = inner_oof(features, "PE", float(delay), subjects)
        threshold, _ = threshold_inner_oof(oof)
        metrics, _ = score_oof(oof, threshold, "PE_DELAY")
        animal = per_animal_metrics(score_oof(oof, threshold, "PE_DELAY")[1], "PE_DELAY")
        macro_auc = float(animal[animal["n_frozen_nvc"] > 0]["AUROC"].dropna().mean())
        rows.append({"delay": float(delay), "threshold": threshold, "animal_macro_AUROC": macro_auc,
                     **metrics})
    table = pd.DataFrame(rows)
    best = table.sort_values(
        ["animal_macro_AUROC", "animal_macro_frozen_sensitivity", "PREVOID_FP",
         "actionable_sensitivity", "delay"],
        ascending=[False, False, True, False, True], na_position="last").iloc[0]
    table["selected"] = np.isclose(table["delay"], float(best["delay"]))
    return float(best["delay"]), float(best["threshold"]), table


def _pareto_threshold(oof: pd.DataFrame, reference_fpr: float) -> tuple[float, pd.DataFrame]:
    scored = oof[oof["p_nvc"].notna()].copy()
    candidates = sorted(set(np.r_[0.0, scored["p_nvc"], np.nextafter(scored["p_nvc"], np.inf), 1.0000001]))
    rows = []
    for threshold in candidates:
        metrics, _ = score_oof(oof, float(threshold), "CANDIDATE+VOIDGUARD")
        rows.append({"threshold": float(threshold), **metrics})
    table = pd.DataFrame(rows)
    feasible = table[table["PREVOID_FPR"] <= reference_fpr + 1e-12]
    if feasible.empty:
        feasible = table
    best = feasible.sort_values(
        ["animal_macro_frozen_sensitivity", "worst_animal_sensitivity", "PPV", "threshold"],
        ascending=[False, False, False, False], na_position="last").iloc[0]
    return float(best["threshold"]), table


def run_nested_candidates(features: pd.DataFrame,
                          subjects: Sequence[str] = SUBJECTS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts, audit_rows, delay_rows, pareto_rows = [], [], [], []
    for held in subjects:
        outer_train = [s for s in subjects if s != held]
        selected_delay, selected_a_threshold, delay_table = select_delay_inner(features, outer_train)
        delay_table["outer_held_out_animal"] = held; delay_rows.append(delay_table)

        for candidate, feature_model in (("PE_DELAY", "PE_DELAY"), ("PE_TRAJECTORY", "PE_TRAJECTORY")):
            oof = inner_oof(features, feature_model, selected_delay, outer_train)
            threshold, _ = threshold_inner_oof(oof)
            frame = prepare_frame(features, feature_model, selected_delay, subjects)
            model = fit_pipeline(frame[frame["subject"].isin(outer_train)], feature_model, outer_train)
            pred = predict_frame(frame[frame["subject"] == held], model, threshold)
            pred["model"] = candidate; pred["outer_held_out_animal"] = held
            pred["training_animals"] = "+".join(outer_train); prediction_parts.append(pred)
            pred["model_classes"] = "+".join(map(str, model.named_steps["logistic"].classes_))
            pred["positive_class"] = model.positive_class_
            pred["positive_class_index"] = model.positive_class_index_
            audit_rows.append({
                "candidate_model": candidate, "outer_held_out_animal": held,
                "outer_training_animals": "+".join(outer_train), "inner_selected_delay": selected_delay,
                "inner_selected_threshold": threshold, "scaler_fit_animals": "+".join(model.fit_subjects_),
                "scaler_mean": json.dumps(model.named_steps["scaler"].mean_.tolist()),
                "scaler_scale": json.dumps(model.named_steps["scaler"].scale_.tolist()),
                "threshold_fit_animals": "+".join(outer_train), "inner_delay_fit_animals": "+".join(outer_train),
                "leakage": False,
            })

        # Candidate C uses the same causal trajectory score as a non-progressive guard.
        trajectory_oof = inner_oof(features, "CANDIDATE+VOIDGUARD", selected_delay, outer_train)
        pe_oof = inner_oof(features, "PE", selected_delay, outer_train)
        pe_threshold, _ = threshold_inner_oof(pe_oof)
        pe_metrics, _ = score_oof(pe_oof, pe_threshold, "PE")
        guard_threshold, pareto = _pareto_threshold(trajectory_oof, pe_metrics["PREVOID_FPR"])
        pareto["outer_held_out_animal"] = held; pareto["decision_delay"] = selected_delay
        pareto_rows.append(pareto)
        frame = prepare_frame(features, "CANDIDATE+VOIDGUARD", selected_delay, subjects)
        model = fit_pipeline(frame[frame["subject"].isin(outer_train)], "CANDIDATE+VOIDGUARD", outer_train)
        pred = predict_frame(frame[frame["subject"] == held], model, guard_threshold)
        pred["model"] = "CANDIDATE+VOIDGUARD"; pred["outer_held_out_animal"] = held
        pred["training_animals"] = "+".join(outer_train); pred["stage_a_candidate_detected"] = True
        pred["model_classes"] = "+".join(map(str, model.named_steps["logistic"].classes_))
        pred["positive_class"] = model.positive_class_
        pred["positive_class_index"] = model.positive_class_index_
        pred["p_void_risk"] = 1.0 - pred["p_nvc"]
        prediction_parts.append(pred)
        audit_rows.append({
            "candidate_model": "CANDIDATE+VOIDGUARD", "outer_held_out_animal": held,
            "outer_training_animals": "+".join(outer_train), "inner_selected_delay": selected_delay,
            "inner_selected_threshold": guard_threshold, "scaler_fit_animals": "+".join(model.fit_subjects_),
            "scaler_mean": json.dumps(model.named_steps["scaler"].mean_.tolist()),
            "scaler_scale": json.dumps(model.named_steps["scaler"].scale_.tolist()),
            "threshold_fit_animals": "+".join(outer_train), "inner_delay_fit_animals": "+".join(outer_train),
            "leakage": False,
        })
    return (pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(audit_rows),
            pd.concat(delay_rows, ignore_index=True), pd.concat(pareto_rows, ignore_index=True))
