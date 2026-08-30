"""V3 NVC recognition development using the unchanged V2.1 feature space."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Tools.dsd_feature_extraction import config as C
from Tools.dsd_feature_extraction.data_io import load_cycle, write_json
from Tools.dsd_feature_extraction.detectors import AdaptiveHistory, adaptive_local_pressure_events
from Tools.dsd_feature_extraction.features import causal_eus_envelope_100hz, decision_feature_at_index
from .version_support import (
    DECISION_DELAY_S,
    EUS_FEATURES,
    LABEL_TO_TARGET,
    MODEL_FEATURES,
    P_FEATURES,
    RANDOM_STATE,
    SPECTRAL_FEATURES,
    TARGET_LABELS,
    _as_bool,
    _eus_features,
    _low_frequency_feature,
    _metric_values,
    _pressure_features,
    assert_feature_schema_safe,
)


SUBJECTS_338 = ("STxF26", "STxF27", "STxF29")
SUBJECTS_164 = ("STxF31", "STxF33", "STxF34", "STxF35", "STxF37")
SUBJECTS = SUBJECTS_338 + SUBJECTS_164
DATASET_BY_SUBJECT = {**{s: "338" for s in SUBJECTS_338},
                      **{s: "164" for s in SUBJECTS_164}}
C0_FEATURES = tuple(C.PRESSURE_FEATURES)
ALL_MODEL_FEATURES = {
    "C0": C0_FEATURES,
    "P": MODEL_FEATURES["P"],
    "PE": MODEL_FEATURES["PE"],
    "PE_SPECTRAL_COMMON": MODEL_FEATURES["PE"],
    "PEF": MODEL_FEATURES["PEF"],
}
MODEL_ORDER = tuple(ALL_MODEL_FEATURES)


def assert_v3_paths(paths: Iterable[Path]) -> None:
    """Reject accidental use of cohorts outside the intended 338 + 164 roots."""
    forbidden = ("stxf21", "stxf30", "dataset165", "dataset166")
    offenders = [str(Path(path).resolve()) for path in paths
                 if any(token in str(Path(path).resolve()).casefold() for token in forbidden)]
    if offenders:
        raise RuntimeError(f"V3 path is outside the registered expanded cohort: {offenders}")


def expanded_animal_class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give each animal equal total weight and balance classes when both exist."""
    counts = frame.groupby(["subject", "target"]).size().to_dict()
    subjects = sorted(frame["subject"].astype(str).unique())
    if not subjects:
        raise ValueError("No training animals")
    class_sets = {
        subject: sorted(target for (animal, target), count in counts.items()
                        if animal == subject and count > 0)
        for subject in subjects
    }
    if any(not targets for targets in class_sets.values()):
        raise ValueError("A training animal has no scorable events")
    return np.asarray([
        1.0 / (len(subjects) * len(class_sets[subject]) * counts[(subject, target)])
        for subject, target in zip(frame["subject"].astype(str), frame["target"])
    ], dtype=np.float64)


def _stage_a_population_priors(reference_root: Path) -> tuple[float, float]:
    params = pd.read_csv(reference_root / "subject_adaptive_params.csv")
    sigma = pd.to_numeric(params["warmup_prior_sigma_p"], errors="coerce").dropna()
    dpdt = pd.to_numeric(params["sigma_dpdt_median"], errors="coerce").dropna()
    if sigma.empty or dpdt.empty:
        raise RuntimeError("Dataset338 Stage-A population priors are unavailable")
    return float(sigma.median()), float(dpdt.median())


def _empty_feature_row(subject: str, cycle_id: str, frozen: pd.Series,
                       confirm_time: float, decision_time: float,
                       feature_time: float, index: int, eligible: bool,
                       reason: str) -> dict:
    return {
        "dataset": DATASET_BY_SUBJECT[subject],
        "subject": subject,
        "cycle_id": cycle_id,
        "event_id": str(frozen["event_id"]),
        "event_uid": str(frozen["event_uid"]),
        "teacher_label": str(frozen["teacher_label"]),
        "confirm_time_s": confirm_time,
        "decision_delay_s": DECISION_DELAY_S,
        "decision_time_s": decision_time,
        "feature_max_time_s": feature_time,
        "decision_index": index,
        "base_eligible": bool(eligible),
        "base_failure_reason": reason,
        "c0_scorable": False,
        "c0_failure_reason": "BASE_EVENT_UNSCORABLE" if not eligible else "",
        **{name: np.nan for name in C0_FEATURES + P_FEATURES + EUS_FEATURES + SPECTRAL_FEATURES},
        "spectral_scorable": False,
        "spectral_failure_reason": "BASE_EVENT_UNSCORABLE" if not eligible else "",
    }


def build_164_features(cycles_root: Path, labels_path: Path,
                       reference_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute the V2.1 feature schema from raw 164 cycles at confirm + 0.5 s."""
    cycles_root, labels_path, reference_root = map(Path, (cycles_root, labels_path, reference_root))
    assert_v3_paths([cycles_root, labels_path, reference_root])
    manifest = pd.read_csv(cycles_root / "nvc_cycle_manifest.csv")
    manifest = manifest[manifest["subject"].isin(SUBJECTS_164)].copy()
    actual_subjects = tuple(sorted(manifest["subject"].astype(str).unique()))
    if actual_subjects != tuple(sorted(SUBJECTS_164)):
        raise RuntimeError(f"SPARC164 subject mismatch: {actual_subjects}")
    labels = pd.read_csv(labels_path)
    labels = labels[labels["subject"].isin(SUBJECTS_164)].copy()
    if labels["event_uid"].duplicated().any():
        raise RuntimeError("SPARC164 frozen event_uid is not unique")

    prior_sigma, prior_dpdt = _stage_a_population_priors(reference_root)
    histories = {subject: AdaptiveHistory() for subject in SUBJECTS_164}
    rows = []
    audit_rows = []
    for _, manifest_row in manifest.sort_values(["subject", "cycle_start_s"]).iterrows():
        subject = str(manifest_row["subject"])
        cycle_id = str(manifest_row["cycle_id"])
        load_row = manifest_row.copy()
        load_row["dsd_cycle_id"] = cycle_id
        cycle = load_cycle(cycles_root, load_row)
        delta, events, adaptive = adaptive_local_pressure_events(
            cycle, histories[subject], prior_sigma, prior_dpdt)
        eus_env, eus_valid = causal_eus_envelope_100hz(cycle)
        frozen_cycle = labels[(labels["subject"] == subject)
                              & (labels["cycle_id"] == cycle_id)].sort_values("confirm_time_s")
        events = sorted(events, key=lambda event: float(event.get("confirm_index") or np.inf))
        if len(events) != len(frozen_cycle):
            raise RuntimeError(
                f"Frozen/detected event count mismatch for {subject}/{cycle_id}: "
                f"{len(frozen_cycle)} != {len(events)}")
        t = np.asarray(cycle["t_abs_s"], dtype=np.float64)
        for event, (_, frozen) in zip(events, frozen_cycle.iterrows()):
            confirm_index = event.get("confirm_index")
            confirm_time = (float(t[int(confirm_index)])
                            if confirm_index is not None and 0 <= int(confirm_index) < len(t)
                            else np.nan)
            frozen_confirm = float(frozen["confirm_time_s"])
            if not np.isfinite(confirm_time) and not np.isfinite(frozen_confirm):
                confirm_error = 0.0
            elif np.isfinite(confirm_time) and np.isfinite(frozen_confirm):
                confirm_error = abs(confirm_time - frozen_confirm)
            else:
                confirm_error = np.inf
            if confirm_error > 1.0 / C.DP_FS_HZ + 1e-9:
                raise RuntimeError(
                    f"Frozen confirmation mismatch for {frozen['event_uid']}: {confirm_error:.6f}s")
            decision_time = confirm_time + DECISION_DELAY_S if np.isfinite(confirm_time) else np.nan
            index = int(np.searchsorted(t, decision_time, side="left")) if np.isfinite(decision_time) else -1
            if 0 <= index < len(t) and t[index] > decision_time + 1e-9:
                index -= 1
            feature_time = float(t[index]) if 0 <= index < len(t) else np.nan
            eligible = bool(confirm_index is not None and 0 <= index < len(t)
                            and not bool(event.get("data_invalid", False))
                            and not bool(event.get("adaptive_warmup", True)))
            if confirm_index is None:
                reason = "NO_CONFIRM_INDEX"
            elif not (0 <= index < len(t)):
                reason = "OUTSIDE_CYCLE"
            elif bool(event.get("data_invalid", False)):
                reason = "DATA_INVALID"
            elif bool(event.get("adaptive_warmup", True)):
                reason = "ADAPTIVE_WARMUP_INCOMPLETE"
            else:
                reason = ""
            row = _empty_feature_row(subject, cycle_id, frozen, confirm_time,
                                     decision_time, feature_time, index, eligible, reason)
            if eligible:
                c0_features = decision_feature_at_index(
                    cycle, delta, eus_env, eus_valid, adaptive, index, event)
                if c0_features is not None:
                    row.update({name: c0_features[name] for name in C0_FEATURES})
                    row["c0_scorable"] = True
                else:
                    row["c0_failure_reason"] = "C0_FEATURE_UNSCORABLE"
                pressure_features = _pressure_features(delta, adaptive, index, pd.Series(event))
                if pressure_features is None:
                    row["base_eligible"] = False
                    row["base_failure_reason"] = "PRESSURE_FEATURE_UNSCORABLE"
                    row["spectral_failure_reason"] = "BASE_EVENT_UNSCORABLE"
                else:
                    row.update(pressure_features)
                    eus_features = _eus_features(delta, eus_env, eus_valid, index)
                    if eus_features is not None:
                        row.update(eus_features)
                    spectral, spectral_reason = _low_frequency_feature(cycle, delta, index)
                    row["relative_pressure_power_0p2_0p6"] = spectral
                    row["spectral_scorable"] = bool(np.isfinite(spectral))
                    row["spectral_failure_reason"] = spectral_reason
            rows.append(row)
            audit_rows.append({
                "dataset": "164", "subject": subject, "cycle_id": cycle_id,
                "event_uid": str(frozen["event_uid"]), "teacher_label": str(frozen["teacher_label"]),
                "frozen_confirm_time_s": frozen_confirm, "recomputed_confirm_time_s": confirm_time,
                "confirm_time_error_s": confirm_error, "feature_max_time_s": feature_time,
                "decision_time_s": decision_time,
                "causal": bool(
                    (not np.isfinite(feature_time) and not np.isfinite(decision_time))
                    or (np.isfinite(feature_time) and np.isfinite(decision_time)
                        and feature_time <= decision_time + 1e-9)),
            })
    result = pd.DataFrame(rows)
    audit = pd.DataFrame(audit_rows)
    if len(result) != len(labels) or not audit["causal"].all():
        raise RuntimeError("SPARC164 feature reconstruction failed coverage or causality")
    return result, audit


def load_combined_features(v21_root: Path, v2_root: Path, cycles_164_root: Path,
                           labels_164_path: Path, reference_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_338 = pd.read_csv(Path(v21_root) / "event_features_v21.csv")
    features_338.insert(0, "dataset", "338")
    if tuple(sorted(features_338["subject"].unique())) != tuple(sorted(SUBJECTS_338)):
        raise RuntimeError("V2.1 feature artifact does not contain the frozen Dataset338 animals")
    c0_338 = pd.read_csv(Path(v2_root) / "event_features_v2.csv")
    c0_338 = c0_338[np.isclose(pd.to_numeric(c0_338["decision_delay_s"]), DECISION_DELAY_S)].copy()
    c0_columns = ["event_uid", "base_eligible", "base_failure_reason", *C0_FEATURES]
    c0_338 = c0_338[c0_columns].rename(columns={
        "base_eligible": "c0_scorable", "base_failure_reason": "c0_failure_reason"})
    if c0_338["event_uid"].duplicated().any() or len(c0_338) != len(features_338):
        raise RuntimeError("Dataset338 C0 feature population does not match V2.1")
    features_338 = features_338.merge(c0_338, on="event_uid", how="left", validate="one_to_one")
    features_164, audit = build_164_features(cycles_164_root, labels_164_path, reference_root)
    columns = list(dict.fromkeys(list(features_338.columns) + list(features_164.columns)))
    combined = pd.concat([features_338.reindex(columns=columns),
                          features_164.reindex(columns=columns)], ignore_index=True, sort=False)
    if combined["event_uid"].duplicated().any():
        raise RuntimeError("Combined event_uid is not unique")
    return combined, audit


def model_frame(features: pd.DataFrame, model_name: str) -> pd.DataFrame:
    columns = list(ALL_MODEL_FEATURES[model_name])
    out = features[features["teacher_label"].isin(TARGET_LABELS)].copy()
    out["target"] = out["teacher_label"].map(LABEL_TO_TARGET).astype(int)
    if model_name == "C0":
        feature_complete = np.isfinite(out[columns].to_numpy(dtype=float)).all(axis=1)
        base_ok = out["c0_scorable"].map(_as_bool)
        out["model_scorable"] = base_ok & feature_complete
        base_reason = out["c0_failure_reason"]
    else:
        feature_complete = np.isfinite(out[list(P_FEATURES)].to_numpy(dtype=float)).all(axis=1)
        base_ok = out["base_eligible"].map(_as_bool)
        out["model_scorable"] = base_ok & feature_complete
        base_reason = out["base_failure_reason"]
    if model_name in ("PE_SPECTRAL_COMMON", "PEF"):
        out["model_scorable"] &= out["spectral_scorable"].map(_as_bool)
    out["model_failure_reason"] = np.where(
        ~base_ok, base_reason,
        np.where(~feature_complete, "FEATURE_UNSCORABLE",
                 np.where((model_name in ("PE_SPECTRAL_COMMON", "PEF"))
                          & ~out["spectral_scorable"].map(_as_bool),
                          out["spectral_failure_reason"], "")))
    assert_feature_schema_safe(columns)
    return out


def fit_model(train: pd.DataFrame, model_name: str,
              allowed_subjects: Sequence[str]) -> Pipeline:
    scored = train[train["model_scorable"].map(_as_bool)].copy()
    allowed = tuple(sorted(map(str, allowed_subjects)))
    if set(scored["subject"].astype(str)) - set(allowed):
        raise AssertionError("Held-out animal entered V3 model fit")
    if not set(scored["subject"].astype(str)):
        raise ValueError("No training animal has scorable events")
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True,
                                  keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                        max_iter=2000, random_state=RANDOM_STATE)),
    ])
    weights = expanded_animal_class_weights(scored)
    pipeline.fit(scored[list(ALL_MODEL_FEATURES[model_name])], scored["target"],
                 logistic__sample_weight=weights)
    pipeline.fit_subjects_ = tuple(sorted(scored["subject"].astype(str).unique()))
    pipeline.fit_features_ = tuple(ALL_MODEL_FEATURES[model_name])
    return pipeline


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray,
                     weights: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import balanced_accuracy_score, recall_score
    candidates = sorted(set(np.r_[0.0, probabilities,
                                  np.nextafter(probabilities, np.inf), 1.0000001]))
    best = None
    for threshold in candidates:
        predicted = probabilities >= threshold
        score = balanced_accuracy_score(y_true, predicted, sample_weight=weights)
        specificity = recall_score(y_true, predicted, pos_label=0,
                                    sample_weight=weights, zero_division=0)
        key = (float(score), float(specificity), float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1], best[0][0]


def run_loso(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts, fold_rows, coefficient_rows = [], [], []
    for model_name in MODEL_ORDER:
        frame = model_frame(features, model_name)
        for held_subject in SUBJECTS:
            train_subjects = [subject for subject in SUBJECTS if subject != held_subject]
            train = frame[frame["subject"].isin(train_subjects)].copy()
            test = frame[frame["subject"] == held_subject].copy()
            model = fit_model(train, model_name, train_subjects)
            train_scored = train[train["model_scorable"].map(_as_bool)].copy()
            train_probability = model.predict_proba(
                train_scored[list(ALL_MODEL_FEATURES[model_name])])[:, 1]
            threshold, training_ba = select_threshold(
                train_scored["target"].to_numpy(dtype=int), train_probability,
                expanded_animal_class_weights(train_scored))
            test["p_nvc"] = np.nan
            scorable = test["model_scorable"].map(_as_bool)
            if scorable.any():
                test.loc[scorable, "p_nvc"] = model.predict_proba(
                    test.loc[scorable, list(ALL_MODEL_FEATURES[model_name])])[:, 1]
            test["threshold_train_only"] = threshold
            test["predicted_nvc"] = test["p_nvc"].ge(threshold) & test["p_nvc"].notna()
            test["predicted_label"] = np.where(
                test["p_nvc"].isna(), "UNSCORABLE",
                np.where(test["predicted_nvc"], "NVC_CORE", "PREVOID_PROGRESSIVE"))
            test["correct"] = np.where(
                test["p_nvc"].isna(), np.nan,
                (test["predicted_label"] == test["teacher_label"]).astype(float))
            test["model"] = model_name
            test["held_out_subject"] = held_subject
            test["train_subjects"] = "+".join(train_subjects)
            test["threshold_training_balanced_accuracy"] = training_ba
            fold_rows.append({
                "model": model_name, "held_out_subject": held_subject,
                "held_out_dataset": DATASET_BY_SUBJECT[held_subject],
                "train_subjects": "+".join(train_subjects),
                "threshold_train_only": threshold,
                "threshold_training_balanced_accuracy": training_ba,
                **_metric_values(test),
            })
            names = model.named_steps["imputer"].get_feature_names_out(
                list(ALL_MODEL_FEATURES[model_name]))
            coefficients = model.named_steps["logistic"].coef_[0]
            for name, value in zip(names, coefficients):
                coefficient_rows.append({
                    "model": model_name, "held_out_subject": held_subject,
                    "train_subjects": "+".join(train_subjects),
                    "threshold_train_only": threshold, "term": name,
                    "coefficient": float(value),
                })
            prediction_parts.append(test)
    predictions = pd.concat(prediction_parts, ignore_index=True, sort=False)
    return predictions, pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def _aggregate(frame: pd.DataFrame) -> dict:
    pooled = _metric_values(frame)
    frozen_nvc = int((frame["target"] == 1).sum())
    per_animal = [_metric_values(group) for _, group in frame.groupby("subject")]
    def macro(name: str) -> float:
        values = [row[name] for row in per_animal if np.isfinite(row[name])]
        return float(np.mean(values)) if values else np.nan
    return {
        "n_events": pooled["n_events"], "n_scorable": pooled["n_scorable"],
        "n_nvc_scorable": pooled["n_nvc_scorable"],
        "n_prevoid_scorable": pooled["n_prevoid_scorable"],
        "pooled_AUROC": pooled["AUROC"], "pooled_AUPRC": pooled["AUPRC"],
        "macro_AUROC": macro("AUROC"), "macro_AUPRC": macro("AUPRC"),
        "macro_sensitivity": macro("sensitivity"), "macro_PPV": macro("PPV"),
        "macro_balanced_accuracy": macro("balanced_accuracy"),
        "NVC_hit_total": pooled["TP"], "PREVOID_FP_total": pooled["FP"],
        "NVC_hit_rate": (float(pooled["TP"] / pooled["n_nvc_scorable"])
                         if pooled["n_nvc_scorable"] else np.nan),
        "frozen_NVC_denominator": frozen_nvc,
        "frozen_NVC_hit_rate": float(pooled["TP"] / frozen_nvc) if frozen_nvc else np.nan,
        "animals_with_nvc_hit": int(sum(_metric_values(group)["TP"] > 0
                                         for _, group in frame.groupby("subject"))),
    }


def build_model_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_ORDER:
        model = predictions[predictions["model"] == model_name]
        row = {"model": model_name, "n_features": len(ALL_MODEL_FEATURES[model_name]),
               **_aggregate(model)}
        for dataset in ("338", "164"):
            metrics = _aggregate(model[model["dataset"] == dataset])
            for key in ("n_scorable", "n_nvc_scorable", "n_prevoid_scorable",
                        "pooled_AUROC", "pooled_AUPRC", "macro_sensitivity", "macro_PPV"):
                row[f"{dataset}_{key}"] = metrics[key]
            row[f"{dataset}_NVC_hit_total"] = metrics["NVC_hit_total"]
            row[f"{dataset}_NVC_hit_rate"] = metrics["NVC_hit_rate"]
            row[f"{dataset}_frozen_NVC_denominator"] = metrics["frozen_NVC_denominator"]
            row[f"{dataset}_frozen_NVC_hit_rate"] = metrics["frozen_NVC_hit_rate"]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final_candidate(features: pd.DataFrame, model_name: str) -> tuple[dict, pd.DataFrame]:
    frame = model_frame(features, model_name)
    model = fit_model(frame, model_name, SUBJECTS)
    scored = frame[frame["model_scorable"].map(_as_bool)].copy()
    probability = model.predict_proba(scored[list(ALL_MODEL_FEATURES[model_name])])[:, 1]
    threshold, training_ba = select_threshold(
        scored["target"].to_numpy(dtype=int), probability,
        expanded_animal_class_weights(scored))
    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    logistic = model.named_steps["logistic"]
    names = list(imputer.get_feature_names_out(list(ALL_MODEL_FEATURES[model_name])))
    bundle = {
        "model_version": "NVC_V3_EXPANDED_DEVELOPMENT_CANDIDATE",
        "deployment_ready": False,
        "stimulation_enabled": False,
        "development_datasets": ["338", "164"],
        "external_validation_remaining": False,
        "model": model_name, "decision_delay_s": DECISION_DELAY_S,
        "input_features": list(ALL_MODEL_FEATURES[model_name]),
        "transformed_features": names,
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
        "coefficient_nvc": logistic.coef_[0].tolist(),
        "intercept_nvc": float(logistic.intercept_[0]),
        "analysis_threshold": threshold,
        "training_balanced_accuracy": training_ba,
        "fit_subjects": list(SUBJECTS),
    }
    coefficients = pd.DataFrame({"term": names, "coefficient": logistic.coef_[0]})
    coefficients = pd.concat([coefficients, pd.DataFrame([{
        "term": "__intercept__", "coefficient": float(logistic.intercept_[0])}])],
        ignore_index=True)
    return bundle, coefficients


def _fmt(value) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{float(value):.3f}"


def report_text(summary: dict, comparison: pd.DataFrame,
                per_animal: pd.DataFrame) -> str:
    lines = [
        "# Dataset338 + Dataset164 NVC V3 expanded development",
        "",
        "V3只扩展开发动物，不改变V2.1的标签任务、0.5 s因果决策、特征集合、L2逻辑回归或训练折阈值规则。",
        "",
        "## Cohort",
        "",
        f"- 动物：8（338={list(SUBJECTS_338)}；164={list(SUBJECTS_164)}）",
        f"- 周期：{summary['cycle_count']}；主任务事件：{summary['target_event_count']}",
        f"- 标签：{summary['label_counts']}",
        "- Dataset164已进入开发集，因此不再是V3的外部验证集。",
        "",
        "## Eight-animal LOSO",
        "",
        "| Model | Scorable NVC/PREVOID | AUROC | AUPRC | Macro AUROC | Macro AUPRC | Scorable hit rate | Frozen NVC hit | PPV | PREVOID FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.n_nvc_scorable)}/{int(row.n_prevoid_scorable)} | "
            f"{_fmt(row.pooled_AUROC)} | {_fmt(row.pooled_AUPRC)} | {_fmt(row.macro_AUROC)} | "
            f"{_fmt(row.macro_AUPRC)} | {_fmt(row.NVC_hit_rate)} | "
            f"{int(row.NVC_hit_total)}/{int(row.frozen_NVC_denominator)} ({_fmt(row.frozen_NVC_hit_rate)}) | "
            f"{_fmt(row.macro_PPV)} | {int(row.PREVOID_FP_total)} |")
    lines += ["", "## Dataset subgroup audit", "",
              "| Model | 338 AUROC/AUPRC | 164 AUROC/AUPRC | 338 NVC hit | 164 NVC hit |",
              "|---|---:|---:|---:|---:|"]
    for _, row in comparison.iterrows():
        lines.append(
            f"| {row['model']} | {_fmt(row['338_pooled_AUROC'])}/{_fmt(row['338_pooled_AUPRC'])} | "
            f"{_fmt(row['164_pooled_AUROC'])}/{_fmt(row['164_pooled_AUPRC'])} | "
            f"{int(row['338_NVC_hit_total'])}/{int(row['338_frozen_NVC_denominator'])} ({_fmt(row['338_frozen_NVC_hit_rate'])}) | "
            f"{int(row['164_NVC_hit_total'])}/{int(row['164_frozen_NVC_denominator'])} ({_fmt(row['164_frozen_NVC_hit_rate'])}) |")
    lines += ["", "## Per-animal LOSO", "",
              "| Model | Animal | Dataset | NVC/PREVOID | AUROC | AUPRC | Sensitivity | PPV | BA |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in per_animal.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.held_out_subject} | {row.held_out_dataset} | "
            f"{int(row.n_nvc_scorable)}/{int(row.n_prevoid_scorable)} | {_fmt(row.AUROC)} | "
            f"{_fmt(row.AUPRC)} | {_fmt(row.sensitivity)} | {_fmt(row.PPV)} | "
            f"{_fmt(row.balanced_accuracy)} |")
    lines += [
        "", "## Development decision", "",
        f"- best_model: `{summary['best_model']}`",
        f"- highest_nvc_hit_model: `{summary['highest_nvc_hit_model']}`",
        f"- development_status: `{summary['development_status']}`",
        "- stimulation_enabled: `false`",
        "- PE_SPECTRAL_COMMON是PEF共同可评分子集上的归因参考，不是独立部署候选。",
        "- V3候选包使用全部8只动物拟合，仅供后续独立队列验证，不是部署模型。",
        "", f"结论：{summary['final_conclusion']}", "",
    ]
    return "\n".join(lines)


def run(v21_root: Path, v2_root: Path, cycles_164_root: Path, labels_164_path: Path,
        reference_root: Path, output_root: Path, overwrite: bool = False,
        reuse_features: bool = False) -> dict:
    paths = list(map(Path, (v21_root, v2_root, cycles_164_root, labels_164_path,
                           reference_root, output_root)))
    v21_root, v2_root, cycles_164_root, labels_164_path, reference_root, output_root = paths
    assert_v3_paths(paths)
    cached_features = output_root / "event_features_v3.csv"
    cached_audit = output_root / "sparc164_reconstruction_audit_v3.csv"
    can_reuse = bool(reuse_features and cached_features.exists() and cached_audit.exists()
                     and set(C0_FEATURES).issubset(pd.read_csv(cached_features, nrows=0).columns))
    if output_root.exists() and any(output_root.iterdir()) and not can_reuse:
        if not overwrite:
            raise FileExistsError(f"Output exists: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if can_reuse:
        features = pd.read_csv(cached_features)
        reconstruction_audit = pd.read_csv(cached_audit)
    else:
        features, reconstruction_audit = load_combined_features(
            v21_root, v2_root, cycles_164_root, labels_164_path, reference_root)
    features["dataset"] = features["dataset"].astype(str)
    target = features[features["teacher_label"].isin(TARGET_LABELS)]
    features.to_csv(output_root / "event_features_v3.csv", index=False)
    reconstruction_audit.to_csv(output_root / "sparc164_reconstruction_audit_v3.csv", index=False)
    predictions, fold_metrics, fold_coefficients = run_loso(features)
    comparison = build_model_comparison(predictions)

    # Spectral paths have much lower causal coverage; select among full-coverage models first.
    full_coverage = comparison[comparison["model"].isin(["C0", "P", "PE"])].copy()
    best_model = str(full_coverage.sort_values(
        ["pooled_AUPRC", "pooled_AUROC"], ascending=False).iloc[0]["model"])
    highest_hit_model = str(comparison.sort_values(
        ["NVC_hit_rate", "NVC_hit_total", "pooled_AUPRC"], ascending=False).iloc[0]["model"])
    best = comparison[comparison["model"] == best_model].iloc[0]
    prevalence = float((target["teacher_label"] == "NVC_CORE").mean())
    positive_animals = int((fold_metrics[(fold_metrics["model"] == best_model)
                                         & (fold_metrics["n_nvc_scorable"] > 0)]["TP"] > 0).sum())
    learned = bool(best["pooled_AUROC"] >= 0.65
                   and best["pooled_AUPRC"] >= prevalence + 0.10
                   and best["macro_balanced_accuracy"] >= 0.60
                   and positive_animals >= 4)
    status = "PASS_V3_EXPANDED_NVC_FEATURE_LEARNING" if learned else "HOLD_V3_NO_STABLE_NVC_FEATURE_LEARNING"
    final_conclusion = (
        "扩展到8只动物后，V2.1特征空间达到预注册的研究级跨动物学习门槛；仍需新的独立SCI队列验证。"
        if learned else
        "扩展到8只动物后，V2.1特征空间仍未达到研究级稳定跨动物NVC学习门槛。")

    bundle, final_coefficients = fit_final_candidate(features, best_model)
    write_json(output_root / "candidate_model_v3.json", bundle)
    final_coefficients.to_csv(output_root / "candidate_model_coefficients_v3.csv", index=False)
    prediction_columns = [
        "dataset", "subject", "cycle_id", "event_id", "event_uid", "teacher_label",
        "model", "decision_time_s", "feature_max_time_s", "p_nvc",
        "threshold_train_only", "predicted_label", "correct", "model_scorable",
        "model_failure_reason", "held_out_subject", "train_subjects",
    ]
    predictions[prediction_columns].to_csv(output_root / "event_predictions_v3.csv", index=False)
    fold_metrics.to_csv(output_root / "per_animal_metrics_v3.csv", index=False)
    fold_coefficients.to_csv(output_root / "fold_model_coefficients_v3.csv", index=False)
    comparison.to_csv(output_root / "model_comparison_v3.csv", index=False)

    label_counts = features["teacher_label"].value_counts().astype(int).to_dict()
    cycle_count = int(features[["dataset", "subject", "cycle_id"]].drop_duplicates().shape[0])
    summary = {
        "development_status": status, "best_model": best_model,
        "highest_nvc_hit_model": highest_hit_model,
        "stimulation_enabled": False, "decision_delay_s": DECISION_DELAY_S,
        "development_datasets": ["338", "164"],
        "dataset164_role": "DEVELOPMENT; NO_LONGER_EXTERNAL_VALIDATION",
        "external_validation_remaining": False,
        "subjects": list(SUBJECTS), "subject_count": len(SUBJECTS),
        "cycle_count": cycle_count, "target_event_count": int(len(target)),
        "label_counts": label_counts,
        "nvc_counts_by_subject": features[features["teacher_label"] == "NVC_CORE"].groupby(
            "subject").size().reindex(SUBJECTS, fill_value=0).astype(int).to_dict(),
        "prevalence_target_population": prevalence,
        "positive_animals_with_hit": positive_animals,
        "learning_gate": {
            "pooled_AUROC_min": 0.65, "pooled_AUPRC_above_prevalence": 0.10,
            "macro_balanced_accuracy_min": 0.60, "positive_animals_with_hit_min": 4,
        },
        "models": comparison.set_index("model").to_dict(orient="index"),
        "PE_minus_P": {
            "delta_pooled_AUROC": float(comparison.loc[comparison["model"] == "PE", "pooled_AUROC"].iloc[0]
                                          - comparison.loc[comparison["model"] == "P", "pooled_AUROC"].iloc[0]),
            "delta_pooled_AUPRC": float(comparison.loc[comparison["model"] == "PE", "pooled_AUPRC"].iloc[0]
                                          - comparison.loc[comparison["model"] == "P", "pooled_AUPRC"].iloc[0]),
            "delta_NVC_hit": int(comparison.loc[comparison["model"] == "PE", "NVC_hit_total"].iloc[0]
                                 - comparison.loc[comparison["model"] == "P", "NVC_hit_total"].iloc[0]),
            "delta_PREVOID_FP": int(comparison.loc[comparison["model"] == "PE", "PREVOID_FP_total"].iloc[0]
                                    - comparison.loc[comparison["model"] == "P", "PREVOID_FP_total"].iloc[0]),
        },
        "actual_read_files": [
            str((v21_root / "event_features_v21.csv").resolve()),
            str((v2_root / "event_features_v2.csv").resolve()),
            str((cycles_164_root / "nvc_cycle_manifest.csv").resolve()),
            str(labels_164_path.resolve()),
            str((reference_root / "subject_adaptive_params.csv").resolve()),
        ],
        "raw_164_cycle_root": str(cycles_164_root.resolve()),
        "feature_schema_unchanged_from_v21": True,
        "final_conclusion": final_conclusion,
    }
    write_json(output_root / "v3_summary.json", summary)
    schema = {
        "development_datasets": ["338", "164"], "subjects": list(SUBJECTS),
        "task": "NVC_CORE=1 vs PREVOID_PROGRESSIVE=0",
        "excluded_labels": ["GREY_ZONE", "INVALID"],
        "decision_delay_s": DECISION_DELAY_S,
        "models": {name: list(columns) for name, columns in ALL_MODEL_FEATURES.items()},
        "preprocessing": "training-fold median imputer -> StandardScaler -> L2 logistic C=1",
        "sample_weight": "animal equal; class balanced within animals that contain both classes",
        "validation": "strict eight-fold leave-one-animal-out",
        "threshold": "training-fold weighted balanced accuracy",
        "voidguard_used_for_selection": False, "stimulation_enabled": False,
    }
    write_json(output_root / "feature_schema_v3.json", schema)
    (output_root / "V3_REPORT.md").write_text(
        report_text(summary, comparison, fold_metrics), encoding="utf-8")
    from .visualization import generate_plots
    generate_plots(output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v21-root", type=Path,
                        default=Path("data/NVC_V3/source_inputs/v21"))
    parser.add_argument("--v2-root", type=Path,
                        default=Path("data/NVC_V3/source_inputs/v2"))
    parser.add_argument("--cycles-164-root", type=Path,
                        default=Path("data/SPARC164_cycles"))
    parser.add_argument("--labels-164-path", type=Path,
                        default=Path("data/SPARC164_nvc_results/sparc164_teacher_labels.csv"))
    parser.add_argument("--reference-root", type=Path,
                        default=Path("data/DSD_nvc_results"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("data/NVC_V3"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    summary = run(args.v21_root, args.v2_root, args.cycles_164_root, args.labels_164_path,
                  args.reference_root, args.output_root, args.overwrite,
                  args.reuse_features)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
