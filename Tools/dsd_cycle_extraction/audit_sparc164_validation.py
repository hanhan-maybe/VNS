"""Audit Dataset164 cycle keys, frozen Dataset338 model use, and event metrics.

This module is read-only with respect to models: it only verifies and reports
the already-generated external-validation artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.casefold().isin({"true", "1", "yes"})


def audit(cycles_root: Path, results_root: Path, reference_root: Path) -> dict:
    cycles_root, results_root, reference_root = map(Path, (cycles_root, results_root, reference_root))
    manifest_path = cycles_root / "nvc_cycle_manifest.csv"
    labels_path = results_root / "sparc164_teacher_labels.csv"
    scores_path = results_root / "sparc164_frozen_model_scores.csv"
    model_path = reference_root / "final_model.json"
    freeze_path = reference_root / "dsd_stage_a_freeze.json"
    priors_path = reference_root / "subject_adaptive_params.csv"
    external_summary_path = results_root / "sparc164_external_validation_summary.json"

    manifest = pd.read_csv(manifest_path)
    keys = ["subject", "cycle_id"]
    duplicate_mask = manifest.duplicated(keys, keep=False)
    counts = manifest.groupby("subject").size().astype(int).to_dict()
    cycle_audit = {
        "manifest": str(manifest_path.resolve()),
        "row_count": int(len(manifest)),
        "unique_subject_cycle_key_count": int(manifest[keys].drop_duplicates().shape[0]),
        "duplicate_key_count": int(manifest.loc[duplicate_mask, keys].drop_duplicates().shape[0]),
        "duplicate_keys": manifest.loc[duplicate_mask, keys].drop_duplicates().to_dict("records"),
        "counts_by_subject": counts,
        "adjudicated_cycle_count": int(manifest[keys].drop_duplicates().shape[0]),
        "adjudication": "35_CYCLES" if len(manifest) == 35 and not duplicate_mask.any() else "REVIEW_REQUIRED",
    }
    _write_json(results_root / "sparc164_cycle_key_audit.json", cycle_audit)

    model = json.loads(model_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    external_summary = json.loads(external_summary_path.read_text(encoding="utf-8"))
    priors = pd.read_csv(priors_path)
    scores = pd.read_csv(scores_path)
    order = list(model["feature_order"])
    x = scores[order].to_numpy(float)
    z = (x - np.asarray(model["scaler_mean"], float)) / np.asarray(model["scaler_scale"], float)
    coefficient = np.asarray(model["coefficient_void_risk"], float)
    intercept = float(model.get("intercept", model.get("intercept_void_risk", 0.0)))
    expected_risk = 1.0 / (1.0 + np.exp(-(intercept + z @ coefficient)))
    expected_nvc = 1.0 - expected_risk
    threshold = float(model["analysis_threshold"])
    expected_call = expected_nvc >= threshold
    recorded_call = _bool(scores["offline_analysis_call"]).to_numpy()
    max_probability_difference = float(np.max(np.abs(expected_nvc - scores["p_nvc"].to_numpy(float))))
    freeze_audit = {
        "external_dataset": "164",
        "reference_dataset": "338",
        "reference_root": str(reference_root.resolve()),
        "reference_training_subjects": freeze["loso_summary"]["training_subjects"],
        "reference_fixed_cycle_counts": freeze["fixed_cycle_counts"],
        "model_file": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "population_prior_file": str(priors_path.resolve()),
        "population_prior_sha256": _sha256(priors_path),
        "population_prior_subjects": priors["subject"].astype(str).tolist(),
        "selected_model": model["selected_model"],
        "decision_delay_s": float(model["decision_delay_s"]),
        "feature_order": order,
        "scaler_mean": model["scaler_mean"],
        "scaler_scale": model["scaler_scale"],
        "coefficient_void_risk": model["coefficient_void_risk"],
        "intercept_void_risk": intercept,
        "analysis_threshold": threshold,
        "safety_threshold": float(model["safety_threshold"]),
        "external_summary_feature_order_source": external_summary["feature_order_source"],
        "scaler_refit": bool(external_summary["scaler_refit"]),
        "threshold_refit": bool(external_summary["threshold_refit"]),
        "decision_delay_refit": bool(external_summary["decision_delay_refit"]),
        "score_rows_recomputed": int(len(scores)),
        "max_abs_p_nvc_difference_from_frozen_recalculation": max_probability_difference,
        "call_mismatch_count_from_frozen_threshold_recalculation": int(np.sum(expected_call != recorded_call)),
        "frozen_338_configuration_verified": bool(
            set(freeze["loso_summary"]["training_subjects"]) == {"STxF26", "STxF27", "STxF29"}
            and set(priors["subject"].astype(str)) == {"STxF26", "STxF27", "STxF29"}
            and not external_summary["scaler_refit"]
            and not external_summary["threshold_refit"]
            and not external_summary["decision_delay_refit"]
            and max_probability_difference < 1e-12
            and not np.any(expected_call != recorded_call)
        ),
    }
    _write_json(results_root / "sparc164_model_freeze_audit.json", freeze_audit)

    labels = pd.read_csv(labels_path)
    if labels["event_uid"].duplicated().any() or scores["event_uid"].duplicated().any():
        raise RuntimeError("event_uid is not unique in event-level artifacts")
    score_columns = ["event_uid", "p_void_risk", "p_nvc", "offline_analysis_call", "model_name"]
    matched = labels.merge(scores[score_columns], on="event_uid", how="left", validate="one_to_one")
    matched["model_scored"] = matched["p_nvc"].notna()
    matched["offline_analysis_call"] = _bool(matched["offline_analysis_call"].fillna(False))
    matched["is_reference_nvc"] = matched["automatic_teacher_label"].eq("NVC_CORE")
    matched["true_positive"] = matched["offline_analysis_call"] & matched["is_reference_nvc"]
    matched["false_positive"] = matched["offline_analysis_call"] & ~matched["is_reference_nvc"]
    matched["void_period_false_positive"] = matched["offline_analysis_call"] & matched[
        "automatic_teacher_label"
    ].isin(["PREVOID_PROGRESSIVE", "VOID_CONFIRMED"])
    matched.to_csv(results_root / "sparc164_event_level_call_match.csv", index=False)

    rows = []
    for subject, cycle_rows in manifest.groupby("subject", sort=True):
        events = matched[matched["subject"].astype(str) == str(subject)]
        hours = float(pd.to_numeric(cycle_rows["cycle_duration_s"]).sum() / 3600.0)
        positives = int(events["is_reference_nvc"].sum())
        calls = int(events["offline_analysis_call"].sum())
        tp = int(events["true_positive"].sum())
        fp = int(events["false_positive"].sum())
        void_fp = int(events["void_period_false_positive"].sum())
        rows.append({
            "subject": subject,
            "nvc_cycle_count": int(len(cycle_rows)),
            "valid_analysis_hours": hours,
            "teacher_nvc_events": positives,
            "model_calls": calls,
            "true_positive_calls": tp,
            "false_positive_calls": fp,
            "void_period_false_positive_calls": void_fp,
            "sensitivity": tp / positives if positives else np.nan,
            "ppv": tp / calls if calls else np.nan,
            "false_triggers_per_hour": fp / hours if hours else np.nan,
            "void_period_false_triggers_per_hour": void_fp / hours if hours else np.nan,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(results_root / "sparc164_event_level_metrics_by_subject.csv", index=False)

    total_hours = float(metrics["valid_analysis_hours"].sum())
    total_positive = int(metrics["teacher_nvc_events"].sum())
    total_calls = int(metrics["model_calls"].sum())
    total_tp = int(metrics["true_positive_calls"].sum())
    total_fp = int(metrics["false_positive_calls"].sum())
    total_void_fp = int(metrics["void_period_false_positive_calls"].sum())
    summary = {
        "cycle_key_count": cycle_audit["adjudicated_cycle_count"],
        "event_match_row_count": int(len(matched)),
        "scored_event_count": int(matched["model_scored"].sum()),
        "unscored_event_count": int((~matched["model_scored"]).sum()),
        "micro": {
            "valid_analysis_hours": total_hours,
            "teacher_nvc_events": total_positive,
            "model_calls": total_calls,
            "true_positive_calls": total_tp,
            "false_positive_calls": total_fp,
            "void_period_false_positive_calls": total_void_fp,
            "sensitivity": total_tp / total_positive if total_positive else None,
            "ppv": total_tp / total_calls if total_calls else None,
            "false_triggers_per_hour": total_fp / total_hours if total_hours else None,
            "void_period_false_triggers_per_hour": total_void_fp / total_hours if total_hours else None,
        },
        "animal_macro": {
            "sensitivity": float(metrics["sensitivity"].mean(skipna=True)),
            "sensitivity_evaluable_animals": int(metrics["sensitivity"].notna().sum()),
            "ppv": float(metrics["ppv"].mean(skipna=True)),
            "ppv_evaluable_animals": int(metrics["ppv"].notna().sum()),
            "false_triggers_per_hour": float(metrics["false_triggers_per_hour"].mean(skipna=True)),
            "void_period_false_triggers_per_hour": float(metrics["void_period_false_triggers_per_hour"].mean(skipna=True)),
        },
        "metric_definitions": {
            "sensitivity": "NVC_CORE model calls / all event-level NVC_CORE teacher events",
            "ppv": "NVC_CORE model calls / all offline_analysis_call events",
            "false_triggers_per_hour": "non-NVC_CORE model calls / summed NVC-cycle duration in hours",
            "void_period_false_positive_calls": "model calls labelled PREVOID_PROGRESSIVE or VOID_CONFIRMED",
            "animal_macro": "unweighted mean of per-animal metrics; undefined sensitivity/PPV omitted",
        },
    }
    _write_json(results_root / "sparc164_event_level_metrics_summary.json", summary)
    return {"cycle_audit": cycle_audit, "freeze_audit": freeze_audit, "metrics": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles-root", type=Path, default=Path("data/SPARC164_cycles"))
    parser.add_argument("--results-root", type=Path, default=Path("data/SPARC164_nvc_results"))
    parser.add_argument("--reference-root", type=Path, default=Path("data/DSD_nvc_results"))
    args = parser.parse_args()
    print(json.dumps(audit(args.cycles_root, args.results_root, args.reference_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
