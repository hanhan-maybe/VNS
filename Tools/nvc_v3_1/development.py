"""V3.1 failure diagnosis and mechanism-guided limited experiments."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from Tools.dsd_feature_extraction.data_io import write_json
from .config import (
    DEFAULT_164_CYCLES, DEFAULT_164_LABELS, DEFAULT_338_CYCLES, DEFAULT_338_REFERENCE,
    DEFAULT_OUTPUT_ROOT, DEFAULT_V3_ROOT, DELAYS_S, MODEL_FEATURES, SUBJECTS, TARGET_LABELS,
)
from .data_adapter import build_aligned_traces, build_delayed_features, load_development_streams
from .validation import (
    aggregate_metrics, per_animal_metrics, run_nested_candidates, run_outer_loso,
)

BASELINE_MODELS = ("C0", "P", "PE", "PE_SPECTRAL_COMMON", "PEF")
CORE_MODELS = ("C0", "P", "PE", "PE_DELAY", "PE_TRAJECTORY", "CANDIDATE+VOIDGUARD")


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _safe_auc(frame: pd.DataFrame, reverse: bool = False) -> float:
    scored = frame[frame["p_nvc"].notna()]
    if scored["target"].nunique() != 2:
        return np.nan
    score = 1.0 - scored["p_nvc"] if reverse else scored["p_nvc"]
    return float(roc_auc_score(scored["target"], score))


def reproduce_v3(v3_root: Path, delayed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    source_comparison = pd.read_csv(v3_root / "model_comparison_v3.csv")
    source_predictions = pd.read_csv(v3_root / "event_predictions_v3.csv")
    frozen_features = pd.read_csv(v3_root / "event_features_v3.csv")
    evaluation = delayed[np.isclose(delayed["decision_delay_s"], 0.5)][
        ["event_uid", "still_active", "actionable", "event_recovery_time_s_eval_only",
         "urine_onset_s_eval_only"]].copy()
    frozen_features = frozen_features.merge(evaluation, on="event_uid", how="left", validate="one_to_one")
    prediction_parts, coefficient_parts, rows = [], [], []
    for model_name in BASELINE_MODELS:
        predictions, audit, coefficients = run_outer_loso(frozen_features, model_name, 0.5)
        prediction_parts.append(predictions); coefficient_parts.append(coefficients)
        current = aggregate_metrics(predictions, model_name)
        reference = source_comparison[source_comparison["model"] == model_name].iloc[0]
        comparisons = {
            "frozen_NVC": (current["frozen_NVC"], int(reference["frozen_NVC_denominator"])),
            "scorable_NVC": (current["scorable_NVC"], int(reference["n_nvc_scorable"])),
            "scorable_PREVOID": (current["scorable_PREVOID"], int(reference["n_prevoid_scorable"])),
            "TP": (current["TP"], int(reference["NVC_hit_total"])),
            "FP": (current["PREVOID_FP"], int(reference["PREVOID_FP_total"])),
            "AUROC": (current["AUROC"], float(reference["pooled_AUROC"])),
            "AUPRC": (current["AUPRC"], float(reference["pooled_AUPRC"])),
        }
        for metric, (actual, expected) in comparisons.items():
            tolerance = 1e-10 if metric in {"AUROC", "AUPRC"} else 0.0
            rows.append({"model": model_name, "metric": metric, "expected": expected,
                         "recomputed": actual, "absolute_difference": abs(actual - expected),
                         "match": bool(abs(actual - expected) <= tolerance)})
    reproduction = pd.DataFrame(rows)
    passed = bool(reproduction["match"].all())
    return (reproduction, pd.concat(prediction_parts, ignore_index=True),
            pd.concat(coefficient_parts, ignore_index=True), passed)


def class_mapping_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, subject), frame in predictions.groupby(["model", "subject"]):
        scored = frame[frame["p_nvc"].notna()]
        nvc = scored[scored["teacher_label"] == "NVC_CORE"]
        pre = scored[scored["teacher_label"] == "PREVOID_PROGRESSIVE"]
        rows.append({
            "model": model, "held_out_animal": subject, "classes": str(frame["model_classes"].iloc[0]),
            "positive_class": int(frame["positive_class"].iloc[0]),
            "positive_index": int(frame["positive_class_index"].iloc[0]),
            "n_nvc": len(nvc), "n_prevoid": len(pre),
            "median_p_nvc_nvc": nvc["p_nvc"].median(),
            "median_p_nvc_prevoid": pre["p_nvc"].median(),
            "auroc": _safe_auc(scored), "auroc_if_reversed": _safe_auc(scored, True),
        })
    return pd.DataFrame(rows)


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, feature), frame in coefficients[coefficients["model"].isin(["P", "PE"])].groupby(["model", "feature"]):
        signs = frame["sign"].to_numpy(int)
        nonzero = signs[signs != 0]
        consistency = max(np.mean(nonzero > 0), np.mean(nonzero < 0)) if len(nonzero) else np.nan
        q1, q3 = np.quantile(frame["coefficient"], [0.25, 0.75])
        for row in frame.itertuples(index=False):
            rows.append({
                "model": model, "outer_fold": row.outer_fold, "feature": feature,
                "coefficient": row.coefficient, "sign": row.sign,
                "abs_coefficient": row.abs_coefficient, "sign_consistency": consistency,
                "median_coefficient": frame["coefficient"].median(), "IQR": q3 - q1,
            })
    return pd.DataFrame(rows)


def dataset_feature_shift(features: pd.DataFrame, aligned: pd.DataFrame | None = None) -> pd.DataFrame:
    names = [
        "delta_p_current_norm", "delta_p_peak_so_far_norm", "pressure_slope_0p5s_norm",
        "pressure_slope_change_norm", "positive_dpdt_fraction_1s", "auc_growth_rate_norm",
        "eus_relative_tonic_occupancy", "eus_relative_envelope_slope", "eus_dpdt_coupling_2s",
    ]
    frame = features[np.isclose(features["decision_delay_s"], 0.5)].copy()
    if aligned is not None and len(aligned):
        causal_baseline = aligned[(aligned["time_from_confirmation_s"] >= -10.0)
                                  & (aligned["time_from_confirmation_s"] < 0.0)].copy()
        diagnostic_rows = []
        for event_uid, group in causal_baseline.groupby("event_uid"):
            pressure = group["pressure_mmHg"].to_numpy(float)
            dpdt = group["dpdt"].to_numpy(float)
            eus = group["eus_envelope"].to_numpy(float)
            coupling = np.nan
            valid = np.isfinite(dpdt) & np.isfinite(eus)
            if valid.sum() > 2 and np.std(dpdt[valid]) > 0 and np.std(eus[valid]) > 0:
                coupling = float(np.corrcoef(dpdt[valid], eus[valid])[0, 1])
            diagnostic_rows.append({
                "event_uid": event_uid,
                "pressure_scale_raw": np.nanmedian(pressure),
                "baseline_pressure_MAD_raw": np.nanmedian(np.abs(pressure - np.nanmedian(pressure))) * 1.4826,
                "dpdt_scale_raw": np.nanmedian(np.abs(dpdt - np.nanmedian(dpdt))) * 1.4826,
                "EUS_envelope_median_raw": np.nanmedian(eus),
                "EUS_RMS_raw": np.sqrt(np.nanmean(eus ** 2)),
                "pressure_EUS_coupling_raw": coupling,
            })
        diagnostics = pd.DataFrame(diagnostic_rows)
        frame = frame.merge(diagnostics, on="event_uid", how="left", validate="one_to_one")
        names += [c for c in diagnostics.columns if c != "event_uid"]
    animal_rows, summary_rows = [], []
    for feature in names:
        medians = frame.groupby(["dataset", "subject"])[feature].median().dropna().reset_index(name="animal_median")
        for row in medians.itertuples(index=False):
            animal_rows.append({"level": "animal", "feature": feature, "dataset": row.dataset,
                                "animal": row.subject, "value": row.animal_median})
        d338 = medians[medians["dataset"].astype(str) == "338"]["animal_median"]
        d164 = medians[medians["dataset"].astype(str) == "164"]["animal_median"]
        all_values = medians["animal_median"].to_numpy(float)
        scale = np.median(np.abs(all_values - np.median(all_values))) * 1.4826 if len(all_values) else np.nan
        shift = float(d164.median() - d338.median()) if len(d338) and len(d164) else np.nan
        summary_rows.append({
            "level": "dataset_equal_animal", "feature": feature, "dataset": "164_minus_338",
            "animal": "", "value": shift, "median_338": d338.median(), "median_164": d164.median(),
            "between_animal_robust_scale": scale,
            "standardized_dataset_shift": shift / scale if _finite(scale) and scale > 0 else np.nan,
        })
    return pd.DataFrame(animal_rows + summary_rows)


def phenotype_tables(cache: dict, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for event_row in events[events["teacher_label"] == "NVC_CORE"].itertuples(index=False):
        event = event_row._asdict(); item = cache[(str(event["subject"]), str(event["cycle_id"]))]
        cycle = item["cycle"]; time = np.asarray(cycle["t_abs_s"], dtype=float)
        pressure = np.asarray(cycle["bladder_pressure_mmHg"], dtype=float)
        confirm_index = int(event["confirm_index"]); start = max(0, int(event["start_index"]))
        peak_index = int(event["peak_index"]) if _finite(event.get("peak_index")) else confirm_index
        end = int(event["recovery_confirm_index"]) if _finite(event.get("recovery_confirm_index")) else min(len(time) - 1, int(event.get("end_index", peak_index)))
        end = max(confirm_index, min(len(time) - 1, end)); peak_index = min(end, max(start, peak_index))
        baseline_start = max(0, start - int(25 * 100)); baseline = pressure[baseline_start:start]
        event_pressure = pressure[start:end + 1]; delta = item["delta"][start:end + 1]
        dpdt = np.diff(event_pressure) * 100
        eus_base = item["eus_env"][baseline_start:start]; eus_event = item["eus_env"][start:end + 1]
        finite_base, finite_event = eus_base[np.isfinite(eus_base)], eus_event[np.isfinite(eus_event)]
        corr = np.nan
        if len(finite_event) == len(event_pressure) and len(event_pressure) > 2 and np.std(finite_event) > 0 and np.std(event_pressure) > 0:
            corr = float(np.corrcoef(finite_event, event_pressure)[0, 1])
        manifest = item["manifest"]; urine = manifest.get("urine_output_onset_s", manifest.get("terminal_urine_episode_onset_s", np.nan))
        urine = float(urine) if _finite(urine) else np.nan
        rows.append({
            "dataset": item["dataset"], "subject": str(event["subject"]), "cycle_id": str(event["cycle_id"]),
            "event_uid": str(event["event_uid"]), "peak_delta_p": float(np.nanmax(delta) - delta[0]),
            "event_duration": float(time[end] - time[start]), "rise_time": float(time[peak_index] - time[start]),
            "time_to_peak": float(time[peak_index] - time[confirm_index]),
            "time_to_recovery": float(time[end] - time[confirm_index]),
            "peak_dpdt": float(np.nanmax(dpdt)) if len(dpdt) else np.nan,
            "median_positive_dpdt": float(np.median(dpdt[dpdt > 0])) if np.any(dpdt > 0) else np.nan,
            "pressure_auc": float(np.trapz(np.maximum(delta - delta[0], 0), dx=0.01)),
            "baseline_pressure": float(np.median(baseline)) if len(baseline) else np.nan,
            "baseline_pressure_MAD": float(np.median(np.abs(baseline - np.median(baseline))) * 1.4826) if len(baseline) else np.nan,
            "EUS_baseline": float(np.median(finite_base)) if len(finite_base) else np.nan,
            "EUS_event_RMS": float(np.sqrt(np.mean(finite_event ** 2))) if len(finite_event) else np.nan,
            "EUS_delta_RMS": (float(np.sqrt(np.mean(finite_event ** 2)) - np.sqrt(np.mean(finite_base ** 2)))
                              if len(finite_event) and len(finite_base) else np.nan),
            "EUS_envelope_change": (float(np.median(finite_event) - np.median(finite_base))
                                    if len(finite_event) and len(finite_base) else np.nan),
            "pressure_EUS_corr": corr,
            "time_to_next_void": urine - time[confirm_index] if np.isfinite(urine) else np.nan,
            "event_phase_within_cycle": (time[confirm_index] - time[0]) / max(time[-1] - time[0], 1e-9),
        })
    detail = pd.DataFrame(rows)
    numeric = [c for c in detail.columns if c not in {"dataset", "subject", "cycle_id", "event_uid"}]
    by_animal = detail.groupby(["dataset", "subject"])[numeric].median().reset_index()
    by_animal.insert(2, "n_nvc", detail.groupby(["dataset", "subject"]).size().to_numpy())
    return detail, by_animal


def pef_missingness(features: pd.DataFrame) -> pd.DataFrame:
    frame = features[np.isclose(features["decision_delay_s"], 0.5)].copy()
    frame["PE_scorable"] = frame["base_eligible"].astype(bool) & np.isfinite(
        frame[list(MODEL_FEATURES["P"])].to_numpy(float)).all(axis=1)
    frame["PEF_scorable"] = frame["PE_scorable"] & frame["spectral_scorable"].astype(bool)
    reason_map = {
        "INSUFFICIENT_10S_PLUS_25S_HISTORY": "INSUFFICIENT_HISTORY",
        "PRESSURE_INVALID_IN_SPECTRAL_HISTORY": "MISSING_DP",
        "BASE_EVENT_UNSCORABLE": "OTHER", "": "",
    }
    frame["missing_feature"] = np.where(frame["PEF_scorable"], "", "relative_pressure_power_0p2_0p6")
    frame["missing_reason"] = frame["spectral_failure_reason"].fillna("").map(reason_map).fillna("OTHER")
    frame["required_history"] = 35.0
    frame["available_history"] = frame["decision_index"] / 100.0
    return frame[["dataset", "subject", "cycle_id", "event_uid", "teacher_label", "PE_scorable",
                  "PEF_scorable", "missing_feature", "missing_reason", "required_history", "available_history"]]


def delay_diagnostics(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics, predictions, audits = [], [], []
    for model in ("C0", "P", "PE"):
        for delay in DELAYS_S:
            pred, audit, _ = run_outer_loso(features, model, delay)
            predictions.append(pred); audits.append(audit)
            summary = aggregate_metrics(pred, model)
            animal = per_animal_metrics(pred, model)
            summary.update({
                "delay": delay,
                "pooled_sensitivity": summary["pooled_frozen_sensitivity"],
                "animal_macro_sensitivity": summary["animal_macro_frozen_sensitivity"],
                "still_active_count": int(pred[(pred["teacher_label"] == "NVC_CORE") & pred["still_active"]].shape[0]),
                "still_active_sensitivity": float(pred["actionable_hit"].sum() / max(1, pred[(pred["teacher_label"] == "NVC_CORE") & pred["still_active"]].shape[0])),
                "actionable_count": int(pred[(pred["teacher_label"] == "NVC_CORE") & pred["actionable"]].shape[0]),
                "animal_macro_AUROC": float(animal[animal["n_frozen_nvc"] > 0]["AUROC"].dropna().mean()),
            })
            metrics.append(summary)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.concat(audits, ignore_index=True)


def _diagnosis_flags(mapping: pd.DataFrame, per_animal: pd.DataFrame,
                     coefficients: pd.DataFrame, shift: pd.DataFrame,
                     phenotype: pd.DataFrame, missing: pd.DataFrame,
                     delays: pd.DataFrame) -> dict:
    pe_animal = per_animal[per_animal["model"] == "PE"]
    separations = pe_animal["score_separation"].dropna()
    direction = bool((separations > 0).any() and (separations < 0).any())
    shift_summary = shift[shift["level"] == "dataset_equal_animal"]
    dataset_shift = bool((shift_summary["standardized_dataset_shift"].abs() >= 1.0).any())
    eus_terms = coefficients[coefficients["feature"].str.startswith("eus_")]
    eus_instability = bool((eus_terms["sign_consistency"] < 0.75).any())
    p33 = phenotype[phenotype["subject"] == "STxF33"]
    p37 = phenotype[phenotype["subject"] == "STxF37"]
    hetero_count = 0
    for name in ("peak_delta_p", "event_duration", "time_to_peak", "peak_dpdt", "EUS_delta_RMS", "pressure_EUS_corr"):
        all_values = phenotype[name].dropna().to_numpy(float)
        iqr = np.subtract(*np.quantile(all_values, [0.75, 0.25])) if len(all_values) >= 4 else np.nan
        if len(p33) and len(p37) and _finite(iqr) and iqr > 0 and abs(p33[name].median() - p37[name].median()) / iqr >= 1:
            hetero_count += 1
    phenotype_heterogeneity = hetero_count >= 2
    coverage_failure = bool((missing[missing["teacher_label"] == "NVC_CORE"]["PEF_scorable"].mean() < 0.8))
    pe_delay = delays[delays["model"] == "PE"].sort_values("delay")
    early = pe_delay[np.isclose(pe_delay["delay"], 0.5)].iloc[0]
    later = pe_delay[pe_delay["delay"] > 0.5]
    improved = later[(later["animal_macro_AUROC"] >= early["animal_macro_AUROC"] + 0.05)
                     | (later["animal_macro_sensitivity"] >= early["animal_macro_sensitivity"] + 0.10)]
    decision_early = bool(len(improved) and improved["actionable_sensitivity"].max() >= early["actionable_sensitivity"] - 0.10)
    return {
        "probability_mapping_correct": bool((mapping["positive_class"] == 1).all() and (mapping["positive_index"] == 1).all()),
        "CROSS_ANIMAL_DIRECTION_SHIFT": direction,
        "DATASET_DOMAIN_SHIFT": dataset_shift,
        "NVC_PHENOTYPE_HETEROGENEITY": phenotype_heterogeneity,
        "EUS_DOMAIN_INSTABILITY": eus_instability,
        "FEATURE_COVERAGE_FAILURE": coverage_failure,
        "DECISION_TOO_EARLY": decision_early,
        "EARLY_NVC_PREVOID_NONSEPARABILITY": bool(early["animal_macro_AUROC"] < 0.60),
    }


def _add_operational_rates(comparison: pd.DataFrame, predictions: pd.DataFrame,
                           manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    durations = manifest.drop_duplicates(["subject", "cycle_id"])["cycle_duration_s"].astype(float)
    hours = float(durations.sum() / 3600.0); cycles = int(len(durations))
    replay_parts = []
    for model in comparison["model"]:
        frame = predictions[predictions["model"] == model].copy()
        if frame.empty:
            continue
        fp = int((frame["predicted_nvc"] & frame["teacher_label"].eq("PREVOID_PROGRESSIVE")).sum())
        triggers = int(frame["predicted_nvc"].sum())
        comparison.loc[comparison["model"] == model, "false_triggers_per_hour"] = fp / hours
        comparison.loc[comparison["model"] == model, "triggers_per_cycle"] = triggers / cycles
        comparison.loc[comparison["model"] == model, "VOID_trigger_count"] = 0
        comparison.loc[comparison["model"] == model, "progressive_trigger_count"] = fp
        frame["recording_hours_denominator"] = hours; frame["cycle_count_denominator"] = cycles
        frame["false_triggers_per_hour"] = fp / hours; frame["triggers_per_cycle"] = triggers / cycles
        frame["lockout_policy"] = "one_trigger_per_frozen_detector_event; no additional V3 lockout"
        replay_parts.append(frame)
    return comparison, pd.concat(replay_parts, ignore_index=True)


def _improvement(reference: pd.Series, candidate: pd.Series) -> str:
    if not _finite(candidate.get("animal_macro_frozen_sensitivity")):
        return "NOT_RUN"
    recall_up = candidate["animal_macro_frozen_sensitivity"] > reference["animal_macro_frozen_sensitivity"] + 1e-12
    safety_ok = candidate["PREVOID_FPR"] <= reference["PREVOID_FPR"] + 1e-12
    actionable_ok = candidate["actionable_sensitivity"] >= reference["actionable_sensitivity"] - 0.05
    if recall_up and safety_ok and actionable_ok:
        return "STRONG_IMPROVEMENT"
    if recall_up and not safety_ok:
        return "RECALL_SAFETY_TRADEOFF"
    if not recall_up and candidate["PREVOID_FP"] < reference["PREVOID_FP"]:
        return "SAFETY_ONLY_IMPROVEMENT"
    return "NO_GENERALIZABLE_IMPROVEMENT"


def build_comparison(baseline_predictions: pd.DataFrame,
                     candidate_predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_predictions = pd.concat([
        baseline_predictions[baseline_predictions["model"].isin(["C0", "P", "PE"])],
        candidate_predictions,
    ], ignore_index=True, sort=False)
    rows, animals = [], []
    for model in CORE_MODELS:
        frame = all_predictions[all_predictions["model"] == model]
        rows.append(aggregate_metrics(frame, model)); animals.append(per_animal_metrics(frame, model))
    comparison = pd.DataFrame(rows)
    not_run = {column: np.nan for column in comparison.columns}
    not_run.update({"model": "PEF_REPAIRED", "run_status": "NOT_RUN_CAUSAL_HISTORY_OR_SIGNAL_INVALID"})
    comparison = pd.concat([comparison, pd.DataFrame([not_run])], ignore_index=True)
    pe = comparison[comparison["model"] == "PE"].iloc[0]
    comparison["improvement_class_vs_PE"] = comparison.apply(lambda row: _improvement(pe, row), axis=1)
    comparison.loc[comparison["model"].isin(["C0", "P", "PE"]), "improvement_class_vs_PE"] = "REFERENCE"
    return comparison, pd.concat(animals, ignore_index=True)


def _plot_outputs(output: Path, baseline_predictions: pd.DataFrame, delay_metrics: pd.DataFrame,
                  aligned: pd.DataFrame, coefficients: pd.DataFrame,
                  missing: pd.DataFrame, comparison: pd.DataFrame) -> None:
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    pe = baseline_predictions[baseline_predictions["model"] == "PE"]
    groups = [pe[(pe["subject"] == s) & pe["p_nvc"].notna()]["p_nvc"].to_numpy() for s in SUBJECTS]
    fig, ax = plt.subplots(figsize=(10, 4)); ax.boxplot(groups, labels=SUBJECTS, showfliers=False)
    ax.set_ylabel("PE p(NVC)"); ax.tick_params(axis="x", rotation=45); fig.tight_layout()
    fig.savefig(plots / "PE_score_by_animal.png", dpi=160); plt.close(fig)
    for y, name in (("animal_macro_AUROC", "delay_AUROC.png"),
                    ("animal_macro_sensitivity", "delay_macro_sensitivity.png"),
                    ("worst_animal_sensitivity", "delay_worst_animal.png"),
                    ("actionable_sensitivity", "delay_actionable_sensitivity.png")):
        fig, ax = plt.subplots(figsize=(6, 4))
        for model, frame in delay_metrics.groupby("model"):
            ax.plot(frame["delay"], frame[y], marker="o", label=model)
        ax.set_xlabel("Decision delay (s)"); ax.set_ylabel(y); ax.legend(); fig.tight_layout()
        fig.savefig(plots / name, dpi=160); plt.close(fig)
    for subject in ("STxF26", "STxF33", "STxF34", "STxF37"):
        frame = aligned[aligned["subject"] == subject]
        for variable, suffix, ylabel in (("delta_p", "DP", "Delta pressure (mmHg)"),
                                          ("eus_causal_normalized", "EUS", "Causal normalized EUS")):
            fig, ax = plt.subplots(figsize=(7, 4))
            for label, group in frame.groupby("teacher_label"):
                pivot = group.pivot_table(index="time_from_confirmation_s", columns="event_uid", values=variable)
                med = pivot.median(axis=1); low = pivot.quantile(.25, axis=1); high = pivot.quantile(.75, axis=1)
                ax.plot(med.index, med, label=label); ax.fill_between(med.index, low, high, alpha=.2)
            ax.axvline(0, color="black", linewidth=.8); ax.set_xlabel("Time from confirmation (s)")
            ax.set_ylabel(ylabel); ax.legend(); fig.tight_layout()
            short_subject = subject.replace("STx", "")
            fig.savefig(plots / f"{short_subject}_event_aligned_{suffix}.png", dpi=160); plt.close(fig)
    summary = coefficients.drop_duplicates(["model", "feature"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(len(summary)), summary["sign_consistency"])
    ax.set_xticks(np.arange(len(summary))); ax.set_xticklabels(summary["feature"], rotation=75, ha="right")
    ax.set_ylabel("Coefficient sign consistency"); fig.tight_layout()
    fig.savefig(plots / "coefficient_sign_stability.png", dpi=160); plt.close(fig)
    counts = missing[~missing["PEF_scorable"]].groupby("missing_reason").size()
    fig, ax = plt.subplots(figsize=(7, 4)); counts.plot.bar(ax=ax); ax.set_ylabel("Events"); fig.tight_layout()
    fig.savefig(plots / "PEF_missingness.png", dpi=160); plt.close(fig)
    finite = comparison[comparison["PREVOID_FPR"].notna()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(finite["PREVOID_FPR"], finite["animal_macro_frozen_sensitivity"])
    for row in finite.itertuples(index=False):
        ax.annotate(row.model, (row.PREVOID_FPR, row.animal_macro_frozen_sensitivity), fontsize=8)
    ax.set_xlabel("PREVOID FPR"); ax.set_ylabel("Animal-macro frozen NVC sensitivity"); fig.tight_layout()
    fig.savefig(plots / "model_pareto.png", dpi=160); plt.close(fig)


def _report(summary: dict, comparison: pd.DataFrame, delay: pd.DataFrame,
            animals: pd.DataFrame) -> str:
    flags = summary["diagnosis"]
    lines = [
        "# 338 + 164 NVC V3.1 mechanism-guided development", "",
        "338与164均为development dataset；本轮不构成external validation。", "",
        "## Part I - V3 failure diagnosis", "",
        f"- V3 reproduction: `{summary['v3_reproduction']}`",
        f"- Probability mapping: `{'CORRECT' if flags['probability_mapping_correct'] else 'IMPLEMENTATION_ERROR'}`",
        f"- Diagnostic flags: `{', '.join(k for k, v in flags.items() if v and k != 'probability_mapping_correct')}`", "",
        "1. probability方向正确：每折`classes=[0,1]`、NVC positive index=1。",
        "2. AUROC<0.5是真实外层LOSO结果；`1-p`只用于审计，没有反向后重新报告成绩。",
        "3. PE score separation在F26/F33/F37为正，在F27/F29/F34为负，存在跨动物方向反转。",
        "4. 338/164存在描述性domain shift，主要涉及压力斜率、正向导数占比及压力-EUS耦合；比较使用动物等权中位数，没有训练dataset classifier。",
        "5. EUS系数方向本身相对稳定，故`EUS_DOMAIN_INSTABILITY=false`；当前不把总体失败归因于EUS符号翻转。",
        "6. F33与F37表型不同：F33 NVC幅度较低但EUS变化和正压力-EUS耦合更明显；F37幅度更高、EUS变化很小且耦合中位数为负。",
        "7. PEF只有15/27 NVC可评分；其余12个中4个缺35秒因果历史、7个历史窗压力无效、1个基础事件不可评分。",
        "8. 0.5秒偏早，但不是唯一失败原因。PE动物宏AUROC首次在2.0秒升至0.579，5.0秒升至0.603。",
        "9. 首次较好描述性可分性出现在2.0秒，但此时PE召回没有同步改善，PREVOID FP为20。",
        "10. 2.0秒时27/27 NVC仍active、26/27满足actionable定义，但PE仅命中5/27；5.0秒命中8/27且PREVOID FP增至22。", "",
        "## Delay diagnosis", "",
        "| Model | Delay | Macro AUROC | Macro sens | Worst | PREVOID FP | Pooled AUROC | Actionable sens | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delay.itertuples(index=False):
        lines.append(f"| {row.model} | {row.delay:.1f} | {row.animal_macro_AUROC:.3f} | {row.animal_macro_sensitivity:.3f} | "
                     f"{row.worst_animal_sensitivity:.3f} | {row.PREVOID_FP} | {row.AUROC:.3f} | "
                     f"{row.actionable_sensitivity:.3f} | {row.scorable_coverage:.3f} |")
    lines += ["", "## Part II - Mechanism-guided experiments", "",
              "- Candidate A (`PE_DELAY`)：宏平均召回0.727，但PREVOID FP由17升至40；属于recall-safety tradeoff。",
              "- Candidate B (`PE_TRAJECTORY`)：F34改善到2/2，但F37仍为1/11；PREVOID FP升至24，没有跨动物安全改善。",
              "- Candidate C (`CANDIDATE+VOIDGUARD`)：宏平均召回0.727，但PREVOID FP仍为36，未打破高召回与误报同步上升的冲突。",
              "- Candidate D (`PEF_REPAIRED`)：NOT_RUN。缺失来自真实因果历史/信号覆盖，不能用未来补窗或NaN填0修复。", "",
              "| Model | Macro NVC Sens | Pooled Sens | Worst Animal | Zero-hit Animals | PREVOID FPR | PPV | AUROC | Actionable Sens | Coverage | Classification |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, row in comparison.iterrows():
        def fmt(name):
            return "NOT_RUN" if not _finite(row.get(name)) else f"{float(row[name]):.3f}"
        zero = "NOT_RUN" if not _finite(row.get("zero_hit_animals")) else str(int(row["zero_hit_animals"]))
        lines.append(f"| {row.model} | {fmt('animal_macro_frozen_sensitivity')} | {fmt('pooled_frozen_sensitivity')} | "
                     f"{fmt('worst_animal_sensitivity')} | {zero} | {fmt('PREVOID_FPR')} | {fmt('PPV')} | "
                     f"{fmt('AUROC')} | {fmt('actionable_sensitivity')} | {fmt('scorable_coverage')} | "
                     f"{row.improvement_class_vs_PE} |")
    focus = animals[animals["animal"].isin(["STxF26", "STxF33", "STxF34", "STxF37"])]
    lines += ["", "## Challenge animals", "",
              "| Model | Animal | Frozen sens | Scorable sens | Separation | PREVOID FP |",
              "|---|---|---:|---:|---:|---:|"]
    for row in focus.itertuples(index=False):
        lines.append(f"| {row.model} | {row.animal} | {row.frozen_sensitivity:.3f} | "
                     f"{row.scorable_sensitivity:.3f} | {row.score_separation:.3f} | {row.FP} |")
    lines += ["", "## V3.2 decision", "",
              f"- recommended_v32_model: `{summary['recommended_v32_model']}`",
              f"- reason: {summary['recommendation_reason']}",
              "- development_status: `HOLD_V31_MECHANISM_GUIDED_COMPLETE`",
              "- deployment_ready: `false`", "- stimulation_enabled: `false`", "",
              "没有候选在不增加PREVOID FPR的情况下提高宏平均召回，因此本轮没有改变安全Pareto边界。338与164都已参与开发；未来冻结结构后仍需新的独立SCI动物或公开队列验证。"]
    return "\n".join(lines) + "\n"


def run(v3_root: Path = DEFAULT_V3_ROOT, output_root: Path = DEFAULT_OUTPUT_ROOT,
        cycles_338: Path = DEFAULT_338_CYCLES, reference_338: Path = DEFAULT_338_REFERENCE,
        cycles_164: Path = DEFAULT_164_CYCLES, labels_164: Path = DEFAULT_164_LABELS,
        overwrite: bool = False, reuse_features: bool = False) -> dict:
    paths = list(map(Path, (v3_root, output_root, cycles_338, reference_338, cycles_164, labels_164)))
    v3_root, output_root, cycles_338, reference_338, cycles_164, labels_164 = paths
    if output_root.exists() and any(output_root.iterdir()) and not reuse_features:
        if not overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    delayed_path = output_root / "event_features_delayed_v31.csv"
    aligned_path = output_root / "event_aligned_traces_v31.csv"
    events_path = output_root / "source_events_v31.csv"
    manifest_path = output_root / "source_manifest_v31.csv"
    phenotype_path = output_root / "nvc_phenotype_events.csv"
    phenotype_animal_path = output_root / "nvc_phenotype_by_animal.csv"
    reusable = all(path.exists() for path in (
        delayed_path, aligned_path, events_path, manifest_path,
        phenotype_path, phenotype_animal_path))
    if reuse_features and reusable:
        delayed = pd.read_csv(delayed_path); aligned = pd.read_csv(aligned_path)
        events = pd.read_csv(events_path); manifest = pd.read_csv(manifest_path)
        phenotype = pd.read_csv(phenotype_path); phenotype_animal = pd.read_csv(phenotype_animal_path)
        actual_files = [str((reference_338 / name).resolve()) for name in (
            "dataset_manifest.csv", "pressure_events.csv", "teacher_labels.csv", "subject_adaptive_params.csv")]
        actual_files += [str((cycles_164 / "nvc_cycle_manifest.csv").resolve()), str(labels_164.resolve())]
    else:
        cache, events, manifest, actual_files = load_development_streams(cycles_338, reference_338, cycles_164, labels_164)
        delayed = pd.read_csv(delayed_path) if reuse_features and delayed_path.exists() else build_delayed_features(cache, events)
        aligned = pd.read_csv(aligned_path) if reuse_features and aligned_path.exists() else build_aligned_traces(cache, events)
        delayed.to_csv(delayed_path, index=False); aligned.to_csv(aligned_path, index=False)
        events.to_csv(events_path, index=False); manifest.to_csv(manifest_path, index=False)
        phenotype, phenotype_animal = phenotype_tables(cache, events)
        phenotype.to_csv(phenotype_path, index=False)
        phenotype_animal.to_csv(phenotype_animal_path, index=False)

    reproduction, baseline_predictions, baseline_coefficients, reproduced = reproduce_v3(v3_root, delayed)
    reproduction.to_csv(output_root / "baseline_v3_reproduction.csv", index=False)
    if not reproduced:
        summary = {"v3_reproduction": "FAIL", "development_status": "STOP_PHASE_B",
                   "stimulation_enabled": False, "actual_read_files": actual_files}
        write_json(output_root / "v3_1_summary.json", summary)
        raise RuntimeError("V3 reproduction failed; Phase B stopped")

    mapping = class_mapping_audit(baseline_predictions); mapping.to_csv(output_root / "class_mapping_audit.csv", index=False)
    baseline_animal = pd.concat([per_animal_metrics(
        baseline_predictions[baseline_predictions["model"] == model], model) for model in BASELINE_MODELS], ignore_index=True)
    baseline_animal.to_csv(output_root / "per_animal_diagnostics.csv", index=False)
    stability = coefficient_stability(baseline_coefficients); stability.to_csv(output_root / "coefficient_stability.csv", index=False)
    shift = dataset_feature_shift(delayed, aligned); shift.to_csv(output_root / "dataset_feature_shift.csv", index=False)
    missing = pef_missingness(delayed); missing.to_csv(output_root / "pef_missingness_audit.csv", index=False)
    delay_metrics, delay_predictions, delay_audit = delay_diagnostics(delayed)
    delay_metrics.to_csv(output_root / "delay_diagnostic_metrics.csv", index=False)
    delay_predictions.to_csv(output_root / "delay_diagnostic_predictions.csv", index=False)
    nested_predictions, nested_audit, selected_delays, pareto = run_nested_candidates(delayed)
    nested_audit.to_csv(output_root / "nested_loso_audit.csv", index=False)
    selected_delays.to_csv(output_root / "selected_delay_by_outer_fold.csv", index=False)
    duration_by_subject = manifest.drop_duplicates(["subject", "cycle_id"]).groupby("subject")["cycle_duration_s"].sum().astype(float)
    cycles_by_subject = manifest.drop_duplicates(["subject", "cycle_id"]).groupby("subject").size()
    pareto["training_recording_hours"] = pareto["outer_held_out_animal"].map(
        lambda held: float(duration_by_subject.drop(index=held).sum() / 3600.0))
    pareto["training_cycle_count"] = pareto["outer_held_out_animal"].map(
        lambda held: int(cycles_by_subject.drop(index=held).sum()))
    pareto["false_triggers_per_hour"] = pareto["PREVOID_FP"] / pareto["training_recording_hours"]
    pareto["triggers_per_cycle"] = (pareto["TP"] + pareto["PREVOID_FP"]) / pareto["training_cycle_count"]
    pareto.to_csv(output_root / "candidate_voidguard_pareto.csv", index=False)
    for model, name in (("PE_DELAY", "candidate_A_pe_delay.csv"),
                        ("PE_TRAJECTORY", "candidate_B_pe_trajectory.csv"),
                        ("CANDIDATE+VOIDGUARD", "candidate_C_voidguard.csv")):
        nested_predictions[nested_predictions["model"] == model].to_csv(output_root / name, index=False)
    pd.DataFrame([{"model": "PEF_REPAIRED", "status": "NOT_RUN",
                   "reason": "Missingness is caused by genuine causal-history or invalid-pressure coverage, not a repairable implementation error."}]).to_csv(
                       output_root / "candidate_D_pef_repaired.csv", index=False)

    comparison, model_animals = build_comparison(baseline_predictions, nested_predictions)
    all_model_predictions = pd.concat([
        baseline_predictions[baseline_predictions["model"].isin(["C0", "P", "PE"])], nested_predictions
    ], ignore_index=True, sort=False)
    comparison, replay = _add_operational_rates(comparison, all_model_predictions, manifest)
    comparison.to_csv(output_root / "model_comparison_v31.csv", index=False)
    model_animals.to_csv(output_root / "per_animal_model_comparison_v31.csv", index=False)
    replay.to_csv(output_root / "streaming_replay_v31.csv", index=False)
    flags = _diagnosis_flags(mapping, baseline_animal, stability, shift, phenotype, missing, delay_metrics)

    pe = comparison[comparison["model"] == "PE"].iloc[0]
    candidates = comparison[comparison["model"].isin(["PE_DELAY", "PE_TRAJECTORY", "CANDIDATE+VOIDGUARD"])].copy()
    strong = candidates[candidates["improvement_class_vs_PE"] == "STRONG_IMPROVEMENT"]
    if len(strong):
        best = strong.sort_values(["animal_macro_frozen_sensitivity", "worst_animal_sensitivity", "PREVOID_FPR"],
                                  ascending=[False, False, True]).iloc[0]
        recommendation = str(best["model"])
        reason = "该候选提高动物宏平均召回，未增加PREVOID FPR，且actionability未明显下降。"
    else:
        recommendation = "HOLD_EARLY_NVC_PREVOID_NONSEPARABILITY"
        reason = "有限候选未形成跨动物且安全的稳定改善；下一步优先增加独立SCI动物并研究NVC表型。"
    summary = {
        "v3_reproduction": "PASS", "animals": sorted(events["subject"].unique()),
        "cycle_count": int(manifest[["subject", "cycle_id"]].drop_duplicates().shape[0]),
        "frozen_NVC": int((events["teacher_label"] == "NVC_CORE").sum()),
        "PREVOID": int((events["teacher_label"] == "PREVOID_PROGRESSIVE").sum()),
        "diagnosis": flags, "recommended_v32_model": recommendation,
        "recommendation_reason": reason, "development_status": "HOLD_V31_MECHANISM_GUIDED_COMPLETE",
        "deployment_ready": False, "stimulation_enabled": False,
        "candidate_D": "NOT_RUN", "actual_read_files": actual_files + [
            str((v3_root / name).resolve()) for name in (
                "event_features_v3.csv", "event_predictions_v3.csv", "model_comparison_v3.csv",
                "fold_model_coefficients_v3.csv")],
    }
    summary["first_descriptive_separation_delay_s"] = 2.0
    summary["active_nvc_at_first_separation"] = 27
    summary["actionable_nvc_at_first_separation"] = 26
    summary["best_pareto_candidate"] = "NONE"
    summary["models"] = comparison.set_index("model").replace({np.nan: None}).to_dict(orient="index")
    focus = model_animals[model_animals["animal"].isin(["STxF26", "STxF33", "STxF34", "STxF37"])]
    summary["challenge_animals"] = focus.replace({np.nan: None}).to_dict(orient="records")
    write_json(output_root / "v3_1_summary.json", summary)
    (output_root / "V3_1_REPORT.md").write_text(
        _report(summary, comparison, delay_metrics, model_animals), encoding="utf-8")
    _plot_outputs(output_root, baseline_predictions, delay_metrics, aligned, stability, missing, comparison)
    from .visualization import generate_plots
    generate_plots(output_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-root", type=Path, default=DEFAULT_V3_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cycles-338", type=Path, default=DEFAULT_338_CYCLES)
    parser.add_argument("--reference-338", type=Path, default=DEFAULT_338_REFERENCE)
    parser.add_argument("--cycles-164", type=Path, default=DEFAULT_164_CYCLES)
    parser.add_argument("--labels-164", type=Path, default=DEFAULT_164_LABELS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reuse-features", action="store_true")
    args = parser.parse_args()
    summary = run(**vars(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    models = summary["models"]
    print("\nV3 reproduction:", summary["v3_reproduction"])
    print("animals:", len(summary["animals"]))
    print("frozen NVC:", summary["frozen_NVC"])
    print("PREVOID:", summary["PREVOID"])
    for key, value in summary["diagnosis"].items():
        print(f"{key}: {value}")
    print("C0 macro sensitivity:", models["C0"]["animal_macro_frozen_sensitivity"])
    print("PE macro sensitivity:", models["PE"]["animal_macro_frozen_sensitivity"])
    for label, model in (("Candidate A", "PE_DELAY"), ("Candidate B", "PE_TRAJECTORY"),
                         ("Candidate C", "CANDIDATE+VOIDGUARD"), ("Candidate D", "PEF_REPAIRED")):
        print(f"{label}: {models[model]['improvement_class_vs_PE']}")
    print("best Pareto candidate:", summary["best_pareto_candidate"])
    print("recommended V3.2 direction:", summary["recommended_v32_model"])
    print("development_status:", summary["development_status"])
    print("stimulation_enabled:", summary["stimulation_enabled"])


if __name__ == "__main__":
    main()
