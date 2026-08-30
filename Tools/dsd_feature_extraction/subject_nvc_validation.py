"""Frozen-rule NVC validation for one subject.

The validator consumes the common stable-cycle contract. Pressure detection,
offline Volume association, causal features, and the exploratory score are
kept separate: Volume is used only by the frozen teacher-label function and
is never included in the eight causal model features.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data_io import load_cycle, write_json
from .detectors import AdaptiveHistory, adaptive_local_pressure_events, associate_adaptive_labels
from .features import causal_eus_envelope_100hz, decision_feature_at_index


def _times(event, t):
    def get(name):
        i = event.get(name)
        return float(t[int(i)]) if i is not None and 0 <= int(i) < len(t) else np.nan
    return {
        "candidate_start_s": get("start_index"),
        "confirm_time_s": get("confirm_index"),
        "local_peak_time_s": get("peak_index"),
        "recovery_start_s": get("recovery_start_index"),
        "recovery_confirm_s": get("recovery_confirm_index"),
        "end_s": get("end_index"),
    }


def _model_score(causal: pd.DataFrame, labels: pd.DataFrame, model: dict) -> pd.DataFrame:
    columns = ["event_uid", "cycle_id", "confirm_time_s", "automatic_teacher_label"]
    if causal.empty:
        return pd.DataFrame(columns=columns + ["p_void_risk", "p_nvc", "offline_analysis_call"])
    label_cols = [c for c in ("event_uid", "automatic_teacher_label") if c in labels.columns]
    out = causal.merge(labels[label_cols].drop_duplicates("event_uid"), on="event_uid", how="left")
    order = list(model["feature_order"])
    x = out.reindex(columns=order).to_numpy(float)
    ok = np.isfinite(x).all(axis=1)
    risk = np.full(len(out), np.nan)
    z = (x[ok] - np.asarray(model["scaler_mean"], float)) / np.asarray(model["scaler_scale"], float)
    coef = np.asarray(model["coefficient_void_risk"], float)
    intercept = float(model.get("intercept", model.get("intercept_void_risk", 0.0)))
    risk[ok] = 1.0 / (1.0 + np.exp(-(intercept + z @ coef)))
    out["p_void_risk"] = risk
    out["p_nvc"] = 1.0 - risk
    cutoff = float(model.get("analysis_threshold", 0.5))
    out["offline_analysis_call"] = np.where(np.isfinite(out.p_nvc), out.p_nvc >= cutoff, False)
    out["model_name"] = str(model.get("selected_model", "M1"))
    out["model_status"] = "HOLD_NO_CAUSAL_SEPARATION"
    out["score_interpretation"] = "EXPLORATORY_EXTERNAL_SCORE_ONLY"
    out["higher_score_means_more_nvc"] = True
    return out


def _frozen_population_priors(reference_root: Path | None) -> tuple[float, float, str]:
    """Return Stage-A population warm-up priors, never F30 estimates."""
    if reference_root is None:
        raise RuntimeError("Stage-A reference root is required for frozen validation")
    path = Path(reference_root) / "subject_adaptive_params.csv"
    table = pd.read_csv(path)
    sigma_col = "warmup_prior_sigma_p" if "warmup_prior_sigma_p" in table else "sigma_p_median"
    sigma = pd.to_numeric(table[sigma_col], errors="coerce").dropna()
    dpdt = pd.to_numeric(table["sigma_dpdt_median"], errors="coerce").dropna()
    if sigma.empty or dpdt.empty:
        raise RuntimeError(f"Frozen Stage-A priors unavailable in {path}")
    return float(sigma.median()), float(dpdt.median()), str(path.resolve())


def validate_subject_nvc(subject: str, cycles_root: Path, results_root: Path,
                         frozen_reference_root: Path | None = None,
                         output_prefix: str = "") -> dict:
    """Run the frozen detector/teacher/model path for one subject.

    ``output_prefix`` controls artifact names only; it never selects thresholds,
    labels, or a subject-specific detection branch.
    """
    cycles_root, results_root = Path(cycles_root), Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    name = lambda stem: results_root / f"{output_prefix}{stem}"

    manifest_path = (cycles_root / "nvc_cycle_manifest.csv"
                     if (cycles_root / "nvc_cycle_manifest.csv").exists()
                     else cycles_root / "cycle_manifest.csv")
    manifest = pd.read_csv(manifest_path).sort_values("cycle_start_s").reset_index(drop=True)
    manifest = manifest[manifest.subject == subject].copy()
    if manifest.empty:
        raise RuntimeError(f"No cycles for subject {subject}")

    # Read global Volume evidence only. No per-cycle Volume detection is done.
    urine_path = cycles_root / "all_volume_events.csv"
    urine = pd.read_csv(urine_path).fillna("") if urine_path.exists() else pd.DataFrame()
    subject_urine = urine[urine.subject.astype(str) == str(subject)].copy() if "subject" in urine.columns else urine.copy()
    # Never expose another subject's urine events to association.  This also
    # prevents per-subject validation artifacts from being populated with the
    # first animal's global event table.
    urine_rows = subject_urine.to_dict("records")
    prior_sigma, prior_dpdt, prior_source = _frozen_population_priors(frozen_reference_root)

    history = AdaptiveHistory()
    pressure_rows, feature_rows, causal_rows, label_rows = [], [], [], []
    for _, row in manifest.iterrows():
        load_row = row.copy()
        load_row["dsd_cycle_id"] = row["cycle_id"]
        cycle = load_cycle(cycles_root, load_row)
        # Exact Stage-A detector with Stage-A population priors. The F30
        # exploratory sigma_p record is not used for formal detection.
        delta, events, adaptive = adaptive_local_pressure_events(cycle, history, prior_sigma, prior_dpdt)
        eus, eus_valid = causal_eus_envelope_100hz(cycle)
        cid = str(row["cycle_id"])
        t = np.asarray(cycle["t_abs_s"])
        labels, terminal_ok, _ = associate_adaptive_labels(subject, cid, events, urine_rows, t)
        for j, (event, label_info) in enumerate(zip(events, labels), 1):
            label, matched, reason = label_info
            uid = f"{subject}::{cid}::L{j:03d}"
            tm = _times(event, t)
            peak_i = event.get("peak_index")
            peak_delta = float(delta[int(peak_i)]) if peak_i is not None and np.isfinite(delta[int(peak_i)]) else np.nan
            base = {
                "subject": subject, "cycle_id": cid, "event_id": f"{cid}_L{j:03d}",
                "event_uid": uid, **tm,
                "peak_delta_p_mmHg": peak_delta,
                "local_prominence_mmHg": event.get("local_prominence_mmHg", np.nan),
                "fall_from_peak_mmHg": event.get("fall_from_peak_mmHg", np.nan),
                "recovery_fraction": event.get("recovery_fraction", np.nan),
                "detection_level": event.get("detection_level", ""),
                "recovered": bool(event.get("recovered", False)),
                "data_invalid": bool(event.get("data_invalid", False)),
                "automatic_teacher_label": label, "teacher_label": label,
                "label_reason": reason,
                "matched_urine_event_id": matched.get("urine_event_id", "") if matched else "",
                "terminal_volume_matched": bool(terminal_ok),
                "teacher_label_source": "STAGE_A_FROZEN_PRESSURE_PLUS_GLOBAL_VOLUME",
                "teacher_label_quality": "RULE_BASED_SILVER_STANDARD",
                "manual_review_used": False,
            }
            pressure_rows.append(base)
            label_rows.append(base)
            if event.get("confirm_index") is not None:
                feat = decision_feature_at_index(
                    cycle, delta, eus, eus_valid, adaptive,
                    int(event["confirm_index"]), event,
                )
                if feat is not None:
                    causal_rows.append({
                        "event_uid": uid, "cycle_id": cid,
                        "confirm_time_s": tm["confirm_time_s"],
                        **{k: feat[k] for k in C.PRESSURE_FEATURES},
                    })
            rec, start, conf, peak = (tm["recovery_confirm_s"], tm["candidate_start_s"],
                                      tm["confirm_time_s"], tm["local_peak_time_s"])
            auc = np.nan
            if np.isfinite(rec) and np.isfinite(start):
                a, b = int(event["start_index"]), int(event["recovery_confirm_index"])
                auc = float(np.trapz(np.maximum(delta[a:b + 1], 0), dx=1 / C.DP_FS_HZ))
            feature_rows.append({
                "event_uid": uid, "subject": subject, "cycle_id": cid,
                "event_id": f"{cid}_L{j:03d}", **tm,
                "automatic_teacher_label": label,
                "local_prominence_mmHg": event.get("local_prominence_mmHg", np.nan),
                "peak_delta_p_mmHg": peak_delta, "pressure_auc": auc,
                "recovery_fraction": event.get("recovery_fraction", np.nan),
                "fall_from_peak_mmHg": event.get("fall_from_peak_mmHg", np.nan),
                "candidate_to_recovery_s": rec - start if np.isfinite(rec) and np.isfinite(start) else np.nan,
                "confirm_to_recovery_s": rec - conf if np.isfinite(rec) and np.isfinite(conf) else np.nan,
                "rise_to_peak_s": peak - start if np.isfinite(peak) and np.isfinite(start) else np.nan,
                "peak_to_recovery_s": rec - peak if np.isfinite(rec) and np.isfinite(peak) else np.nan,
                "duration_outlier_flag": bool(np.isfinite(rec - start) and rec - start > 30)
                if np.isfinite(rec) and np.isfinite(start) else False,
            })

    pressure = pd.DataFrame(pressure_rows)
    labels = pd.DataFrame(label_rows)
    features = pd.DataFrame(feature_rows)
    causal = pd.DataFrame(causal_rows)
    pressure.to_csv(name("pressure_events.csv"), index=False)
    labels.to_csv(name("teacher_labels.csv"), index=False)
    features.to_csv(name("event_features.csv"), index=False)
    causal.to_csv(name("causal_features.csv"), index=False)
    features[features.automatic_teacher_label == "NVC_CORE"].to_csv(name("nvc_events.csv"), index=False)
    subject_urine.to_csv(name("urine_events.csv"), index=False)

    model = {}
    if frozen_reference_root is not None and (Path(frozen_reference_root) / "final_model.json").exists():
        model = json.loads((Path(frozen_reference_root) / "final_model.json").read_text(encoding="utf-8"))
    model_ok = bool(model.get("feature_order") and model.get("scaler_mean") and model.get("scaler_scale") and model.get("coefficient_void_risk"))
    scores = _model_score(causal, labels, model) if model_ok else pd.DataFrame()
    scores.to_csv(name("frozen_model_scores.csv"), index=False)
    calls = scores[scores.offline_analysis_call.astype(bool)] if not scores.empty else pd.DataFrame()
    nvc = features[features.automatic_teacher_label == "NVC_CORE"]
    call_labels = calls["automatic_teacher_label"] if "automatic_teacher_label" in calls.columns else pd.Series(dtype=object)
    accepted = int((call_labels == "NVC_CORE").sum())
    metrics = {
        "nvc_event_sensitivity": accepted / len(nvc) if len(nvc) else None,
        "nvc_ppv": accepted / len(calls) if len(nvc) and len(calls) else None,
        "prevoid_false_calls": int((call_labels == "PREVOID_PROGRESSIVE").sum()),
        "void_false_calls": int((call_labels == "VOID_CONFIRMED").sum()),
        "calls_per_cycle": len(calls) / len(manifest),
        "model_status": "HOLD_NO_CAUSAL_SEPARATION" if model_ok else "MODEL_BUNDLE_INCOMPLETE",
    }
    pd.DataFrame([metrics]).to_csv(name("frozen_model_metrics.csv"), index=False)

    counts = {k: int((features.automatic_teacher_label == k).sum())
              for k in ("NVC_CORE", "PREVOID_PROGRESSIVE", "GREY_ZONE", "INVALID")}
    hours = float(manifest.cycle_duration_s.sum() / 3600)
    final_status = "F30_VALIDATION_COMPLETE_NO_AUTOMATIC_NVC" if counts["NVC_CORE"] == 0 else "F30_VALIDATION_COMPLETE"
    summary = {
        "subject": subject, "external_validation_only": True,
        "cycle_count": int(len(manifest)), "valid_cycle_count": int(len(manifest)),
        "automatic_label_counts": counts,
        "nvc_cycle_count": int(nvc.cycle_id.nunique()),
        "nvc_frequency_per_h": float(len(nvc) / hours) if hours else 0.0,
        "valid_analysis_hours": hours, "pressure_event_count": int(len(pressure)),
        "pressure_event_count_by_cycle": pressure.groupby("cycle_id").size().astype(int).to_dict() if len(pressure) else {},
        "global_volume_event_count": int(len(subject_urine)),
        "assigned_volume_event_count": int(subject_urine.cycle_id.astype(str).isin(manifest.cycle_id.astype(str)).sum()) if len(subject_urine) else 0,
        "frozen_model_available": model_ok, "frozen_model_metrics": metrics,
        "feature_reference_status": "FROZEN_STAGE_A_REFERENCE",
        "pressure_detection_function": "adaptive_local_pressure_events",
        "teacher_label_function": "associate_adaptive_labels",
        "pressure_prior_source": prior_source, "f30_sigma_p_exploratory_only": True,
        "causal_feature_order": C.PRESSURE_FEATURES,
        "causal_features_exclude_volume_and_future": True,
        "manual_review_used": False, "stimulation_enabled": False,
        "positive_nvc_validation_status": "EVALUABLE" if counts["NVC_CORE"] else "NOT_EVALUABLE_NO_POSITIVE_EVENTS",
        "prevoid_negative_validation_status": "EVALUABLE",
        "causal_classifier_status": "HOLD_NO_CAUSAL_SEPARATION", "final_status": final_status,
    }
    write_json(name("external_validation_summary.json"), summary)
    pd.DataFrame([{
        "subject": subject, "valid_cycles": len(manifest), "valid_analysis_hours": hours,
        "n_nvc_core": counts["NVC_CORE"], "nvc_cycles": nvc.cycle_id.nunique(),
        "nvc_frequency_per_h": summary["nvc_frequency_per_h"],
        "n_prevoid_progressive": counts["PREVOID_PROGRESSIVE"],
        "n_grey_zone": counts["GREY_ZONE"], "n_invalid": counts["INVALID"],
        "prevoid_false_calls": metrics["prevoid_false_calls"],
    }]).to_csv(name("subject_summary.csv"), index=False)

    # Presentation only: copy the already-frozen cycle quicklooks.
    quicklook_root = results_root / "quicklooks"
    quicklook_root.mkdir(parents=True, exist_ok=True)
    for cid in manifest.cycle_id.astype(str):
        src = cycles_root / subject / cid / "quicklook.png"
        if src.exists():
            shutil.copy2(src, quicklook_root / f"{cid}.png")
    return summary
