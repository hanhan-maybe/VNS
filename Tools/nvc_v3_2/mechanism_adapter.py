"""Raw-cycle adapter for the combined 338 + 164 development cohort."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from Tools.dsd_feature_extraction import config as C
from Tools.dsd_feature_extraction.data_io import load_cycle
from Tools.dsd_feature_extraction.detectors import AdaptiveHistory, adaptive_local_pressure_events
from Tools.dsd_feature_extraction.features import causal_eus_envelope_100hz, decision_feature_at_index, robust_location_scale
from .version_support import _eus_features, _low_frequency_feature
from .version_support import C0_FEATURES, EUS_FEATURES, P_FEATURES
from .config import DELAYS_S, SUBJECTS, SUBJECTS_164, SUBJECTS_338, TRAJECTORY_FEATURES
from .features import causal_trajectory_features


def _bool(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).casefold() in {"true", "1", "yes"}


def _float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pressure_features(delta: np.ndarray, adaptive: Dict[str, np.ndarray], index: int,
                       event: dict) -> dict | None:
    """V2.1 pressure schema with NumPy 1.21-compatible integration."""
    fs = float(C.DP_FS_HZ)
    one, half = int(round(fs)), int(round(0.5 * fs))
    if index < one or index >= len(delta):
        return None
    confirm = float(adaptive["adaptive_confirm"][index])
    sigma = float(adaptive["sigma_dpdt"][index])
    if not np.isfinite(confirm) or confirm <= 0 or not np.isfinite(sigma) or sigma <= 0:
        return None
    start = max(0, min(index, int(event["start_index"])))
    trough = max(0, min(index, int(event["local_trough_index"])))
    history = np.asarray(delta[trough:index + 1], dtype=float)
    recent = np.asarray(delta[index - one:index + 1], dtype=float)
    auc_delta = np.asarray(delta[start:index + 1], dtype=float)
    auc_start = np.asarray(adaptive["adaptive_start"][start:index + 1], dtype=float)
    if (history.size < 2 or recent.size != one + 1 or auc_delta.size < 2
            or not np.isfinite(history).all() or not np.isfinite(recent).all()
            or not np.isfinite(auc_delta).all() or not np.isfinite(auc_start).all()):
        return None
    dpdt = np.diff(recent) * fs
    slope1 = float(np.mean(dpdt) / sigma)
    slope05 = float(np.mean(dpdt[-half:]) / sigma)
    elapsed = (auc_delta.size - 1) / fs
    auc = float(np.trapz(np.maximum(auc_delta - auc_start, 0.0), dx=1.0 / fs))
    return {
        "delta_p_current_norm": float(delta[index] / confirm),
        "delta_p_peak_so_far_norm": float((np.max(history) - history[0]) / confirm),
        "pressure_slope_0p5s_norm": slope05,
        "pressure_slope_change_norm": slope05 - slope1,
        "positive_dpdt_fraction_1s": float(np.mean(dpdt > 0)),
        "auc_growth_rate_norm": float(auc / max(elapsed * confirm, np.finfo(float).eps)),
    }


def _population_priors(reference_root: Path) -> tuple[float, float]:
    params = pd.read_csv(reference_root / "subject_adaptive_params.csv")
    return (float(pd.to_numeric(params["warmup_prior_sigma_p"], errors="coerce").median()),
            float(pd.to_numeric(params["sigma_dpdt_median"], errors="coerce").median()))


def _load_338(cycles_root: Path, reference_root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = pd.read_csv(reference_root / "dataset_manifest.csv")
    events = pd.read_csv(reference_root / "pressure_events.csv")
    params = pd.read_csv(reference_root / "subject_adaptive_params.csv")
    manifest = manifest[manifest["subject"].isin(SUBJECTS_338)].copy()
    events = events[events["subject"].isin(SUBJECTS_338)].copy()
    priors = {str(r.subject): (float(r.warmup_prior_sigma_p), float(r.sigma_dpdt_median))
              for r in params.itertuples(index=False)}
    histories = {subject: AdaptiveHistory() for subject in SUBJECTS_338}
    cache = {}
    for _, row in manifest.sort_values(["subject", "cycle_start_s"]).iterrows():
        subject, cycle_id = str(row["subject"]), str(row["cycle_id"])
        load_row = row.copy(); load_row["dsd_cycle_id"] = cycle_id
        cycle = load_cycle(cycles_root, load_row)
        delta, _, adaptive = adaptive_local_pressure_events(cycle, histories[subject], *priors[subject])
        eus, eus_valid = causal_eus_envelope_100hz(cycle)
        cache[(subject, cycle_id)] = {
            "dataset": "338", "cycle": cycle, "delta": delta, "adaptive": adaptive,
            "eus_env": eus, "eus_valid": eus_valid, "manifest": row.to_dict(),
        }
    events["dataset"] = "338"
    files = [reference_root / "dataset_manifest.csv", reference_root / "pressure_events.csv",
             reference_root / "teacher_labels.csv", reference_root / "subject_adaptive_params.csv"]
    return cache, events, manifest.assign(dataset="338"), [str(p.resolve()) for p in files]


def _load_164(cycles_root: Path, labels_path: Path, reference_root: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = pd.read_csv(cycles_root / "nvc_cycle_manifest.csv")
    manifest = manifest[manifest["subject"].isin(SUBJECTS_164)].copy()
    labels = pd.read_csv(labels_path)
    labels = labels[labels["subject"].isin(SUBJECTS_164)].copy()
    prior_sigma, prior_dpdt = _population_priors(reference_root)
    histories = {subject: AdaptiveHistory() for subject in SUBJECTS_164}
    cache, rows = {}, []
    for _, row in manifest.sort_values(["subject", "cycle_start_s"]).iterrows():
        subject, cycle_id = str(row["subject"]), str(row["cycle_id"])
        load_row = row.copy(); load_row["dsd_cycle_id"] = cycle_id
        cycle = load_cycle(cycles_root, load_row)
        delta, detected, adaptive = adaptive_local_pressure_events(
            cycle, histories[subject], prior_sigma, prior_dpdt)
        eus, eus_valid = causal_eus_envelope_100hz(cycle)
        cache[(subject, cycle_id)] = {
            "dataset": "164", "cycle": cycle, "delta": delta, "adaptive": adaptive,
            "eus_env": eus, "eus_valid": eus_valid, "manifest": row.to_dict(),
        }
        frozen = labels[(labels["subject"] == subject) & (labels["cycle_id"] == cycle_id)].copy()
        frozen = frozen.sort_values("confirm_time_s").reset_index(drop=True)
        detected = sorted(detected, key=lambda e: _float(e.get("confirm_index"), np.inf))
        if len(frozen) != len(detected):
            raise RuntimeError(f"164 event mismatch for {subject}/{cycle_id}: {len(frozen)} != {len(detected)}")
        for descriptor, (_, label) in zip(detected, frozen.iterrows()):
            event = dict(descriptor)
            event.update({
                "dataset": "164", "subject": subject, "cycle_id": cycle_id,
                "event_id": str(label["event_id"]), "event_uid": str(label["event_uid"]),
                "teacher_label": str(label["teacher_label"]),
                "data_valid": not _bool(label.get("data_invalid", False)),
                "matched_urine_event_id": str(label.get("matched_urine_event_id", "")),
                "peak_delta_p": _float(label.get("peak_delta_p_mmHg")),
                "local_prominence_mmHg": _float(label.get("local_prominence_mmHg")),
                "local_peak_time_s": _float(label.get("local_peak_time_s")),
                "local_recovery_time_s": _float(label.get("recovery_confirm_s")),
                "recovery_start_s": _float(label.get("recovery_start_s")),
                "end_s": _float(label.get("end_s")),
                "start_s": _float(label.get("candidate_start_s")),
                "cycle_duration_s": _float(row.get("cycle_duration_s")),
            })
            rows.append(event)
    events = pd.DataFrame(rows)
    files = [cycles_root / "nvc_cycle_manifest.csv", labels_path]
    return cache, events, manifest.assign(dataset="164"), [str(p.resolve()) for p in files]


def load_development_streams(cycles_338: Path, reference_338: Path,
                             cycles_164: Path, labels_164: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str]]:
    c338, e338, m338, f338 = _load_338(Path(cycles_338), Path(reference_338))
    c164, e164, m164, f164 = _load_164(Path(cycles_164), Path(labels_164), Path(reference_338))
    events = pd.concat([e338, e164], ignore_index=True, sort=False)
    manifest = pd.concat([m338, m164], ignore_index=True, sort=False)
    if tuple(sorted(events["subject"].unique())) != tuple(sorted(SUBJECTS)):
        raise RuntimeError("V3.1 raw event cohort does not contain all registered animals")
    if events["event_uid"].duplicated().any():
        raise RuntimeError("V3.1 event_uid is not unique")
    return {**c338, **c164}, events, manifest, f338 + f164


def _failure_reason(event: dict, index: int, cycle: dict) -> str:
    if not np.isfinite(_float(event.get("confirm_index"))):
        return "NO_CONFIRM_INDEX"
    if not (0 <= index < len(cycle["t_abs_s"])):
        return "OUTSIDE_CYCLE"
    if not _bool(event.get("data_valid", False)):
        return "DATA_INVALID"
    if _bool(event.get("adaptive_warmup", True)):
        return "ADAPTIVE_WARMUP_INCOMPLETE"
    return ""


def build_delayed_features(cache: dict, events: pd.DataFrame,
                           delays: Iterable[float] = DELAYS_S) -> pd.DataFrame:
    rows = []
    feature_names = tuple(C0_FEATURES) + tuple(P_FEATURES) + tuple(EUS_FEATURES) + TRAJECTORY_FEATURES
    for event_row in events.itertuples(index=False):
        event = event_row._asdict()
        subject, cycle_id = str(event["subject"]), str(event["cycle_id"])
        item = cache[(subject, cycle_id)]
        cycle, delta, adaptive = item["cycle"], item["delta"], item["adaptive"]
        time = np.asarray(cycle["t_abs_s"], dtype=float)
        confirm_index = int(event["confirm_index"]) if np.isfinite(_float(event.get("confirm_index"))) else -1
        confirm_time = float(time[confirm_index]) if 0 <= confirm_index < len(time) else np.nan
        recovery = _float(event.get("local_recovery_time_s"))
        end_time = _float(event.get("end_s"))
        if not np.isfinite(recovery):
            recovery = end_time
        urine = _float(item["manifest"].get("urine_output_onset_s"))
        if not np.isfinite(urine):
            urine = _float(item["manifest"].get("terminal_urine_episode_onset_s"))
        for delay in delays:
            decision_time = confirm_time + float(delay) if np.isfinite(confirm_time) else np.nan
            index = int(np.searchsorted(time, decision_time, side="right") - 1) if np.isfinite(decision_time) else -1
            reason = _failure_reason(event, index, cycle)
            eligible = reason == ""
            row = {
                "dataset": item["dataset"], "subject": subject, "cycle_id": cycle_id,
                "event_id": str(event["event_id"]), "event_uid": str(event["event_uid"]),
                "teacher_label": str(event["teacher_label"]), "confirm_index": confirm_index,
                "confirm_time_s": confirm_time, "decision_delay_s": float(delay),
                "decision_time_s": decision_time, "decision_index": index,
                "feature_max_time_s": float(time[index]) if 0 <= index < len(time) else np.nan,
                "base_eligible": eligible, "base_failure_reason": reason,
                "data_valid": _bool(event.get("data_valid", False)),
                "event_recovery_time_s_eval_only": recovery,
                "urine_onset_s_eval_only": urine,
                "still_active": bool(np.isfinite(recovery) and decision_time < recovery),
                "actionable": bool(eligible and np.isfinite(recovery) and decision_time < recovery
                                   and (not np.isfinite(urine) or decision_time < urine)),
                **{name: np.nan for name in feature_names},
                "spectral_scorable": False, "spectral_failure_reason": "BASE_EVENT_UNSCORABLE" if not eligible else "",
            }
            if eligible:
                p = _pressure_features(delta, adaptive, index, event)
                if p is None:
                    row["base_eligible"] = False; row["base_failure_reason"] = "PRESSURE_FEATURE_UNSCORABLE"
                else:
                    row.update(p)
                    e = _eus_features(delta, item["eus_env"], item["eus_valid"], index)
                    if e is not None:
                        row.update(e)
                    c0 = decision_feature_at_index(
                        cycle, delta, item["eus_env"], item["eus_valid"], adaptive, index,
                        {"start_index": int(event["start_index"]),
                         "local_trough_index": int(event["local_trough_index"])}, require_eus=False)
                    if c0 is not None:
                        row.update({name: c0[name] for name in C0_FEATURES})
                    trajectory = causal_trajectory_features(
                        delta, item["eus_env"], item["eus_valid"], index, event, adaptive)
                    if trajectory is not None:
                        row.update(trajectory)
                    spectral, spectral_reason = _low_frequency_feature(cycle, delta, index)
                    row["relative_pressure_power_0p2_0p6"] = spectral
                    row["spectral_scorable"] = bool(np.isfinite(spectral))
                    row["spectral_failure_reason"] = spectral_reason
            rows.append(row)
    result = pd.DataFrame(rows)
    causal = result["feature_max_time_s"].isna() | (result["feature_max_time_s"] <= result["decision_time_s"] + 1e-9)
    if not causal.all():
        raise RuntimeError("V3.1 future feature timestamp detected")
    return result


def build_aligned_traces(cache: dict, events: pd.DataFrame,
                         subjects=("STxF26", "STxF33", "STxF34", "STxF37")) -> pd.DataFrame:
    rows = []
    for event_row in events[events["subject"].isin(subjects)].itertuples(index=False):
        event = event_row._asdict(); item = cache[(str(event["subject"]), str(event["cycle_id"]))]
        cycle = item["cycle"]; time = np.asarray(cycle["t_abs_s"], dtype=float)
        confirm = _float(event.get("confirm_time_s"))
        if not np.isfinite(confirm) and np.isfinite(_float(event.get("confirm_index"))):
            confirm = float(time[int(event["confirm_index"])])
        if not np.isfinite(confirm):
            continue
        indices = np.flatnonzero((time >= confirm - 10.0) & (time <= confirm + 5.0))
        baseline_indices = np.flatnonzero((time >= confirm - 10.0) & (time < confirm - 2.0))
        med, mad = robust_location_scale(item["eus_env"][baseline_indices])
        if not np.isfinite(mad) or mad <= 0:
            mad = np.nan
        pressure = np.asarray(cycle["bladder_pressure_mmHg"], dtype=float)
        dpdt = np.r_[0.0, np.diff(pressure) * C.DP_FS_HZ]
        for index in indices[::5]:
            rows.append({
                "dataset": item["dataset"], "subject": str(event["subject"]),
                "cycle_id": str(event["cycle_id"]), "event_uid": str(event["event_uid"]),
                "teacher_label": str(event["teacher_label"]), "time_from_confirmation_s": float(time[index] - confirm),
                "pressure_mmHg": pressure[index], "delta_p": item["delta"][index], "dpdt": dpdt[index],
                "eus_envelope": item["eus_env"][index],
                "eus_causal_normalized": (item["eus_env"][index] - med) / mad if np.isfinite(mad) else np.nan,
            })
    return pd.DataFrame(rows)
