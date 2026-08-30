"""V5 data construction with repaired event timing and prospective splits.

The V4 teacher labels and causal feature extractor remain immutable.  This
adapter only re-materializes rows for F26/F37 so that missing F37 absolute
confirmation times are recovered from the registered sample index/time axis,
and so that stable windows are available in cycles such as F26/B15 that have no
NVC anchor.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .source_adapter import load_frozen_v31_features, _load_pair
from .stable_sampling import _stable_candidates, _select_stable_for_nvc
from .feature_extraction import extract_v4_features
from . import config as C


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def event_confirm_time(cycle: dict, row) -> float:
    """Return registered confirmation time, repairing missing F37 values."""
    if _finite(getattr(row, "confirm_time_s", np.nan)):
        return float(row.confirm_time_s)
    idx = int(float(row.confirm_index)) if _finite(getattr(row, "confirm_index", np.nan)) else -1
    t = np.asarray(cycle.get("t_abs_s", []), dtype=float)
    return float(t[idx]) if 0 <= idx < t.size else np.nan


def _event_frame_with_times(cycle: dict, events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["confirm_time_s"] = [event_confirm_time(cycle, r) for r in out.itertuples(index=False)]
    # The stable sampler also uses onset/end/recovery fields when available;
    # those fields are already in the frozen source artifact.
    return out


def _row_from_features(r, cycle, features, role, source_event_uid, sample_uid,
                       decision_index, decision_time_s, reason, target):
    start = float(np.asarray(cycle.get("cycle_start_s", np.nan)).item())
    end = float(np.asarray(cycle.get("cycle_end_s", np.nan)).item())
    row = {
        "sample_uid": str(sample_uid), "subject": str(r.subject), "cycle_id": str(r.cycle_id),
        "dataset": str(getattr(r, "dataset", "")), "sample_role": role,
        "teacher_label": role, "target": int(target), "decision_index": int(decision_index),
        "decision_time_s": float(decision_time_s), "source_event_uid": str(source_event_uid),
        "feature_failure_reason": str(reason or ""), "cycle_start_s": start,
        "cycle_end_s": end, "cycle_duration_s": end - start,
    }
    row.update(features)
    return row


def build_v5_dataset(v31_root: Path = C.V31_ROOT):
    """Build V5 rows, challenges, and registered cycle metadata.

    Stable windows are sampled at most three per NVC event, with an additional
    three per cycle when a cycle contains no NVC.  This keeps the teacher set
    small while preserving negative-only cycles for prospective false-trigger
    evaluation.
    """
    _, events, manifest, paths = load_frozen_v31_features(v31_root)
    events = events[events.subject.isin(C.SUBJECTS)].copy()
    manifest = manifest[manifest.subject.isin(C.SUBJECTS)].copy()
    rows, challenge_rows, stable_audit, cycle_rows = [], [], [], []

    for mr in manifest.sort_values(["subject", "cycle_start_s", "cycle_id"]).itertuples(index=False):
        key = (str(mr.subject), str(mr.cycle_id))
        if key not in paths:
            continue
        try:
            item = _load_pair(paths[key])
        except (FileNotFoundError, OSError):
            continue
        cycle = item["cycle"]
        cg = events[(events.subject == key[0]) & (events.cycle_id == key[1])].copy()
        cg = _event_frame_with_times(cycle, cg)
        cycle_rows.append({
            "subject": key[0], "cycle_id": key[1], "dataset": str(mr.dataset),
            "cycle_start_s": float(mr.cycle_start_s), "cycle_end_s": float(mr.cycle_end_s),
            "cycle_duration_s": float(mr.cycle_duration_s),
        })
        candidates = _stable_candidates(cycle, cg, key[0], key[1], set())
        used = set()
        nvc = cg[cg.teacher_label.eq("NVC_CORE")].sort_values("confirm_time_s")
        # NVC positive rows are immutable labels; only their missing time field
        # is repaired for causal EUS/spectral feature extraction.
        for er in nvc.itertuples(index=False):
            idx = int(float(er.confirm_index)) if _finite(er.confirm_index) else -1
            t = event_confirm_time(cycle, er)
            f, reason = extract_v4_features(cycle, idx, int(er.start_index), t)
            rows.append(_row_from_features(er, cycle, f, "NVC_CORE", er.event_uid,
                                            er.event_uid, idx, t, reason, 1))
            picked = _select_stable_for_nvc(er, candidates, used, needed=3)
            for sidx, sts in picked:
                sf, sreason = extract_v4_features(cycle, int(sidx),
                                                   max(0, int(sidx - round(2 * C.DP_FS_HZ))),
                                                   float(sts))
                uid = f"STABLE::{key[0]}::{key[1]}::{int(sidx)}"
                sr = _row_from_features(er, cycle, sf, "STABLE_FILLING", er.event_uid,
                                        uid, int(sidx), float(sts), sreason, 0)
                rows.append(sr)
                stable_audit.append({
                    "sample_uid": uid, "matched_nvc_event_uid": str(er.event_uid),
                    "subject": key[0], "cycle_id": key[1],
                    "distance_to_nvc_s": abs(float(sts) - float(t)),
                    "stable_sampling_rule": "same_cycle_causal_stable_window",
                })
        # Negative-only cycles are retained explicitly.  For NVC cycles the
        # matched samples above are used; for empty cycles choose three stable
        # causal windows so B15 can be scored for false-trigger rate.
        if len(nvc) == 0:
            for sidx, sts in sorted(candidates, key=lambda q: q[1])[:3]:
                if int(sidx) in used:
                    continue
                used.add(int(sidx))
                sf, sreason = extract_v4_features(cycle, int(sidx),
                                                   max(0, int(sidx - round(2 * C.DP_FS_HZ))),
                                                   float(sts))
                uid = f"STABLE::{key[0]}::{key[1]}::{int(sidx)}"
                sr = _row_from_features(
                    type("R", (), {"subject": key[0], "cycle_id": key[1], "dataset": mr.dataset})(),
                    cycle, sf, "STABLE_FILLING", "", uid, int(sidx), float(sts), sreason, 0)
                rows.append(sr)
                stable_audit.append({
                    "sample_uid": uid, "matched_nvc_event_uid": "", "subject": key[0],
                    "cycle_id": key[1], "distance_to_nvc_s": np.nan,
                    "stable_sampling_rule": "negative_only_cycle_causal_stable_window",
                })

        # Challenges are materialized but never enter the training frame.
        for er in cg[cg.teacher_label.eq("PREVOID_PROGRESSIVE")].itertuples(index=False):
            idx = int(float(er.confirm_index)) if _finite(er.confirm_index) else -1
            t = event_confirm_time(cycle, er)
            f, reason = extract_v4_features(cycle, idx, int(er.start_index), t)
            row = _row_from_features(er, cycle, f, "PREVOID_PROGRESSIVE", er.event_uid,
                                     er.event_uid, idx, t, reason, 0)
            row["challenge_type"] = "PREVOID_CHALLENGE"
            challenge_rows.append(row)
        void = cycle.get("void_start_s", np.nan)
        void = float(np.asarray(void).item()) if np.asarray(void).ndim == 0 else np.nan
        if np.isfinite(void):
            tarr = np.asarray(cycle.get("t_abs_s", []), dtype=float)
            idx = int(np.searchsorted(tarr, void, side="right") - 1) if tarr.size else -1
            vr = type("R", (), {"subject": key[0], "cycle_id": key[1], "dataset": mr.dataset})()
            f, reason = extract_v4_features(cycle, idx, max(0, idx - int(2 * C.DP_FS_HZ)), void)
            row = _row_from_features(vr, cycle, f, "VOID_CONFIRMED", "", f"VOID::{key[0]}::{key[1]}",
                                     idx, void, reason, 0)
            row["challenge_type"] = "VOID_CHALLENGE"
            challenge_rows.append(row)

    train = pd.DataFrame(rows)
    challenges = pd.DataFrame(challenge_rows)
    audit = pd.DataFrame(stable_audit)
    cycles = pd.DataFrame(cycle_rows)
    if train.empty:
        raise RuntimeError("V5 dataset construction produced no rows")
    return train, challenges, audit, cycles, paths, events
