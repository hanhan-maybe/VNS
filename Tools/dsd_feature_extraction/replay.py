"""Event-time causal VoidGuard replay and event-level metrics."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config as C
from .models import DANGEROUS, replay_trigger_mask


GUARD_CODES = {"CLEAR": 0, "VOID_SUSPECT": 1, "VOID_ACTIVE": 2, "POST_VOID_HOLD": 3, "DATA_INVALID": 4}


def replay_model(frame: pd.DataFrame, model_name: str, threshold: float, probability_col: str) -> pd.DataFrame:
    out = frame.copy()
    if model_name == "M0":
        out["probability"] = np.where((out["sample_type"] == "EVENT") & (out["local_prominence_mmHg"] > C.CONFIRM_THRESHOLD_MMHG), 1.0, 0.0)
        probability_col = "probability"; threshold = 0.5
    elif model_name == "M0A":
        out["probability"] = np.where((out["sample_type"] == "EVENT") & out["teacher_label"].isin(
            ["NVC_CORE", "NVC_ADAPTIVE", "PREVOID_PROGRESSIVE", "VOID_CONFIRMED"]), 1.0, 0.0)
        probability_col = "probability"; threshold = 0.5
    out["trigger"] = replay_trigger_mask(out, probability_col, threshold)
    out["model"] = model_name
    out["probability_threshold"] = threshold
    out["guard_state_at_decision"] = np.where(out["data_valid"], "CLEAR", "DATA_INVALID")
    out.loc[~out["trigger"] & (out["sample_type"] == "EVENT") & out["data_valid"], "guard_state_at_decision"] = "VOID_SUSPECT"
    out["trigger_label"] = np.where(out["trigger"], out["teacher_label"], "NO_TRIGGER")
    out["trigger_delay_s"] = np.where(out["trigger"], 0.0, np.nan)
    return out


def event_metrics(replay: pd.DataFrame, subject: str, model_name: str) -> Dict:
    f = replay[(replay["subject"] == subject) & (replay["sample_type"] == "EVENT")]
    nvc = f["teacher_label"].isin(["NVC_CORE", "NVC_ADAPTIVE"])
    hits = int((f["trigger"] & nvc).sum())
    triggers = int(f["trigger"].sum())
    duration_h = float(f.groupby("cycle_id")["cycle_duration_s"].first().sum() / 3600.0)
    other_false = int((f["trigger"] & ~nvc & ~f["teacher_label"].isin(DANGEROUS)).sum())
    cycles = max(1, f["cycle_id"].nunique())
    return {
        "model": model_name, "held_out_subject": subject, "nvc_sensitivity": hits / int(nvc.sum()) if nvc.sum() else np.nan,
        "nvc_ppv": hits / triggers if triggers else np.nan, "storage_false_triggers_per_h": other_false / duration_h if duration_h else np.nan,
        "triggers_per_cycle": triggers / cycles, "median_confirm_to_trigger_s": 0.0 if triggers else np.nan,
        "void_confirmed_triggers": int((f["trigger"] & (f["teacher_label"] == "VOID_CONFIRMED")).sum()),
        "prevoid_progressive_triggers": int((f["trigger"] & (f["teacher_label"] == "PREVOID_PROGRESSIVE")).sum()),
        "n_valid_nvc": int(nvc.sum()), "valid_analysis_hours": duration_h, "n_triggers": triggers,
    }


def stream_vectors(cycle: Dict, delta: np.ndarray, eus_env: np.ndarray, event_rows: pd.DataFrame,
                   selected_model: str, threshold: float, adaptive=None) -> Dict[str, np.ndarray]:
    n = len(delta)
    state = np.full(n, GUARD_CODES["CLEAR"], dtype=np.int8)
    valid = np.asarray(cycle["cmg_valid_100hz"], dtype=bool) & np.isfinite(delta)
    state[~valid] = GUARD_CODES["DATA_INVALID"]
    score = np.full(n, np.nan, dtype=np.float32)
    trigger = np.zeros(n, dtype=bool)
    for _, row in event_rows[event_rows["sample_type"] == "EVENT"].iterrows():
        i = int(row["confirm_index"])
        if 0 <= i < n:
            score[i] = float(row.get("probability", 1.0 if selected_model == "M0" else np.nan))
            if bool(row.get("trigger", False)):
                trigger[i] = True
                hold_end = min(n, i + int(C.POST_EVENT_LOCKOUT_S * C.DP_FS_HZ))
                state[i:hold_end] = GUARD_CODES["POST_VOID_HOLD"]
            else:
                recovery = int(row["recovery_confirm_index"]) if np.isfinite(row["recovery_confirm_index"]) else n - 1
                active_start = min(recovery + 1, i + int(C.GUARD_PROGRESSIVE_S * C.DP_FS_HZ))
                state[i:active_start] = GUARD_CODES["VOID_SUSPECT"]
                if recovery >= active_start:
                    state[active_start:recovery + 1] = GUARD_CODES["VOID_ACTIVE"]
                    hold_end = min(n, recovery + 1 + int(C.POST_EVENT_LOCKOUT_S * C.DP_FS_HZ))
                    state[recovery + 1:hold_end] = GUARD_CODES["POST_VOID_HOLD"]
    state[~valid] = GUARD_CODES["DATA_INVALID"]
    out = {"t_abs_s": np.asarray(cycle["t_abs_s"]), "pressure_mmHg": np.asarray(cycle["bladder_pressure_mmHg"]),
            "cmg_valid": np.asarray(cycle["cmg_valid_100hz"]), "eus_envelope": np.asarray(eus_env),
            "delta_p": np.asarray(delta), "expected_guard_state": state, "expected_score": score,
            "expected_trigger": trigger, "guard_state_codes_json": np.array(str(GUARD_CODES))}
    if adaptive is not None:
        out.update({"adaptive_start": np.asarray(adaptive["adaptive_start"]),
                    "adaptive_confirm": np.asarray(adaptive["adaptive_confirm"]),
                    "adaptive_recovery": np.asarray(adaptive["adaptive_recovery"]),
                    "adaptive_warmup": np.asarray(adaptive["adaptive_warmup"])})
    return out
