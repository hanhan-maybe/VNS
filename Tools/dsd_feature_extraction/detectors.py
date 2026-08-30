"""Native-Volume teacher evidence and causal 100 Hz pressure event detection."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import lfilter

from . import config as C


@dataclass
class AdaptiveHistory:
    """Subject-only causal CLEAR history, carried chronologically across cycles."""
    residual_clear: deque = field(default_factory=lambda: deque(maxlen=int(C.ADAPTIVE_HISTORY_MAX_S * C.DP_FS_HZ)))
    dpdt_clear: deque = field(default_factory=lambda: deque(maxlen=int(C.ADAPTIVE_HISTORY_MAX_S * C.DP_FS_HZ)))
    valid_clear_samples: int = 0


def adaptive_threshold_from_history(history: AdaptiveHistory, prior_sigma_p: float, prior_sigma_dpdt: float,
                                    exploratory: bool = False) -> Dict[str, float]:
    ready_n = int(C.BASELINE_WINDOW_S * C.DP_FS_HZ)
    warmup = len(history.residual_clear) < ready_n
    if warmup:
        sigma = max(float(prior_sigma_p), 1e-6)
        q = (C.EXPLORATORY_SIGMA_MULTIPLIER if exploratory else C.ADAPTIVE_SIGMA_MULTIPLIER) * sigma
        sigma_dpdt = max(float(prior_sigma_dpdt), 1e-6)
    else:
        residual = np.asarray(history.residual_clear, dtype=np.float64)
        med = float(np.median(residual))
        sigma = max(1.4826 * float(np.median(np.abs(residual - med))), 1e-6)
        q = float(np.percentile(residual, C.EXPLORATORY_Q_PERCENTILE if exploratory else C.ADAPTIVE_Q_PERCENTILE))
        deriv = np.asarray(history.dpdt_clear, dtype=np.float64)
        dmed = float(np.median(deriv)) if deriv.size else 0.0
        sigma_dpdt = max(1.4826 * float(np.median(np.abs(deriv - dmed))) if deriv.size else prior_sigma_dpdt, 1e-6)
    multiplier = C.EXPLORATORY_SIGMA_MULTIPLIER if exploratory else C.ADAPTIVE_SIGMA_MULTIPLIER
    confirm = float(np.clip(max(multiplier * sigma, q), *C.ADAPTIVE_CONFIRM_BOUNDS_MMHG))
    return {"sigma_p": sigma, "sigma_dpdt": sigma_dpdt, "q_positive": q, "confirm": confirm,
            "start": C.ADAPTIVE_START_RATIO * confirm, "recovery": C.ADAPTIVE_RECOVERY_RATIO * confirm,
            "warmup": bool(warmup)}


def adaptive_local_pressure_events(cycle: Dict, history: AdaptiveHistory, prior_sigma_p: float,
                                   prior_sigma_dpdt: float):
    """Causal adaptive detector with local trough/peak recovery splitting."""
    p = np.asarray(cycle["bladder_pressure_mmHg"], dtype=np.float64)
    valid = np.asarray(cycle["cmg_valid_100hz"], dtype=bool) & np.isfinite(p)
    valid &= (p >= C.CMG_VALID_RANGE_MMHG[0]) & (p <= C.CMG_VALID_RANGE_MMHG[1])
    fs = C.DP_FS_HZ
    jump = np.r_[0.0, np.abs(np.diff(p)) * fs]
    valid &= jump <= C.PRESSURE_JUMP_LIMIT_MMHG_S
    baseline_n = int(C.BASELINE_WINDOW_S * fs)
    update_n = max(1, int(C.UPDATE_STEP_S * fs))
    confirm_n = int(C.CONFIRM_HOLD_S * fs)
    recovery_n = int(C.RECOVERY_HOLD_S * fs)
    baseline = deque(maxlen=baseline_n)
    residual = np.full(p.size, np.nan)
    arrays = {name: np.full(p.size, np.nan) for name in
              ("adaptive_start", "adaptive_confirm", "adaptive_recovery", "explore_start", "explore_confirm",
               "sigma_p", "sigma_dpdt")}
    warmup = np.ones(p.size, dtype=bool)
    state = "IDLE"
    trough_i = None; start_i = None; peak_i = None
    main_first = possible_first = None
    main_count = possible_count = recovery_count = 0
    recovery_first = None; data_invalid = False
    current_main = adaptive_threshold_from_history(history, prior_sigma_p, prior_sigma_dpdt)
    current_explore = adaptive_threshold_from_history(history, prior_sigma_p, prior_sigma_dpdt, exploratory=True)
    events = []

    def finish(end_i, recovered, invalid=False):
        nonlocal state, trough_i, start_i, peak_i, main_first, possible_first
        nonlocal main_count, possible_count, recovery_count, recovery_first, data_invalid
        if start_i is None or peak_i is None:
            state = "IDLE"; trough_i = end_i; return
        prominence = float(residual[peak_i] - residual[trough_i])
        fall = float(residual[peak_i] - residual[end_i]) if np.isfinite(residual[end_i]) else np.nan
        fraction = fall / prominence if prominence > 0 and np.isfinite(fall) else np.nan
        if main_first is not None:
            decision_i, level = main_first, "MAIN"
        elif possible_first is not None:
            decision_i, level = possible_first, "POSSIBLE"
        else:
            decision_i, level = None, "GREY"
        events.append({"start_index": start_i, "end_index": end_i, "confirm_index": decision_i,
                       "main_confirm_index": main_first, "possible_confirm_index": possible_first,
                       "peak_index": peak_i, "local_trough_index": trough_i,
                       "recovery_start_index": recovery_first, "recovery_confirm_index": end_i if recovered else None,
                       "recovered": bool(recovered), "locally_recovered": bool(recovered), "data_invalid": bool(invalid),
                       "detection_level": level, "local_prominence_mmHg": prominence,
                       "fall_from_peak_mmHg": fall, "recovery_fraction": fraction,
                       "adaptive_start_at_confirm": float(arrays["adaptive_start"][decision_i]) if decision_i is not None else np.nan,
                       "adaptive_confirm_at_confirm": float(arrays["adaptive_confirm"][decision_i]) if decision_i is not None else np.nan,
                       "adaptive_recovery_at_confirm": float(arrays["adaptive_recovery"][decision_i]) if decision_i is not None else np.nan,
                       "explore_confirm_at_confirm": float(arrays["explore_confirm"][decision_i]) if decision_i is not None else np.nan,
                       "sigma_p_at_confirm": float(arrays["sigma_p"][decision_i]) if decision_i is not None else np.nan,
                       "sigma_dpdt_at_confirm": float(arrays["sigma_dpdt"][decision_i]) if decision_i is not None else np.nan,
                       "adaptive_warmup": bool(warmup[decision_i]) if decision_i is not None else True})
        state = "IDLE"; trough_i = end_i; start_i = peak_i = None
        main_first = possible_first = recovery_first = None
        main_count = possible_count = recovery_count = 0; data_invalid = False

    for i in range(p.size):
        if i % update_n == 0:
            current_main = adaptive_threshold_from_history(history, prior_sigma_p, prior_sigma_dpdt)
            current_explore = adaptive_threshold_from_history(history, prior_sigma_p, prior_sigma_dpdt, exploratory=True)
        for key, value in (("adaptive_start", current_main["start"]), ("adaptive_confirm", current_main["confirm"]),
                           ("adaptive_recovery", current_main["recovery"]), ("explore_start", current_explore["start"]),
                           ("explore_confirm", current_explore["confirm"]), ("sigma_p", current_main["sigma_p"]),
                           ("sigma_dpdt", current_main["sigma_dpdt"])):
            arrays[key][i] = value
        warmup[i] = current_main["warmup"]
        if not valid[i]:
            if state != "IDLE": finish(max(0, i - 1), False, invalid=True)
            continue
        if len(baseline) < baseline_n:
            baseline.append(p[i]); continue
        base = float(np.median(np.asarray(baseline)))
        residual[i] = p[i] - base
        if state == "IDLE":
            if trough_i is None or not np.isfinite(residual[trough_i]) or residual[i] <= residual[trough_i]:
                trough_i = i
            rise = residual[i] - residual[trough_i]
            if rise > current_explore["start"]:
                state = "CANDIDATE"; start_i = i; peak_i = i
            else:
                # CLEAR-only causal updates; no EUS, Volume or labels are consulted.
                previous = residual[i - 1] if i > 0 and np.isfinite(residual[i - 1]) else residual[i]
                deriv = (residual[i] - previous) * fs
                if abs(deriv) <= C.PRESSURE_JUMP_LIMIT_MMHG_S:
                    history.residual_clear.append(float(residual[i])); history.dpdt_clear.append(float(deriv)); history.valid_clear_samples += 1
                baseline.append(p[i])
        else:
            if residual[i] > residual[peak_i]: peak_i = i
            prominence = residual[i] - residual[trough_i]
            if possible_count < confirm_n:
                if prominence > current_explore["confirm"]:
                    if possible_count == 0: possible_first = i
                    possible_count += 1
                else:
                    possible_count = 0; possible_first = None
            if main_count < confirm_n:
                if prominence > current_main["confirm"]:
                    if main_count == 0: main_first = i
                    main_count += 1
                else:
                    main_count = 0; main_first = None
            peak_prom = residual[peak_i] - residual[trough_i]
            fall = residual[peak_i] - residual[i]
            recovery_limit = residual[trough_i] + max(current_main["recovery"], 0.40 * peak_prom)
            recovery_ok = fall >= C.LOCAL_RECOVERY_FRACTION * peak_prom and residual[i] <= recovery_limit
            if recovery_ok:
                if recovery_count == 0: recovery_first = i
                recovery_count += 1
                if recovery_count >= recovery_n:
                    finish(i, True)
            else:
                recovery_count = 0; recovery_first = None
    if state != "IDLE": finish(p.size - 1, False, invalid=False)
    arrays["adaptive_warmup"] = warmup
    arrays["valid"] = valid
    return residual, events, arrays


def estimate_subject_noise(cycles: List[Dict]) -> Tuple[float, float]:
    """Conservative prepass used only to initialize other animals' first 25 s."""
    residuals, derivatives = [], []
    for cycle in cycles:
        delta, _, ready = causal_pressure_events(cycle)
        mask = ready & np.isfinite(delta) & (np.abs(delta) < C.CANDIDATE_THRESHOLD_MMHG)
        x = delta[mask]
        if x.size:
            residuals.append(x)
            derivatives.append(np.diff(x) * C.DP_FS_HZ)
    x = np.concatenate(residuals) if residuals else np.array([C.CONFIRM_THRESHOLD_MMHG / 4])
    d = np.concatenate(derivatives) if derivatives else np.array([1.0])
    sigma = max(1.4826 * float(np.median(np.abs(x - np.median(x)))), 1e-3)
    sigma_d = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), 1e-3)
    return sigma, sigma_d


def _causal_moving_average(x: np.ndarray, n: int) -> np.ndarray:
    n = max(1, int(n))
    return lfilter(np.ones(n, dtype=np.float64) / n, [1.0], x)


def determine_volume_direction(volume: Dict, cycle_rows=None) -> int:
    """Infer native Volume direction from absolute-time windows only."""
    x, fs = np.asarray(volume["raw"], dtype=float), float(volume["fs"])
    time_s = np.asarray(volume.get("time_s", np.arange(x.size) / fs), dtype=float)
    signs = []
    magnitudes = []
    rows = cycle_rows.iterrows() if cycle_rows is not None else []
    for _, row in rows:
        end = float(row["cycle_end_s"])
        a, b = np.searchsorted(time_s, [max(time_s[0], end - 15), max(time_s[0], end - 10)])
        c, d = np.searchsorted(time_s, [max(time_s[0], end - 3), end], side="left")
        if b > a and d > c:
            change = float(np.nanmedian(x[c:d]) - np.nanmedian(x[a:b]))
            if np.isfinite(change) and change != 0:
                signs.append(1 if change > 0 else -1)
                magnitudes.append(abs(change))
    if not signs and time_s.size >= 20:
        span = max(1, int(round(0.05 * time_s.size)))
        early = float(np.nanmedian(x[:span])); late = float(np.nanmedian(x[-span:]))
        if np.isfinite(early) and np.isfinite(late) and late != early:
            signs.append(1 if late > early else -1); magnitudes.append(abs(late - early))
    if not signs:
        raise RuntimeError(f"{volume['subject']}: unable to determine native Volume direction")
    vote = int(np.sign(np.sum(signs)))
    if vote == 0:
        vote = signs[int(np.argmax(magnitudes))]
    return vote


def estimate_native_volume_parameters(volume: Dict) -> Dict[str, float]:
    """Estimate direction, noise and a persistent-step threshold from this animal only."""
    x = np.asarray(volume["raw"], dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size < 10:
        raise RuntimeError("native Volume is too short for subject-adaptive estimation")
    direction = determine_volume_direction(volume)
    diff = np.diff(finite)
    med = float(np.median(diff)); noise = max(1.4826 * float(np.median(np.abs(diff - med))), 1e-6)
    span = float(np.nanpercentile(finite, 99) - np.nanpercentile(finite, 1))
    # Persistent steps are separated from native noise using only this
    # animal's robust derivative scale and total signal span.  The constants
    # are protocol multipliers, not subject-specific thresholds.
    threshold = max(4.0 * noise, 0.03 * span, 1e-6)
    return {"direction": int(direction), "noise_scale_ml": float(noise),
            "step_threshold_ml": float(threshold), "signal_span_ml": span}


def detect_native_urine_events(volume: Dict, cycle_rows, threshold_override: float | None = None) -> List[Dict]:
    """Detect persistent directed excursions at the original native sample rate."""
    subject, x, fs = volume["subject"], volume["raw"], volume["fs"]
    direction = determine_volume_direction(volume, cycle_rows)
    # External animals use a frozen population threshold supplied by the
    # validation stage.  The default path remains the original per-subject
    # frozen DSD threshold, so the Stage-A pipeline is unchanged.
    threshold = float(threshold_override) if threshold_override is not None else estimate_native_volume_parameters(volume)["step_threshold_ml"]
    events = []
    smooth_n = int(round(C.URINE_SMOOTH_S * fs))
    lookback_n = int(round(C.URINE_STEP_LOOKBACK_S * fs))
    hold_n = int(round(C.URINE_STEP_HOLD_S * fs))
    for _, row in cycle_rows.iterrows():
        cycle_id = str(row["dsd_cycle_id"])
        start_s, end_s = float(row["cycle_start_s"]), float(row["cycle_end_s"])
        time_s = np.asarray(volume.get("time_s", np.arange(x.size) / fs), dtype=float)
        i0, i1 = np.searchsorted(time_s, [start_s, end_s], side="left")
        raw = direction * x[i0:i1]
        valid = np.isfinite(raw)
        if raw.size < lookback_n + hold_n or valid.mean() < 0.99:
            continue
        fill = raw.copy()
        if not valid.all():
            good = np.flatnonzero(valid)
            fill[~valid] = np.interp(np.flatnonzero(~valid), good, fill[good])
        y = _causal_moving_average(fill, smooth_n)
        # Causal rolling minimum implemented in O(n), retaining native-rate detection.
        q = deque()
        rolling_min = np.empty(y.size, dtype=np.float64)
        for i, value in enumerate(y):
            while q and q[0] < i - lookback_n:
                q.popleft()
            while q and y[q[-1]] >= value:
                q.pop()
            q.append(i)
            rolling_min[i] = y[q[0]]
        excursion = y - rolling_min
        # A full-threshold crossing starts a candidate; a fixed half-threshold
        # hysteresis must then persist for the hold interval. This rejects a
        # single derivative spike while preserving quantized staircase signals.
        excursion[:lookback_n + smooth_n] = 0.0
        high = excursion >= threshold
        sustain = excursion >= threshold * C.URINE_STEP_SUSTAIN_RATIO
        segments = []
        i = 0
        while i < y.size:
            hits = np.flatnonzero(high[i:])
            if not hits.size:
                break
            a = i + int(hits[0]); b = a
            while b < y.size and sustain[b]:
                b += 1
            if b - a >= hold_n:
                segments.append((a, b))
            i = max(b + 1, a + 1)
        merged = []
        merge_n = int(round(C.URINE_STEP_MERGE_S * fs))
        for a, b in segments:
            if merged and a - merged[-1][1] <= merge_n:
                merged[-1] = (merged[-1][0], b)
            else:
                merged.append((a, b))
        for j, (a, b) in enumerate(merged, 1):
            search_a = max(0, a - lookback_n)
            prior_low = np.flatnonzero(excursion[search_a:a + 1] <= threshold * C.URINE_ONSET_BACKTRACK_RATIO)
            onset_i = search_a + int(prior_low[-1]) + 1 if prior_low.size else max(0, a - hold_n + 1)
            offset_i = min(y.size - 1, b - 1 + int(C.URINE_EVENT_TAIL_S * fs))
            events.append({
                "subject": subject, "cycle_id": cycle_id, "urine_event_id": f"{cycle_id}_U{j:02d}",
                "onset_s": float(time_s[i0 + onset_i]), "offset_s": float(time_s[i0 + offset_i]),
                "urine_present": 1, "direction": direction, "detection_valid": True,
                "threshold_ml": threshold, "native_fs_hz": fs,
                "step_size_ml": float(np.nanmax(excursion[a:offset_i + 1])) if offset_i >= a else np.nan,
            })
    return events


def causal_pressure_events(cycle: Dict) -> Tuple[np.ndarray, List[Dict], np.ndarray]:
    p = np.asarray(cycle["bladder_pressure_mmHg"], dtype=np.float64)
    valid = np.asarray(cycle["cmg_valid_100hz"], dtype=bool) & np.isfinite(p)
    valid &= (p >= C.CMG_VALID_RANGE_MMHG[0]) & (p <= C.CMG_VALID_RANGE_MMHG[1])
    fs = C.DP_FS_HZ
    baseline_n = int(C.BASELINE_WINDOW_S * fs)
    candidate_n = int(C.CANDIDATE_HOLD_S * fs)
    confirm_n = int(C.CONFIRM_HOLD_S * fs)
    recovery_n = int(C.RECOVERY_HOLD_S * fs)
    history = deque(maxlen=baseline_n)
    delta = np.full(p.size, np.nan, dtype=np.float64)
    baseline_ready = np.zeros(p.size, dtype=bool)
    state = "IDLE"
    start_i = None
    frozen_baseline = np.nan
    candidate_count = confirm_count = recovery_count = 0
    confirm_first = recovery_first = None
    peak_i = None
    events = []

    def finish(end_i, recovered, invalid=False):
        nonlocal state, start_i, candidate_count, confirm_count, recovery_count, confirm_first, recovery_first, peak_i
        if start_i is None:
            return
        peak = peak_i if peak_i is not None else start_i
        events.append({"start_index": start_i, "end_index": end_i, "confirm_index": confirm_first,
                       "peak_index": peak, "recovery_start_index": recovery_first,
                       "recovery_confirm_index": end_i if recovered else None,
                       "recovered": bool(recovered), "data_invalid": bool(invalid),
                       "peak_delta_p": float(delta[peak]) if np.isfinite(delta[peak]) else np.nan})
        state = "IDLE"; start_i = peak_i = confirm_first = recovery_first = None
        candidate_count = confirm_count = recovery_count = 0

    for i in range(p.size):
        if not valid[i]:
            if state != "IDLE":
                finish(max(i - 1, 0), False, invalid=True)
            continue
        if state == "IDLE":
            if len(history) >= baseline_n:
                base = float(np.median(np.asarray(history)))
                baseline_ready[i] = True
                delta[i] = p[i] - base
                if delta[i] > C.CANDIDATE_THRESHOLD_MMHG:
                    state = "CANDIDATE"; start_i = i; frozen_baseline = base
                    peak_i = i; candidate_count = 1
                else:
                    history.append(p[i])
            else:
                history.append(p[i])
        else:
            delta[i] = p[i] - frozen_baseline
            baseline_ready[i] = True
            if peak_i is None or delta[i] > delta[peak_i]:
                peak_i = i
            if state == "CANDIDATE":
                candidate_count = candidate_count + 1 if delta[i] > C.CANDIDATE_THRESHOLD_MMHG else 0
                if candidate_count >= candidate_n:
                    state = "ACTIVE"
                elif delta[i] <= C.RECOVERY_THRESHOLD_MMHG:
                    finish(i, True)
            if state in {"ACTIVE", "CONFIRMED"}:
                if state == "ACTIVE":
                    if delta[i] > C.CONFIRM_THRESHOLD_MMHG:
                        if confirm_count == 0:
                            confirm_first = i
                        confirm_count += 1
                        if confirm_count >= confirm_n:
                            state = "CONFIRMED"
                    else:
                        confirm_count = 0; confirm_first = None
                if delta[i] < C.RECOVERY_THRESHOLD_MMHG:
                    if recovery_count == 0:
                        recovery_first = i
                    recovery_count += 1
                    if recovery_count >= recovery_n:
                        finish(i, True)
                else:
                    recovery_count = 0; recovery_first = None
    if state != "IDLE":
        finish(p.size - 1, False, invalid=True)
    return delta, events, baseline_ready


def associate_and_label(subject: str, cycle_id: str, pressure_events: List[Dict], urine_events: List[Dict], t_abs: np.ndarray):
    cycle_urine = sorted([u for u in urine_events if u["subject"] == subject and u["cycle_id"] == cycle_id], key=lambda x: x["onset_s"])
    terminal = cycle_urine[-1] if cycle_urine else None
    matched = {}
    used_events, used_urine = set(), set()
    # Globally closest eligible peak/onset pairs prevent an early noise step
    # from consuming the pressure event that continuously develops to terminal void.
    pairs = []
    for ui, urine in enumerate(cycle_urine):
        for idx, event in enumerate(pressure_events):
            if event["confirm_index"] is None:
                continue
            start_s = float(t_abs[event["start_index"]])
            recovery_s = float(t_abs[event["recovery_confirm_index"]]) if event["recovery_confirm_index"] is not None else np.inf
            if start_s <= urine["onset_s"] and recovery_s >= urine["onset_s"]:
                pairs.append((abs(urine["onset_s"] - float(t_abs[event["peak_index"]])), idx, ui))
    for _, idx, ui in sorted(pairs):
        if idx not in used_events and ui not in used_urine:
            matched[idx] = cycle_urine[ui]; used_events.add(idx); used_urine.add(ui)
    terminal_matched = terminal is not None and any(u["urine_event_id"] == terminal["urine_event_id"] for u in matched.values())
    cycle_valid = bool(terminal_matched)
    labels = []
    for idx, event in enumerate(pressure_events):
        peak = event["peak_delta_p"]
        if not cycle_valid:
            label = "INVALID"
        elif event["confirm_index"] is None:
            label = "GREY_ZONE" if peak > C.CANDIDATE_THRESHOLD_MMHG else "INVALID"
        elif idx in matched:
            urine = matched[idx]
            confirm_s = float(t_abs[event["confirm_index"]])
            label = "VOID_CONFIRMED" if urine["onset_s"] <= confirm_s else "PREVOID_PROGRESSIVE"
        elif event["data_invalid"]:
            label = "INVALID"
        elif event["recovered"] and peak > C.CONFIRM_THRESHOLD_MMHG:
            label = "NVC_CORE"
        else:
            label = "INVALID"
        labels.append((label, matched.get(idx)))
    return labels, cycle_valid, terminal


def associate_adaptive_labels(subject: str, cycle_id: str, pressure_events: List[Dict], urine_events: List[Dict],
                              t_abs: np.ndarray):
    """One-to-one offline labeling; locally recovered events are never eligible for later urine."""
    cycle_urine = sorted([u for u in urine_events if u["subject"] == subject and u["cycle_id"] == cycle_id],
                         key=lambda x: x["onset_s"])
    terminal = cycle_urine[-1] if cycle_urine else None
    pairs = []
    for ui, urine in enumerate(cycle_urine):
        for ei, event in enumerate(pressure_events):
            if event["confirm_index"] is None:
                continue
            start_s = float(t_abs[event["start_index"]])
            recovery_s = (float(t_abs[event["recovery_confirm_index"]])
                          if event["recovery_confirm_index"] is not None else np.inf)
            # Recovery completed before urine makes the event permanently independent;
            # recovery after urine onset is the falling limb of the void contraction.
            if start_s <= urine["onset_s"] and recovery_s >= urine["onset_s"]:
                pairs.append((abs(urine["onset_s"] - float(t_abs[event["peak_index"]])), ei, ui))
    matched, used_e, used_u = {}, set(), set()
    for _, ei, ui in sorted(pairs):
        if ei not in used_e and ui not in used_u:
            matched[ei] = cycle_urine[ui]; used_e.add(ei); used_u.add(ui)
    terminal_matched = terminal is not None and any(
        u["urine_event_id"] == terminal["urine_event_id"] for u in matched.values())
    labels = []
    for ei, event in enumerate(pressure_events):
        urine = matched.get(ei)
        prom = event["local_prominence_mmHg"]
        if event["data_invalid"] or not terminal_matched:
            label, reason = "INVALID", "SIGNAL_INVALID" if event["data_invalid"] else "TERMINAL_URINE_UNMATCHED"
        elif urine is not None:
            confirm_s = float(t_abs[event["confirm_index"]])
            label = "VOID_CONFIRMED" if urine["onset_s"] <= confirm_s else "PREVOID_PROGRESSIVE"
            reason = ""
        elif event["locally_recovered"] and event["detection_level"] == "MAIN":
            label = "NVC_CORE" if prom > C.CONFIRM_THRESHOLD_MMHG else "NVC_ADAPTIVE"; reason = ""
        elif event["locally_recovered"] and event["detection_level"] == "POSSIBLE":
            label, reason = "NVC_POSSIBLE", "EXPLORATORY_ONLY"
        elif event["detection_level"] == "GREY":
            label, reason = "GREY_ZONE", "BELOW_ADAPTIVE_CONFIRM_OR_INCOMPLETE_MORPHOLOGY"
        else:
            label, reason = "INVALID", "UNRECOVERED_UNMATCHED_OR_TRUNCATED"
        labels.append((label, urine, reason))
    return labels, bool(terminal_matched), terminal
