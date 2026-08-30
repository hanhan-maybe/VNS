"""One non-analytic quicklook per cycle."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C


def plot_cycle(path: Path, cycle, delta, eus_env, urine_rows, pressure_rows, replay_rows, adaptive=None, sweep_rows=None):
    t = np.asarray(cycle["t_abs_s"])
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    axes[0].plot(t, delta, lw=.8, color="black", label="causal delta_p")
    for value, color, name in [(C.CANDIDATE_THRESHOLD_MMHG, "#d99b00", "candidate"),
                               (C.CONFIRM_THRESHOLD_MMHG, "#d62728", "confirm"),
                               (C.RECOVERY_THRESHOLD_MMHG, "#2ca02c", "recovery")]:
        axes[0].axhline(value, color=color, ls="--", lw=.8, label=name)
    if adaptive is not None:
        axes[0].plot(t, adaptive["adaptive_start"], color="#f2b134", lw=.7, label="adaptive start")
        axes[0].plot(t, adaptive["adaptive_confirm"], color="#b2182b", lw=.8, label="adaptive confirm")
        axes[0].plot(t, adaptive["adaptive_recovery"], color="#1b7837", lw=.7, label="adaptive recovery")
    colors = {"NVC_CORE": "#2ca02c", "PREVOID_PROGRESSIVE": "#ff7f0e", "VOID_CONFIRMED": "#d62728",
              "NVC_ADAPTIVE": "#17becf", "NVC_POSSIBLE": "#9467bd", "GREY_ZONE": "#999999", "INVALID": "#555555"}
    for _, row in pressure_rows.iterrows():
        axes[0].axvspan(row.start_s, row.end_s, color=colors.get(row.teacher_label, "#999999"), alpha=.16)
        if np.isfinite(row.confirm_time_s): axes[0].axvline(row.confirm_time_s, color=colors.get(row.teacher_label), lw=.8)
        if np.isfinite(row.recovery_confirm_s): axes[0].axvline(row.recovery_confirm_s, color="#2ca02c", lw=.5, ls=":")
        if np.isfinite(row.local_trough_time_s): axes[0].scatter(row.local_trough_time_s, 0, marker="v", s=14, color="#2166ac")
        if np.isfinite(row.local_peak_time_s): axes[0].scatter(row.local_peak_time_s, row.local_prominence_mmHg, marker="^", s=16, color=colors.get(row.teacher_label))
        if "original_event_start_s" in row and np.isfinite(row.original_event_start_s):
            axes[0].axvspan(row.original_event_start_s, row.original_event_end_s, facecolor="none", edgecolor="#444444", lw=.4, hatch="//")
    axes[0].legend(ncol=4, fontsize=7); axes[0].set_ylabel("delta_p (mmHg)")
    axes[1].plot(t, eus_env, color="#6a3d9a", lw=.6); axes[1].set_ylabel("causal EUS env")
    urine_binary = np.zeros(t.size)
    for _, row in urine_rows.iterrows(): urine_binary[(t >= row.onset_s) & (t <= row.offset_s)] = 1
    axes[2].step(t, urine_binary, where="post", color="#1f77b4", label="native Volume event")
    marker = {"M0": "x", "M0A": "s", "M1": "o", "M2": "^"}
    if replay_rows is not None and not replay_rows.empty and "trigger" in replay_rows.columns:
        for model, group in replay_rows[replay_rows.trigger].groupby("model"):
            axes[2].scatter(group.confirm_time_s, np.full(len(group), 1.05), marker=marker.get(model, "o"), s=35, label=f"{model} trigger")
    if sweep_rows is not None and not sweep_rows.empty:
        for model, group in sweep_rows.groupby("model"):
            for _, row in group.iterrows():
                y = .35 + min(.55, max(0., float(row.p_void_risk)))
                color = "#d62728" if row.trigger else "#444444"
                axes[2].scatter(row.decision_time_s, y, marker=marker.get(model, "o"), s=12, color=color, alpha=.7)
    axes[2].set_ylim(-.1, 1.25); axes[2].set_ylabel("urine / trigger"); axes[2].set_xlabel("absolute time (s)")
    axes[2].legend(fontsize=7, ncol=4)
    fig.suptitle(f"{cycle['subject'].item()}/{cycle['dsd_cycle_id'].item()} | native Volume labels; quicklook not used for training")
    fig.savefig(path, dpi=130); plt.close(fig)
