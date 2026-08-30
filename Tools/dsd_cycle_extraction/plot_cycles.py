"""Quicklooks for stable cycles and whole-subject PRE_STIM selection."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MARKERS = (
    ("cycle_start_s", "cycle start", "0.25"),
    ("void_start_s", "void start", "tab:purple"),
    ("cmg_peak_s", "CMG peak", "tab:blue"),
    ("urine_output_onset_s", "urine onset", "tab:green"),
    ("void_end_s", "detected void end", "tab:orange"),
    ("cycle_end_s", "cycle end", "tab:red"),
)


def _plot_cumulative_drops(axis, start_s: float, end_s: float, drops: np.ndarray,
                           final_group: np.ndarray | None = None) -> None:
    drops = np.asarray(drops, dtype=np.float64)
    x = np.r_[start_s, drops, end_s]
    y = np.r_[0, np.arange(1, drops.size + 1), drops.size]
    axis.step(x, y, where="post", color="tab:green", lw=0.9,
              label="Channel 5 Leaks button: cumulative urine drops")
    if final_group is not None and len(final_group):
        ranks = np.searchsorted(drops, final_group, side="right")
        axis.scatter(final_group, ranks, color="tab:red", s=18, zorder=4,
                     label="final-void associated drops")
    axis.set_ylim(0.0, max(1.0, float(drops.size) + 0.5))
    axis.set_ylabel("Cumulative\nurine drops")
    axis.legend(fontsize=8, loc="upper left")


def plot_cycle_quicklook(path: Path, row: dict, aligned: dict) -> None:
    t = aligned["t_abs_s"]
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": (2.2, 1.5, 1.3, 1.0)})
    axes[0].plot(t, aligned["cmg_processed_100hz"], lw=0.75, color="tab:blue")
    axes[0].set_ylabel("CMG (mmHg)")
    axes[1].plot(t, aligned["eus_envelope_100hz"], lw=0.65, color="tab:orange")
    axes[1].set_ylabel("EUS envelope (mV)")

    urine_source = str(aligned["urine_source_type"])
    if urine_source == "CONTINUOUS_WEIGHT":
        axes[2].plot(t, aligned["urine_output_auxiliary_100hz"], lw=0.7, color="tab:green")
        axes[2].set_ylabel("Continuous weight\n(auxiliary)")
    else:
        drops = aligned["urine_event_times_abs_s"]
        final_group = drops[(drops >= float(row["void_start_s"]) - 1.0)
                            & (drops <= float(row["void_end_s"]) + 5.0)]
        _plot_cumulative_drops(
            axes[2], float(row["cycle_start_s"]), float(row["cycle_end_s"]),
            drops, final_group,
        )

    axes[3].set_ylim(-0.5, len(MARKERS) - 0.5)
    axes[3].set_yticks(range(len(MARKERS)), [label for _, label, _ in MARKERS])
    for level, (field, label, color) in enumerate(MARKERS):
        value = float(row[field])
        axes[3].scatter([value], [level], s=34, color=color, zorder=3)
        axes[3].axvline(value, color=color, ls="--", lw=0.7, alpha=0.8, label=label)
    axes[3].set_ylabel("Events")
    axes[3].set_xlabel("Absolute PRE_STIM time (s)")

    for axis in axes[:3]:
        for field, _, color in MARKERS:
            axis.axvline(float(row[field]), color=color, ls="--", lw=0.55, alpha=0.65)
    axes[0].legend(*axes[3].get_legend_handles_labels(), ncol=3, fontsize=8, loc="upper left")
    axes[-1].set_xlim(float(row["cycle_start_s"]), float(row["cycle_end_s"]))
    fig.suptitle(
        f"{row['subject']} {row['dsd_cycle_id']} | global={row['global_cycle_id']} | "
        f"duration={float(row['cycle_duration_s']):.2f} s | PASS_STABLE | Urine source={urine_source}"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_stable_overview(path: Path, result: dict) -> None:
    cache = result["cache"]
    cycles = result["cycles"]
    time_s = cache["time_100hz"]
    pressure = cache["pressure_100hz"]
    first_stim_s = float(cache["first_stim_s"])
    fig, axes = plt.subplots(2, 1, figsize=(17, 8), sharex=True, constrained_layout=True)
    axes[0].plot(time_s, pressure, lw=0.5, color="tab:blue")
    axes[0].set_ylabel("CMG (mmHg)")

    urine = cache["urine"]
    if urine["source_type"] == "CONTINUOUS_WEIGHT":
        stride = max(1, len(urine["time_s"]) // 120000)
        axes[1].plot(urine["time_s"][::stride], urine["trace"][::stride], lw=0.55, color="tab:green")
        axes[1].set_ylabel("Continuous weight\n(auxiliary)")
    else:
        _plot_cumulative_drops(axes[1], 0.0, first_stim_s, urine["drop_times"])

    first_index = result["first_stable_index"]
    if first_index is not None:
        stable_start = float(cycles[first_index]["cycle_start_s"])
        for axis in axes:
            axis.axvspan(0, stable_start, color="0.75", alpha=0.25, label="setup/acclimation")
            axis.axvline(stable_start, color="tab:green", ls="--", lw=1.2, label="first stable cycle")

    status_colors = {
        "PASS_STABLE": "tab:green",
        "EXCLUDE_ACCLIMATION": "0.55",
        "EXCLUDE_INCOMPLETE": "tab:gray",
        "EXCLUDE_PRESSURE_ARTIFACT": "tab:red",
        "EXCLUDE_TRANSITIONAL": "tab:orange",
        "EXCLUDE_PRE_STIM_BOUNDARY": "tab:red",
        "REVIEW_REQUIRED": "tab:purple",
    }
    for row in cycles:
        color = status_colors.get(row["cycle_status"], "0.4")
        start = row["cycle_start_s"]
        end = row["cycle_end_s"]
        if np.isfinite(float(start)):
            for axis in axes:
                axis.axvspan(float(start), float(end), color=color, alpha=0.10)
        axes[0].axvline(float(row["cmg_peak_s"]), color=color, lw=0.6, alpha=0.75)
        axes[0].text(float(row["cmg_peak_s"]), float(row["cmg_peak_pressure"]),
                     row["global_cycle_id"], fontsize=7, ha="center", va="bottom", color=color)
        if urine["source_type"] == "LEAK_BUTTON_EVENT":
            onset = float(row["urine_output_onset_s"])
            rank = np.searchsorted(urine["drop_times"], onset, side="right")
            axes[1].scatter([onset], [rank], color="tab:red", s=16,
                            zorder=4, label="confirmed void group onset")

    for axis in axes:
        axis.axvline(first_stim_s, color="black", ls="--", lw=1.2, label="first_stim_s")
        axis.set_xlim(0, first_stim_s)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), ncol=3, fontsize=8, loc="upper left")
    axes[1].set_xlabel("PRE_STIM time (s)")
    axes[0].set_title(
        f"{result['subject']} stable-cycle overview | Urine source={urine['source_type']} | "
        "green=PASS, gray=acclimation/incomplete, orange=transitional, red=artifact/boundary"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)
