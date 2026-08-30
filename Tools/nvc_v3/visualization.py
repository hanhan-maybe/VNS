"""Generate presentation-ready figures from the frozen V3 result tables."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"C0": "#8c8c8c", "P": "#4c78a8", "PE": "#f58518"}


def _short_model(value: str) -> str:
    return str(value).replace("PE_SPECTRAL_COMMON", "PE-SPEC").replace("PEF", "PEF")


def _read(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_summary(root: Path) -> dict:
    path = root / "v3_summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _comparison_from_summary(summary: dict) -> pd.DataFrame:
    """Recover the frozen model table when reorganized V3 keeps only JSON."""
    rows = []
    for model, metrics in summary.get("models", {}).items():
        row = {"model": model, **metrics}
        if metrics.get("n_events"):
            row["coverage"] = metrics.get("n_scorable", 0) / metrics["n_events"]
        rows.append(row)
    return pd.DataFrame(rows)


def _finish(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _heatmap(ax, frame: pd.DataFrame, value: str, title: str) -> None:
    pivot = frame.pivot(index="held_out_subject", columns="model", values=value)
    image = ax.imshow(pivot.to_numpy(float), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), [x.replace("STx", "") for x in pivot.index])
    ax.set_title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value_ij = pivot.iloc[i, j]
            ax.text(j, i, "NA" if pd.isna(value_ij) else f"{value_ij:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(image, ax=ax, fraction=.046, pad=.04)


def generate_plots(output_root: Path = Path("data/NVC_V3")) -> list[Path]:
    root = Path(output_root)
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    summary = _read_summary(root)
    comparison = _read(root, "model_comparison_v3.csv")
    if comparison.empty:
        comparison = _comparison_from_summary(summary)
    elif "coverage" not in comparison and {"n_scorable", "n_events"}.issubset(comparison):
        denominator = pd.to_numeric(comparison.n_events, errors="coerce").replace(0, np.nan)
        comparison["coverage"] = pd.to_numeric(comparison.n_scorable, errors="coerce") / denominator
    per_animal = _read(root, "per_animal_metrics_v3.csv")
    predictions = _read(root, "event_predictions_v3.csv")
    features = _read(root, "event_features_v3.csv")

    if not comparison.empty:
        metrics = [("pooled_AUROC", "Pooled AUROC", (0, 1)), ("pooled_AUPRC", "Pooled AUPRC", (0, 1)),
                   ("macro_sensitivity", "Animal-macro sensitivity", (0, 1)), ("PREVOID_FP_total", "PREVOID false positives", None)]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (column, title, ylim) in zip(axes.flat, metrics):
            values = pd.to_numeric(comparison[column], errors="coerce")
            labels = [_short_model(x) for x in comparison.model]
            ax.bar(labels, values, color=[COLORS.get(x, "#72b7b2") for x in comparison.model])
            ax.set_title(title); ax.grid(axis="y", alpha=.25); ax.tick_params(axis="x", rotation=20)
            if ylim: ax.set_ylim(*ylim)
            for i, value in enumerate(values):
                if pd.notna(value): ax.text(i, value, f"{value:.2f}" if value < 2 else f"{value:.0f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("V3 population LOSO model overview", fontsize=15, fontweight="bold")
        path = plots / "v3_model_overview.png"; _finish(fig, path); created.append(path)

        if {"macro_sensitivity", "PREVOID_FP_total"}.issubset(comparison):
            fig, ax = plt.subplots(figsize=(8.4, 5.8))
            x = pd.to_numeric(comparison.PREVOID_FP_total, errors="coerce")
            y = pd.to_numeric(comparison.macro_sensitivity, errors="coerce")
            coverage = pd.to_numeric(comparison["coverage"], errors="coerce").fillna(0) if "coverage" in comparison else pd.Series(1.0, index=comparison.index)
            ax.scatter(x, y, s=180 + 650 * coverage, c="#4c78a8", alpha=.78, edgecolor="white")
            for xx, yy, label in zip(x, y, comparison.model.astype(str)):
                ax.annotate(_short_model(label), (xx, yy), xytext=(6, 5), textcoords="offset points", fontsize=8)
            ax.set_xlabel("PREVOID false positives (lower is safer)")
            ax.set_ylabel("Animal-macro NVC sensitivity")
            ax.set_ylim(-.02, 1.02)
            ax.set_title("V3 recall-safety trade-off\n(marker size = feature coverage)", fontweight="bold")
            ax.grid(alpha=.25)
            path = plots / "v3_recall_safety_tradeoff.png"; _finish(fig, path); created.append(path)

        if "coverage" in comparison:
            fig, ax = plt.subplots(figsize=(10, 4.8))
            values = pd.to_numeric(comparison.coverage, errors="coerce")
            ax.bar([_short_model(x) for x in comparison.model.astype(str)], values, color="#72b7b2")
            ax.axhline(.95, color="#e45756", linestyle="--", linewidth=1.2, label="95% reference")
            ax.set_ylim(0, 1.08); ax.set_ylabel("Scorable coverage")
            ax.set_title("V3 feature availability by model", fontweight="bold")
            ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25); ax.legend()
            for i, value in enumerate(values):
                if pd.notna(value): ax.text(i, value + .02, f"{value:.0%}", ha="center", fontsize=8)
            path = plots / "v3_model_coverage.png"; _finish(fig, path); created.append(path)

    if not per_animal.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        _heatmap(axes[0], per_animal, "sensitivity", "NVC sensitivity by held-out animal")
        _heatmap(axes[1], per_animal, "specificity", "PREVOID specificity by held-out animal")
        fig.suptitle("V3 cross-animal generalization", fontsize=15, fontweight="bold")
        path = plots / "v3_per_animal_heatmap.png"; _finish(fig, path); created.append(path)

    if not predictions.empty and "p_nvc" in predictions:
        score = predictions[predictions.model_scorable.astype(str).str.lower().eq("true")].copy()
        score["p_nvc"] = pd.to_numeric(score.p_nvc, errors="coerce")
        labels = [x for x in ("NVC_CORE", "PREVOID_PROGRESSIVE") if x in set(score.teacher_label)]
        models = [x for x in ("C0", "P", "PE") if x in set(score.model)]
        if labels and models:
            fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.5), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, model in zip(axes, models):
                groups = [score[(score.model == model) & (score.teacher_label == label)].p_nvc.dropna() for label in labels]
                ax.boxplot(groups, labels=["NVC", "PREVOID"][:len(labels)], showfliers=False)
                ax.set_title(model); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
            axes[0].set_ylabel("LOSO NVC score")
            fig.suptitle("V3 score overlap explains limited separability", fontsize=14, fontweight="bold")
            path = plots / "v3_score_distribution.png"; _finish(fig, path); created.append(path)

    if not features.empty and {"subject", "teacher_label"}.issubset(features):
        count = features[features.teacher_label.isin(("NVC_CORE", "PREVOID_PROGRESSIVE"))].groupby(["subject", "teacher_label"]).size().unstack(fill_value=0)
        if not count.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            count.rename(columns={"NVC_CORE": "NVC", "PREVOID_PROGRESSIVE": "PREVOID"}).plot.bar(ax=ax, color=["#e45756", "#72b7b2"][:count.shape[1]])
            ax.set_title("V3 frozen teacher-label distribution by animal", fontweight="bold"); ax.set_ylabel("Events"); ax.set_xlabel("Animal"); ax.grid(axis="y", alpha=.25)
            path = plots / "v3_data_distribution.png"; _finish(fig, path); created.append(path)
    elif summary.get("nvc_counts_by_subject"):
        nvc = pd.Series(summary["nvc_counts_by_subject"], dtype=float)
        labels = pd.Series(summary.get("label_counts", {}), dtype=float)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        axes[0].bar([x.replace("STx", "") for x in nvc.index], nvc.values, color="#4c78a8")
        axes[0].set_title("Frozen NVC events by animal"); axes[0].set_ylabel("Events"); axes[0].grid(axis="y", alpha=.25)
        for i, value in enumerate(nvc.values): axes[0].text(i, value + .2, f"{int(value)}", ha="center", fontsize=8)
        shown = labels.reindex(["NVC_CORE", "PREVOID_PROGRESSIVE", "GREY_ZONE", "INVALID"]).dropna()
        axes[1].bar([x.replace("_PROGRESSIVE", "").replace("_CORE", "") for x in shown.index], shown.values,
                    color=["#4c78a8", "#f58518", "#bab0ac", "#e45756"][:len(shown)])
        axes[1].set_title("Frozen teacher-label composition"); axes[1].set_ylabel("Events"); axes[1].tick_params(axis="x", rotation=20); axes[1].grid(axis="y", alpha=.25)
        for i, value in enumerate(shown.values): axes[1].text(i, value + 1, f"{int(value)}", ha="center", fontsize=8)
        fig.suptitle("V3 development data: 8 animals, 67 cycles", fontsize=15, fontweight="bold")
        path = plots / "v3_data_distribution.png"; _finish(fig, path); created.append(path)

    (plots / "PLOT_INDEX.md").write_text(
        "# V3 图片索引\n\n"
        "- `v3_model_overview.png`：冻结模型的 AUROC、AUPRC、宏敏感度与 PREVOID 假阳性。\n"
        "- `v3_recall_safety_tradeoff.png`：NVC 召回与 PREVOID 安全性的权衡。\n"
        "- `v3_model_coverage.png`：各模型可评分覆盖率。\n"
        "- `v3_data_distribution.png`：8 只动物的 NVC 数量和全局标签组成。\n"
        "- `v3_per_animal_heatmap.png`：存在逐动物明细时生成的敏感度/特异度热图。\n"
        "- `v3_score_distribution.png`：存在预测明细时生成的 NVC/PREVOID 分数分布。\n",
        encoding="utf-8")
    return created


if __name__ == "__main__":
    generate_plots()
