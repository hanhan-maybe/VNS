"""Generate presentation-ready figures from V4 NVC feature-learning outputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short(x):
    return str(x).replace("P0_anchor","P0").replace("P1_morphology_dynamics","P1").replace("M1_P_NVC","M1 P").replace("E0_time","E0").replace("M2_E_NVC","M2 EUS").replace("E2_slow_modulation","E2 slow").replace("M3_PE_NVC_LATE_FUSION","M3 PE-LR").replace("M4_PE_NVC_SHRINKAGE_LDA","M4 PE-LDA")


def _finish(fig,path): fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)


def generate_plots(output_root: Path=Path("data/NVC_V4")) -> list[Path]:
    root=Path(output_root); plots=root/"plots"; plots.mkdir(parents=True,exist_ok=True); made=[]
    comp=pd.read_csv(root/"v4_model_comparison_full_coverage.csv") if (root/"v4_model_comparison_full_coverage.csv").exists() else pd.DataFrame()
    per=pd.read_csv(root/"v4_per_animal_metrics.csv") if (root/"v4_per_animal_metrics.csv").exists() else pd.DataFrame()
    challenge=pd.read_csv(root/"v4_challenge_summary.csv") if (root/"v4_challenge_summary.csv").exists() else pd.DataFrame()
    stability=pd.read_csv(root/"v4_feature_stability.csv") if (root/"v4_feature_stability.csv").exists() else pd.DataFrame()
    dataset=pd.read_csv(root/"v4_dataset_summary.csv") if (root/"v4_dataset_summary.csv").exists() else pd.DataFrame()
    if not comp.empty:
        labels=[_short(x) for x in comp.model]
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for ax,col,title in zip(axes.flat,("AUROC","AUPRC","pooled_sensitivity","coverage"),("NVC vs stable AUROC","NVC vs stable AUPRC","Pooled NVC sensitivity","Feature coverage")):
            vals=pd.to_numeric(comp[col],errors="coerce"); ax.bar(range(len(vals)),vals,color="#4c78a8"); ax.set_xticks(range(len(vals)),labels,rotation=30,ha="right"); ax.set_ylim(0,1.08); ax.set_title(title); ax.grid(axis="y",alpha=.25)
            for i,value in enumerate(vals):
                ypos = .015 if pd.isna(value) or value == 0 else value + .015
                ax.text(i, ypos, "NA" if pd.isna(value) else f"{value:.2f}", ha="center", va="bottom", fontsize=8)
        fig.suptitle("V4 task redesign: NVC feature learning",fontsize=15,fontweight="bold"); p=plots/"v4_model_overview.png"; _finish(fig,p); made.append(p)
    if not per.empty:
        pivot=per.pivot(index="animal",columns="model",values="frozen_sensitivity")
        fig,ax=plt.subplots(figsize=(14,5.5)); im=ax.imshow(pivot.to_numpy(float),vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        ax.set_xticks(range(len(pivot.columns)),[_short(x) for x in pivot.columns],rotation=30,ha="right"); ax.set_yticks(range(len(pivot.index)),[x.replace("STx","") for x in pivot.index]); ax.set_title("V4 cross-animal NVC sensitivity and phenotype heterogeneity",fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v=pivot.iloc[i,j]; ax.text(j,i,"NA" if pd.isna(v) else f"{v:.2f}",ha="center",va="center",fontsize=7)
        plt.colorbar(im,ax=ax,fraction=.025,pad=.02); p=plots/"v4_per_animal_heatmap.png"; _finish(fig,p); made.append(p)
    if not challenge.empty:
        summary=challenge.groupby(["model","challenge_type"],as_index=False).fraction_above_threshold.mean()
        pivot=summary.pivot(index="model",columns="challenge_type",values="fraction_above_threshold").fillna(0)
        fig,ax=plt.subplots(figsize=(11,5)); pivot.index=[_short(x) for x in pivot.index]; pivot.plot.bar(ax=ax,color=["#f58518","#e45756"][:pivot.shape[1]])
        ax.set_ylim(0,1); ax.set_ylabel("Fraction above frozen threshold"); ax.set_title("V4 post-freeze PREVOID/VOID challenge replay",fontweight="bold"); ax.grid(axis="y",alpha=.25)
        p=plots/"v4_challenge_trigger_fraction.png"; _finish(fig,p); made.append(p)
    if not stability.empty:
        one=stability.drop_duplicates("feature").copy(); one["abs_effect"]=pd.to_numeric(one.median_effect_size,errors="coerce").abs(); one=one.sort_values("abs_effect",ascending=False).head(15).sort_values("abs_effect")
        fig,ax=plt.subplots(figsize=(9,6)); colors=np.where(pd.to_numeric(one.direction_consistency,errors="coerce")>=.75,"#54a24b","#bab0ac")
        ax.barh(one.feature,one.abs_effect,color=colors); ax.set_xlabel("Absolute median effect size"); ax.set_title("V4 top reproducible NVC features\n(green: direction consistency >= 75%)",fontweight="bold"); ax.grid(axis="x",alpha=.25)
        p=plots/"v4_feature_stability.png"; _finish(fig,p); made.append(p)
    if not dataset.empty:
        labels=dataset.dataset.astype(str); x=np.arange(len(dataset)); width=.36
        fig,axes=plt.subplots(1,2,figsize=(11.5,4.8))
        axes[0].bar(x-width/2,pd.to_numeric(dataset.NVC_CORE,errors="coerce"),width,label="NVC",color="#4c78a8")
        axes[0].bar(x+width/2,pd.to_numeric(dataset.STABLE_FILLING,errors="coerce"),width,label="Stable filling",color="#72b7b2")
        axes[0].set_xticks(x,labels); axes[0].set_ylabel("Training samples"); axes[0].set_title("Matched primary-task samples"); axes[0].legend(); axes[0].grid(axis="y",alpha=.25)
        axes[1].bar(x-width/2,pd.to_numeric(dataset.animals,errors="coerce"),width,label="Animals",color="#f58518")
        axes[1].bar(x+width/2,pd.to_numeric(dataset.cycles,errors="coerce"),width,label="Cycles",color="#bab0ac")
        axes[1].set_xticks(x,labels); axes[1].set_title("Development cohort composition"); axes[1].legend(); axes[1].grid(axis="y",alpha=.25)
        fig.suptitle("V4 changes the primary task to NVC versus stable filling",fontsize=14,fontweight="bold")
        p=plots/"v4_dataset_design.png"; _finish(fig,p); made.append(p)
    (plots/"PLOT_INDEX.md").write_text("# V4 图片索引\n\n- `v4_model_overview.png`：NVC vs Stable Filling 模型总体指标。\n- `v4_per_animal_heatmap.png`：跨动物敏感度和表型差异。\n- `v4_challenge_trigger_fraction.png`：冻结后 PREVOID/VOID challenge 回放。\n- `v4_feature_stability.png`：跨动物方向较稳定的特征。\n- `v4_dataset_design.png`：338/164 数据组成与 NVC/Stable Filling 主任务样本。\n",encoding="utf-8")
    return made


if __name__=="__main__": generate_plots()
