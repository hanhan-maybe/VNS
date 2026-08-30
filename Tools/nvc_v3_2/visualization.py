"""Generate presentation-ready figures from V3.2 parallel-model outputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short(value: str) -> str:
    return str(value).replace("B0-primary","B0").replace("M1_P_SPEC_SHORT","M1 P-SPEC").replace("M2_PE_EUS_STFT_SPARSE","M2 EUS-STFT").replace("M3_PE_TF_COMPACT_LR","M3 TF-LR").replace("M5_PE_TF_COMPACT_SVM","M5 TF-SVM").replace("M4_EVENT_PROGRESSION_GUARD","M4 GUARD")


def _finish(fig,path): fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)


def generate_plots(output_root: Path=Path("data/NVC_V3_2")) -> list[Path]:
    root=Path(output_root); plots=root/"plots"; plots.mkdir(parents=True,exist_ok=True); made=[]
    comp=pd.read_csv(root/"v32_primary_model_comparison.csv") if (root/"v32_primary_model_comparison.csv").exists() else pd.DataFrame()
    per=pd.read_csv(root/"v32_per_animal_metrics.csv") if (root/"v32_per_animal_metrics.csv").exists() else pd.DataFrame()
    coverage=pd.read_csv(root/"v32_model_coverage.csv") if (root/"v32_model_coverage.csv").exists() else pd.DataFrame()
    classifier=pd.read_csv(root/"m3_vs_m5_classifier_comparison.csv") if (root/"m3_vs_m5_classifier_comparison.csv").exists() else pd.DataFrame()
    if not comp.empty:
        labels=[_short(x) for x in comp.model]
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for ax,col,title in zip(axes.flat,("AUROC","AUPRC","animal_macro_frozen_sensitivity","PREVOID_FPR"),("AUROC","AUPRC","Animal-macro NVC sensitivity","PREVOID false-positive rate")):
            vals=pd.to_numeric(comp[col],errors="coerce"); ax.bar(range(len(vals)),vals,color="#4c78a8"); ax.set_xticks(range(len(vals)),labels,rotation=30,ha="right")
            maximum=float(vals.max()) if vals.notna().any() else 0.0
            ax.set_ylim(0,1 if col in ("AUROC","AUPRC") else max(.1,maximum*1.35)); ax.set_title(title); ax.grid(axis="y",alpha=.25)
            for i,value in enumerate(vals):
                ypos = .015 if pd.isna(value) or value == 0 else value + .015
                ax.text(i, ypos, "NA" if pd.isna(value) else f"{value:.2f}", ha="center", va="bottom", fontsize=8)
            if maximum == 0:
                ax.text(.5,.72,"No NVC detected at frozen thresholds",transform=ax.transAxes,ha="center",fontsize=10,color="#7a7a7a")
        fig.suptitle("V3.2 five parallel mechanism models",fontsize=15,fontweight="bold"); p=plots/"v32_parallel_model_overview.png"; _finish(fig,p); made.append(p)
        fig,ax=plt.subplots(figsize=(7,5.5)); x=pd.to_numeric(comp.PREVOID_FPR,errors="coerce"); y=pd.to_numeric(comp.animal_macro_frozen_sensitivity,errors="coerce"); s=200+800*pd.to_numeric(comp.coverage,errors="coerce").fillna(0)
        ax.scatter(x,y,s=s,c=np.arange(len(comp)),cmap="viridis",alpha=.8,edgecolor="white")
        for xx,yy,label in zip(x,y,labels): ax.annotate(label,(xx,yy),xytext=(5,5),textcoords="offset points",fontsize=8)
        ax.set_xlim(left=-.02); ax.set_ylim(-.02,1); ax.set_xlabel("PREVOID FPR (lower is safer)"); ax.set_ylabel("Macro NVC sensitivity"); ax.set_title("V3.2 recall-safety Pareto view",fontweight="bold"); ax.grid(alpha=.25)
        p=plots/"v32_recall_safety_pareto.png"; _finish(fig,p); made.append(p)
    if not per.empty:
        pivot=per.pivot(index="animal",columns="model",values="frozen_sensitivity")
        fig,ax=plt.subplots(figsize=(13,5.5)); im=ax.imshow(pivot.to_numpy(float),vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        ax.set_xticks(range(len(pivot.columns)),[_short(x) for x in pivot.columns],rotation=30,ha="right"); ax.set_yticks(range(len(pivot.index)),[x.replace("STx","") for x in pivot.index]); ax.set_title("V3.2 per-animal NVC sensitivity",fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v=pivot.iloc[i,j]; ax.text(j,i,"NA" if pd.isna(v) else f"{v:.2f}",ha="center",va="center",fontsize=8)
        plt.colorbar(im,ax=ax,fraction=.025,pad=.02); p=plots/"v32_per_animal_heatmap.png"; _finish(fig,p); made.append(p)
    if not coverage.empty:
        fig,ax=plt.subplots(figsize=(10,4.8)); vals=pd.to_numeric(coverage.coverage,errors="coerce"); labels=[_short(x) for x in coverage.model]
        ax.bar(labels,vals,color="#72b7b2"); ax.axhline(.95,color="#e45756",ls="--",label="95% target"); ax.set_ylim(0,1.05); ax.set_ylabel("Coverage"); ax.set_title("V3.2 structural feature coverage",fontweight="bold"); ax.tick_params(axis="x",rotation=25); ax.legend(); ax.grid(axis="y",alpha=.25)
        p=plots/"v32_model_coverage.png"; _finish(fig,p); made.append(p)
    if not classifier.empty:
        labels=[_short(x) for x in classifier.model]; x=np.arange(len(classifier)); width=.25
        fig,ax=plt.subplots(figsize=(8.4,5.2))
        for offset,column,label,color in ((-width,"AUROC","AUROC","#4c78a8"),(0,"AUPRC","AUPRC","#f58518"),(width,"animal_macro_frozen_sensitivity","Macro sensitivity","#54a24b")):
            values=pd.to_numeric(classifier[column],errors="coerce")
            ax.bar(x+offset,values,width,label=label,color=color)
            for i,value in enumerate(values):
                if pd.notna(value): ax.text(i+offset,value+.02,f"{value:.2f}",ha="center",fontsize=8)
        ax.set_xticks(x,labels); ax.set_ylim(0,1.05); ax.set_ylabel("Metric value"); ax.set_title("V3.2 same features: LR versus Linear SVM",fontweight="bold"); ax.legend(); ax.grid(axis="y",alpha=.25)
        p=plots/"v32_m3_m5_classifier_comparison.png"; _finish(fig,p); made.append(p)
    (plots/"PLOT_INDEX.md").write_text("# V3.2 图片索引\n\n- `v32_parallel_model_overview.png`：五个并列模型总体指标。\n- `v32_recall_safety_pareto.png`：召回—PREVOID 安全权衡。\n- `v32_per_animal_heatmap.png`：跨动物敏感度。\n- `v32_model_coverage.png`：模型结构性覆盖率。\n- `v32_m3_m5_classifier_comparison.png`：同一紧凑特征下 LR 与 Linear SVM 的直接对比。\n",encoding="utf-8")
    return made


if __name__=="__main__": generate_plots()
