"""Generate summary figures for V3.1 without importing another version."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short(value):
    return (str(value).replace("CANDIDATE+VOIDGUARD", "DELAY+GUARD")
            .replace("PE_TRAJECTORY", "PE TRAJ").replace("PE_DELAY", "PE DELAY"))


def _finish(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)


def generate_plots(output_root: Path = Path("data/NVC_V3_1")) -> list[Path]:
    root=Path(output_root); plots=root/"plots"; plots.mkdir(parents=True,exist_ok=True); made=[]
    comp=pd.read_csv(root/"model_comparison_v31.csv") if (root/"model_comparison_v31.csv").exists() else pd.DataFrame()
    animals=pd.read_csv(root/"per_animal_model_comparison_v31.csv") if (root/"per_animal_model_comparison_v31.csv").exists() else pd.DataFrame()
    delay=pd.read_csv(root/"delay_diagnostic_metrics.csv") if (root/"delay_diagnostic_metrics.csv").exists() else pd.DataFrame()
    if not comp.empty:
        fig,axes=plt.subplots(1,3,figsize=(15,4.8))
        models=comp.model.astype(str); display=[_short(x) for x in models]; colors=["#4c78a8" if x=="PE" else "#9ecae9" for x in models]
        for ax,col,title in zip(axes,("animal_macro_frozen_sensitivity","PREVOID_FPR","scorable_coverage"),("Macro NVC sensitivity","PREVOID false-positive rate","Feature coverage")):
            vals=pd.to_numeric(comp[col],errors="coerce"); ax.bar(range(len(vals)),vals,color=colors); ax.set_xticks(range(len(vals)),display,rotation=30,ha="right"); ax.set_ylim(0,max(1,float(vals.max())*1.15)); ax.set_title(title); ax.grid(axis="y",alpha=.25)
        fig.suptitle("V3.1 mechanism-guided candidates: recall-safety conflict",fontsize=14,fontweight="bold")
        p=plots/"v31_summary_dashboard.png"; _finish(fig,p); made.append(p)
        fig,ax=plt.subplots(figsize=(8.2,5.8))
        x=pd.to_numeric(comp.PREVOID_FPR,errors="coerce"); y=pd.to_numeric(comp.animal_macro_frozen_sensitivity,errors="coerce")
        coverage=pd.to_numeric(comp.scorable_coverage,errors="coerce").fillna(0)
        ax.scatter(x,y,s=180+650*coverage,c="#4c78a8",alpha=.78,edgecolor="white")
        offsets={"C0":(8,10),"P":(8,-18),"PE":(8,10),"PE_TRAJECTORY":(10,14),
                 "PE_DELAY":(10,18),"CANDIDATE+VOIDGUARD":(10,-28)}
        for xx,yy,label in zip(x,y,models):
            ax.annotate(_short(label),(xx,yy),xytext=offsets.get(str(label),(7,8)),textcoords="offset points",fontsize=8,
                        arrowprops={"arrowstyle":"-","color":"#777777","lw":.6})
        ax.set_xlim(-.03,1.03); ax.set_ylim(-.03,1.03); ax.set_xlabel("PREVOID false-positive rate (lower is safer)"); ax.set_ylabel("Animal-macro NVC sensitivity")
        ax.set_title("V3.1 no candidate moves the safe Pareto frontier",fontweight="bold"); ax.grid(alpha=.25)
        p=plots/"v31_recall_safety_pareto.png"; _finish(fig,p); made.append(p)
    if not animals.empty:
        pivot=animals.pivot(index="animal",columns="model",values="frozen_sensitivity")
        fig,ax=plt.subplots(figsize=(12,5.5)); im=ax.imshow(pivot.to_numpy(float),vmin=0,vmax=1,cmap="YlOrRd",aspect="auto")
        ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha="right"); ax.set_yticks(range(len(pivot.index)),[x.replace("STx","") for x in pivot.index]); ax.set_title("V3.1 NVC sensitivity remains animal-dependent",fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v=pivot.iloc[i,j]; ax.text(j,i,"NA" if pd.isna(v) else f"{v:.2f}",ha="center",va="center",fontsize=8)
        plt.colorbar(im,ax=ax,fraction=.025,pad=.02); p=plots/"v31_per_animal_heatmap.png"; _finish(fig,p); made.append(p)
    if not delay.empty:
        fig,axes=plt.subplots(1,2,figsize=(12.5,4.8))
        for model,color in (("PE","#4c78a8"),("PE_DELAY","#f58518"),("PE_TRAJECTORY","#54a24b")):
            group=delay[delay.model.astype(str).eq(model)].sort_values("delay")
            if group.empty: continue
            xx=pd.to_numeric(group.delay,errors="coerce")
            axes[0].plot(xx,pd.to_numeric(group.animal_macro_sensitivity,errors="coerce"),marker="o",label=model,color=color)
            axes[1].plot(xx,pd.to_numeric(group.PREVOID_FPR,errors="coerce"),marker="o",label=model,color=color)
        axes[0].set_title("Macro NVC sensitivity"); axes[1].set_title("PREVOID false-positive rate")
        for ax in axes: ax.set_xlabel("Decision delay (s)"); ax.set_ylim(-.03,1.03); ax.grid(alpha=.25)
        axes[0].set_ylabel("Rate"); axes[1].legend(loc="best")
        fig.suptitle("V3.1 waiting longer raises recall and unsafe PREVOID triggers together",fontsize=14,fontweight="bold")
        p=plots/"v31_delay_tradeoff.png"; _finish(fig,p); made.append(p)
    (plots/"PLOT_INDEX.md").write_text("# V3.1 图片索引\n\n- `v31_summary_dashboard.png`：候选模型召回、安全性和覆盖率总览。\n- `v31_recall_safety_pareto.png`：候选模型的召回—安全 Pareto 图。\n- `v31_per_animal_heatmap.png`：逐动物 NVC 敏感度。\n- `v31_delay_tradeoff.png`：延迟增加对召回和 PREVOID 假阳性的同步影响。\n- 其余图片为原 V3.1 延迟、表型和机制诊断图。\n",encoding="utf-8")
    return made


if __name__=="__main__": generate_plots()
