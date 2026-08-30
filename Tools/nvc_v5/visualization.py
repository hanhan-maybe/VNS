"""Generate V5 individualized-model and final replay presentation figures."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_LABELS={"R0":"R0 reference","M1":"M1 P-EARLY","M2":"M2 E-EARLY","M3":"M3 PE-EARLY","M4":"M4 EUS-SP-LASSO"}


def _finish(fig,path): fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches="tight",facecolor="white"); plt.close(fig)


def _read(path): return pd.read_csv(path) if path.exists() else pd.DataFrame()


def generate_plots(output_root: Path=Path("data/NVC_V5")) -> list[Path]:
    root=Path(output_root); plots=root/"plots"; final_plots=plots/"final_validation"; plots.mkdir(parents=True,exist_ok=True); final_plots.mkdir(parents=True,exist_ok=True); made=[]
    results=_read(root/"v5_parallel_model_results.csv")
    streaming=_read(root/"v5_parallel_streaming_summary.csv")
    timing=_read(root/"v5_parallel_timing.csv")
    final=_read(root/"v5_final_validation"/"m1_t0_t1_comparison.csv")
    runtime=_read(root/"v5_final_validation"/"m1_old_vs_corrected_runtime.csv")
    latency=_read(root/"v5_final_validation"/"m1_nvc_event_latency.csv")
    per_cycle=_read(root/"v5_parallel_per_cycle_metrics.csv")
    if not results.empty:
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        metrics=(("sensitivity","Event sensitivity",(0,1.05)),("coverage","Feature coverage",(0,1.05)),("fp_per_cycle","Stable-window FP/cycle",None),("ppv","Positive predictive value",(0,1.05)))
        x=np.arange(results.model.nunique()); width=.36
        models=list(dict.fromkeys(results.model));
        for ax,(col,title,ylim) in zip(axes.flat,metrics):
            for k,(animal,color) in enumerate((("STxF37","#4c78a8"),("STxF26","#f58518"))):
                g=results[results.animal.eq(animal)].set_index("model").reindex(models); vals=pd.to_numeric(g[col],errors="coerce")
                ax.bar(x+(k-.5)*width,vals,width,label=animal.replace("STx",""),color=color)
            ax.set_xticks(x,[MODEL_LABELS.get(m,m) for m in models],rotation=25,ha="right"); ax.set_title(title); ax.grid(axis="y",alpha=.25)
            if ylim: ax.set_ylim(*ylim)
        axes[0,0].legend(title="Animal"); fig.suptitle("V5 five-model individualized prospective comparison",fontsize=15,fontweight="bold")
        p=plots/"v5_parallel_model_overview.png"; _finish(fig,p); made.append(p)
    if not streaming.empty:
        agg=streaming.groupby(["animal","model"],as_index=False)[["t0_false_triggers","t1_false_triggers"]].sum()
        labels=[f"{a.replace('STx','')}\n{MODEL_LABELS.get(m,m)}" for a,m in zip(agg.animal,agg.model)]; x=np.arange(len(agg)); width=.38
        fig,ax=plt.subplots(figsize=(13,5)); ax.bar(x-width/2,agg.t0_false_triggers,width,label="T0",color="#e45756"); ax.bar(x+width/2,agg.t1_false_triggers,width,label="T1",color="#72b7b2")
        ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylabel("Full-cycle false triggers"); ax.set_title("V5 continuous replay: temporal confirmation effect",fontweight="bold"); ax.legend(); ax.grid(axis="y",alpha=.25)
        p=plots/"v5_streaming_false_triggers.png"; _finish(fig,p); made.append(p)
    if not timing.empty:
        timing=timing.copy(); timing["t0_latency"]=pd.to_numeric(timing.t0_first_crossing_s,errors="coerce")-pd.to_numeric(timing.confirm_time_s,errors="coerce"); timing["t1_latency"]=pd.to_numeric(timing.t1_first_crossing_s,errors="coerce")-pd.to_numeric(timing.confirm_time_s,errors="coerce")
        rows=[]
        for (animal,model),g in timing.groupby(["animal","model"]): rows.append({"animal":animal,"model":model,"T0":g.t0_latency.median(),"T1":g.t1_latency.median()})
        med=pd.DataFrame(rows)
        if not med.empty:
            labels=[f"{a.replace('STx','')}\n{MODEL_LABELS.get(m,m)}" for a,m in zip(med.animal,med.model)]; x=np.arange(len(med)); width=.38
            fig,ax=plt.subplots(figsize=(13,5)); ax.bar(x-width/2,med.T0,width,label="T0",color="#4c78a8"); ax.bar(x+width/2,med.T1,width,label="T1",color="#f58518"); ax.axhline(0,color="black",lw=.8)
            ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylabel("Median latency from teacher confirm (s)"); ax.set_title("V5 detection timing",fontweight="bold"); ax.legend(); ax.grid(axis="y",alpha=.25)
            p=plots/"v5_detection_timing.png"; _finish(fig,p); made.append(p)
    if not final.empty:
        fig,axes=plt.subplots(1,3,figsize=(14,4.8)); labels=[f"{a.replace('STx','')} {p}" for a,p in zip(final.animal,final.policy)]
        for ax,col,title,ylim in ((axes[0],"sensitivity","NVC sensitivity",(0,1.05)),(axes[1],"FP/cycle","False triggers/cycle",None),(axes[2],"median_onset_latency_s","Median onset latency (s)",None)):
            vals=pd.to_numeric(final[col],errors="coerce"); ax.bar(labels,vals,color=["#4c78a8" if "T0" in x else "#72b7b2" for x in labels]); ax.tick_params(axis="x",rotation=25); ax.set_title(title); ax.grid(axis="y",alpha=.25)
            if ylim: ax.set_ylim(*ylim)
            pad=.02 if ylim else max(float(vals.max())*.03,.015)
            for i,value in enumerate(vals):
                if pd.notna(value): ax.text(i,value+pad,f"{value:.2f}",ha="center",fontsize=8)
        fig.suptitle("V5 frozen M1 continuous replay (development-only)\nT1 preserves sensitivity but adds 0.25 s without reducing FP/cycle",fontsize=14,fontweight="bold"); p=final_plots/"v5_final_acceptance.png"; _finish(fig,p); made.append(p)
    if not runtime.empty:
        metric=runtime.copy(); metric["label"]=metric.animal.str.replace("STx","",regex=False)+" "+metric.policy
        fig,axes=plt.subplots(1,2,figsize=(12,4.8))
        for ax,col,title in ((axes[0],"sensitivity","NVC sensitivity"),(axes[1],"FP_total","False-trigger total")):
            pivot=metric.pivot(index="label",columns="runtime",values=col); pivot.plot.bar(ax=ax,color=["#e45756","#54a24b"][:pivot.shape[1]]); ax.set_title(title); ax.set_xlabel(""); ax.tick_params(axis="x",rotation=25); ax.grid(axis="y",alpha=.25)
            if col=="sensitivity": ax.set_ylim(0,1.05)
        fig.suptitle("V5 runtime correction audit",fontsize=14,fontweight="bold"); p=final_plots/"v5_old_vs_corrected_runtime.png"; _finish(fig,p); made.append(p)
    if not latency.empty:
        lat=latency.copy(); lat["event"]=lat.animal.str.replace("STx","",regex=False)+" "+lat.cycle_id.astype(str)+" "+lat.event_id.astype(str).str.split("::").str[-1]
        lat=lat.sort_values(["animal","cycle_id","event_id"]); x=np.arange(len(lat)); width=.38
        fig,ax=plt.subplots(figsize=(13,5)); ax.bar(x-width/2,pd.to_numeric(lat.latency_from_onset_T0_s,errors="coerce"),width,label="T0",color="#4c78a8"); ax.bar(x+width/2,pd.to_numeric(lat.latency_from_onset_T1_s,errors="coerce"),width,label="T1",color="#f58518")
        ax.set_xticks(x,lat.event,rotation=35,ha="right"); ax.set_ylabel("Latency from NVC onset (s)"); ax.set_title("V5 all nine prospective NVC event latencies",fontweight="bold"); ax.legend(); ax.grid(axis="y",alpha=.25)
        p=final_plots/"v5_event_latency.png"; _finish(fig,p); made.append(p)
    if not per_cycle.empty:
        frame=per_cycle[per_cycle.policy.astype(str).eq("T0")].copy()
        frame["event_sensitivity"]=pd.to_numeric(frame.nvc_detected,errors="coerce")/pd.to_numeric(frame.nvc_total,errors="coerce").replace(0,np.nan)
        frame["row"]=frame.animal.str.replace("STx","",regex=False)+" "+frame.cycle_id.astype(str)
        pivot=frame.pivot(index="row",columns="model",values="event_sensitivity").reindex(columns=["R0","M1","M2","M3","M4"])
        fig,ax=plt.subplots(figsize=(10.5,5.5)); im=ax.imshow(pivot.to_numpy(float),vmin=0,vmax=1,cmap="YlGnBu",aspect="auto")
        ax.set_xticks(range(len(pivot.columns)),[MODEL_LABELS.get(x,x) for x in pivot.columns],rotation=25,ha="right"); ax.set_yticks(range(len(pivot.index)),pivot.index)
        ax.set_title("V5 prospective NVC detection by future test cycle (T0)",fontweight="bold")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value=pivot.iloc[i,j]; ax.text(j,i,"NA" if pd.isna(value) else f"{value:.0%}",ha="center",va="center",fontsize=8)
        plt.colorbar(im,ax=ax,fraction=.03,pad=.02,label="Event sensitivity")
        p=plots/"v5_per_cycle_detection_heatmap.png"; _finish(fig,p); made.append(p)
    (plots/"PLOT_INDEX.md").write_text("# V5 图片索引\n\n- `v5_parallel_model_overview.png`：F26/F37 五模型并列结果。\n- `v5_per_cycle_detection_heatmap.png`：各未来测试 cycle 的五模型事件敏感度。\n- `v5_streaming_false_triggers.png`：连续回放 T0/T1 误触发。\n- `v5_detection_timing.png`：五模型检测时间。\n- `final_validation/v5_final_acceptance.png`：冻结 M1 最终验收。\n- `final_validation/v5_old_vs_corrected_runtime.png`：旧/修正回放逻辑对比。\n- `final_validation/v5_event_latency.png`：9 个测试 NVC 的逐事件延迟。\n",encoding="utf-8")
    return made


if __name__=="__main__": generate_plots()
