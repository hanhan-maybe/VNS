"""Run the complete V4 NVC feature-learning development analysis."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from Tools.dsd_feature_extraction.data_io import write_json
from . import config as C
from .data_adapter import build_v4_dataset
from .validation import run_single_loso, run_fusion_loso, _fit, _score, select_balanced_threshold


def _feature_stability(frame, features):
    rows=[]
    for a,g in frame.groupby("subject",sort=True):
        n=g[g.teacher_label.eq("NVC_CORE")]; s=g[g.teacher_label.eq("STABLE_FILLING")]
        for f in features:
            nv=pd.to_numeric(n[f],errors="coerce").dropna(); sv=pd.to_numeric(s[f],errors="coerce").dropna()
            if len(nv)==0 or len(sv)==0: continue
            diff=float(nv.median()-sv.median()); pooled=float(np.sqrt((nv.var(ddof=1) if len(nv)>1 else 0)+(sv.var(ddof=1) if len(sv)>1 else 0))/2)
            rows.append({"animal":str(a),"dataset":str(g.dataset.iloc[0]),"feature":f,"NVC_median":float(nv.median()),"Stable_median":float(sv.median()),"difference":diff,"effect_size":diff/max(pooled,np.finfo(float).eps),"direction":int(np.sign(diff))})
    d=pd.DataFrame(rows)
    if d.empty: return d
    summ=d.groupby("feature",as_index=False).agg(direction_consistency=("direction",lambda x:float(np.mean(x>0) if np.mean(x>0)>=.5 else np.mean(x<0))),median_effect_size=("effect_size","median"),n_animals=("animal","nunique"),n_positive=("direction",lambda x:int(np.sum(x>0))),n_negative=("direction",lambda x:int(np.sum(x<0))))
    return d.merge(summ,on="feature",suffixes=("","_summary"))


def _within_animal(frame, features, name):
    rows=[]
    for a,g in frame[frame.teacher_label.isin(("NVC_CORE","STABLE_FILLING"))].sort_values("decision_time_s").groupby("subject"):
        g=g.copy(); cut=max(1,len(g)//2); tr=g.iloc[:cut]; te=g.iloc[cut:]
        if tr.target.nunique()<2 or te.target.nunique()<2: rows.append({"model":name,"animal":a,"status":"INSUFFICIENT_TWO_CLASS_SPLIT","AUROC":np.nan}); continue
        try:
            m=_fit(tr.assign(model_scorable=np.isfinite(tr[list(features)].to_numpy(float)).all(axis=1)),features,"lr"); ok=np.isfinite(te[list(features)].to_numpy(float)).all(axis=1); sc=np.full(len(te),np.nan); sc[ok]=_score(m,te.loc[ok]); rows.append({"model":name,"animal":a,"status":"WITHIN_ANIMAL_UPPER_BOUND","AUROC":float(roc_auc_score(te.target[ok],sc[ok])) if ok.sum() and te.target[ok].nunique()==2 else np.nan,"n_train":len(tr),"n_test":len(te)})
        except Exception as exc: rows.append({"model":name,"animal":a,"status":f"ERROR_{type(exc).__name__}","AUROC":np.nan})
    return pd.DataFrame(rows)


def _challenge_summary(challenge_preds):
    if challenge_preds.empty: return pd.DataFrame()
    rows=[]
    for (model,typ,a),g in challenge_preds.groupby(["model","challenge_type","subject"],sort=True):
        rows.append({"model":model,"challenge_type":typ,"animal":a,"n":len(g),"scorable":int(g.score.notna().sum()),"median_score":float(g.score.median()) if g.score.notna().any() else np.nan,"threshold":float(g.threshold.median()) if g.threshold.notna().any() else np.nan,"fraction_above_threshold":float(np.mean(g.score>=g.threshold)) if g.score.notna().any() else np.nan,"first_score_time_s":float(g.loc[g.score>=g.threshold,"decision_time_s"].min()) if (g.score>=g.threshold).any() else np.nan})
    return pd.DataFrame(rows)


def run(output_root: Path=C.OUTPUT_ROOT):
    output_root=Path(output_root); output_root.mkdir(parents=True,exist_ok=True)
    train,challenges,stable_audit,manifest,events,paths=build_v4_dataset()
    train.to_csv(output_root/"v4_training_samples.csv",index=False); challenges.to_csv(output_root/"v4_challenge_samples.csv",index=False); stable_audit.to_csv(output_root/"v4_stable_windows.csv",index=False)
    # Main models and preregistered ablations run independently.
    results=[]; per=[]; preds=[]; audits=[]; thresholds=[]; challenge_preds=[]
    for name,feats in (("P0_anchor",C.P0_FEATURES),("P1_morphology_dynamics",C.P1_FEATURES),("M1_P_NVC",C.P2_FEATURES),("E0_time",C.E0_FEATURES),("M2_E_NVC",C.E1_FEATURES)):
        p,m,a,t,th,ch=run_single_loso(train,name,feats,subjects=C.SUBJECTS,challenges=challenges); results.append({k:v for k,v in m.items() if k!="per_animal"}); per.append(a); preds.append(p); audits.append(t); thresholds.append(th); challenge_preds.append(ch)
    # E2 slow modulation is a diagnostic submodel and may have low coverage.
    p,m,a,t,th,ch=run_single_loso(train,"E2_slow_modulation",C.E2_FEATURES,subjects=C.SUBJECTS,challenges=challenges); results.append({k:v for k,v in m.items() if k!="per_animal"});per.append(a);preds.append(p);audits.append(t);thresholds.append(th);challenge_preds.append(ch)
    p,m,a,t,th,ch=run_fusion_loso(train,classifier="lr",subjects=C.SUBJECTS,challenges=challenges); results.append({k:v for k,v in m.items() if k!="per_animal"});per.append(a);preds.append(p);audits.append(t);thresholds.append(th);challenge_preds.append(ch)
    p,m,a,t,th,ch=run_fusion_loso(train,classifier="lda",subjects=C.SUBJECTS,challenges=challenges); results.append({k:v for k,v in m.items() if k!="per_animal"});per.append(a);preds.append(p);audits.append(t);thresholds.append(th);challenge_preds.append(ch)
    comp=pd.DataFrame(results); per_df=pd.concat(per,ignore_index=True); pred_df=pd.concat(preds,ignore_index=True); challenge_df=pd.concat(challenge_preds,ignore_index=True)
    comp.to_csv(output_root/"v4_model_comparison_full_coverage.csv",index=False); per_df.to_csv(output_root/"v4_per_animal_metrics.csv",index=False); pred_df.to_csv(output_root/"v4_event_predictions.csv",index=False); pd.concat(audits,ignore_index=True).to_csv(output_root/"v4_outer_fold_audit.csv",index=False); pd.concat(thresholds,ignore_index=True).to_csv(output_root/"v4_threshold_audit.csv",index=False); challenge_df.to_csv(output_root/"v4_challenge_predictions.csv",index=False)
    # Main-model common-scorable comparison.
    main_names=["M1_P_NVC","M2_E_NVC","M3_PE_NVC_LATE_FUSION","M4_PE_NVC_SHRINKAGE_LDA"]
    common=None
    for n in main_names:
        u=pred_df[pred_df.model.eq(n)&pred_df.score.notna()][["sample_uid","score"]].rename(columns={"score":n})
        common=u if common is None else common.merge(u,on="sample_uid",how="inner")
    common.to_csv(output_root/"v4_common_scorable_comparison.csv",index=False)
    schema=[]
    for name, feats in (("P0_anchor",C.P0_FEATURES),("P1_morphology_dynamics",C.P1_FEATURES),("M1_P_NVC",C.P2_FEATURES),("E0_time",C.E0_FEATURES),("M2_E_NVC",C.E1_FEATURES),("E2_slow_modulation",C.E2_FEATURES),("M3_PE_NVC_LATE_FUSION",C.FUSION_FEATURES),("M4_PE_NVC_SHRINKAGE_LDA",C.FUSION_FEATURES)):
        schema.extend({"model":name,"feature_order":i,"feature":f,"role":"ANCHOR_FEATURE" if f in C.P0_FEATURES else "DISCOVERY_FEATURE","preregistered":True} for i,f in enumerate(feats))
    pd.DataFrame(schema).to_csv(output_root/"v4_feature_schema.csv",index=False)
    # Feature direction and within-animal diagnostics.
    stability=_feature_stability(train,C.P2_FEATURES+C.E1_FEATURES+C.COUPLING_FEATURES); stability.to_csv(output_root/"v4_feature_stability.csv",index=False)
    within=pd.concat([_within_animal(train,C.P2_FEATURES,"M1_P_NVC"),_within_animal(train,C.E1_FEATURES,"M2_E_NVC")],ignore_index=True); within.to_csv(output_root/"v4_within_animal_upper_bound.csv",index=False)
    chsum=_challenge_summary(challenge_df); chsum.to_csv(output_root/"v4_challenge_summary.csv",index=False)
    # Diagnostic score trajectories: frozen confirm-point scores and delay labels.
    traj=pred_df[pred_df.model.isin(main_names)][[c for c in ["model","sample_uid","subject","teacher_label","decision_time_s","score","threshold","predicted_nvc"] if c in pred_df.columns]].copy(); traj["status"]="CONFIRM_POINT_FROZEN"
    if len(challenge_df):
        ct=challenge_df[challenge_df.model.isin(main_names)][[c for c in ["model","sample_uid","subject","teacher_label","decision_time_s","score","threshold"] if c in challenge_df.columns]].copy(); ct["status"]="PREVOID_OR_VOID_CHALLENGE_FROZEN"; ct["predicted_nvc"]=ct.score>=ct.threshold; traj=pd.concat([traj,ct],ignore_index=True)
    traj.to_csv(output_root/"v4_score_trajectories.csv",index=False)
    dataset=manifest.assign(dataset=manifest.dataset.astype(str)).groupby("dataset",as_index=False).agg(animals=("subject","nunique"),cycles=("cycle_id","nunique")); counts=train.assign(dataset=train.dataset.astype(str)).groupby(["dataset","teacher_label"]).size().unstack(fill_value=0).reset_index(); dataset=dataset.merge(counts,on="dataset",how="left"); dataset.to_csv(output_root/"v4_dataset_summary.csv",index=False)
    # Explicit failure audit.
    fail=train[["sample_uid","subject","teacher_label","feature_failure_reason"]].copy(); fail.to_csv(output_root/"v4_failure_reasons.csv",index=False)
    v31_repro=pd.read_csv(C.V31_ROOT/"baseline_v3_reproduction.csv"); v31_pass=bool(v31_repro["match"].astype(bool).all())
    statuses={}
    def auc(name):
        r=comp[comp.model.eq(name)]; return float(r.AUROC.iloc[0]) if len(r) and np.isfinite(r.AUROC.iloc[0]) else np.nan
    statuses["PRESSURE_NVC_SUPPORTED"] = bool(np.isfinite(auc("M1_P_NVC")) and auc("M1_P_NVC")>0.7)
    statuses["EUS_NVC_SUPPORTED"] = bool(np.isfinite(auc("M2_E_NVC")) and auc("M2_E_NVC")>0.7)
    statuses["MULTIMODAL_NVC_SUPPORTED"] = bool(np.isfinite(auc("M3_PE_NVC_LATE_FUSION")) and auc("M3_PE_NVC_LATE_FUSION")>0.7)
    statuses["SHRINKAGE_LDA_IMPROVEMENT_SUPPORTED"] = bool(np.isfinite(auc("M4_PE_NVC_SHRINKAGE_LDA")) and np.isfinite(auc("M3_PE_NVC_LATE_FUSION")) and auc("M4_PE_NVC_SHRINKAGE_LDA")>auc("M3_PE_NVC_LATE_FUSION") and auc("M4_PE_NVC_SHRINKAGE_LDA")>0.5)
    summary={"v31_reproduction":"PASS" if v31_pass else "FAIL","subjects":list(C.SUBJECTS),"subjects_338":list(C.SUBJECTS_338),"subjects_164":list(C.SUBJECTS_164),"primary_task":"NVC_CORE vs STABLE_FILLING","positive_nvc":int((train.teacher_label=="NVC_CORE").sum()),"stable_filling":int((train.teacher_label=="STABLE_FILLING").sum()),"prevoid_challenge":int((challenges.challenge_type=="PREVOID_CHALLENGE").sum()),"void_challenge":int((challenges.challenge_type=="VOID_CHALLENGE").sum()),"models_parallel":True,"main_models":["M1_P_NVC","M2_E_NVC","M3_PE_NVC_LATE_FUSION","M4_PE_NVC_SHRINKAGE_LDA"],"model_results":comp.to_dict(orient="records"),"feature_stability_threshold_descriptive":0.75,"common_scorable_n":int(len(common)),"statuses":statuses,"development_status":"COMPLETED_DEVELOPMENT_ONLY","deployment_ready":False,"stimulation_enabled":False}
    write_json(output_root/"v4_summary.json",summary)
    report=["# V4 NVC feature learning","","## 1. Task and cohort\n- Primary task: NVC_CORE vs STABLE_FILLING\n- 338 + 164 remain development-only; 164 is not external validation\n- NVC samples: %d; stable samples: %d"%(summary["positive_nvc"],summary["stable_filling"]),"","## 2. V3.1 reproduction\n- %s"%summary["v31_reproduction"],"","## 3. Parallel models",comp.to_markdown(index=False),"","## 4. Feature stability\nPer-animal median differences and direction consistency are in `v4_feature_stability.csv`; 75%% is descriptive only.","","## 5. Common-scorable comparison\nSee `v4_common_scorable_comparison.csv`.","","## 6. Within-animal upper bound\nSee `v4_within_animal_upper_bound.csv`; this is diagnostic, not external validation.","","## 7. PREVOID / VOID challenge\nChallenges were scored only after models and thresholds were frozen; they were not used for fitting or threshold selection.","","## 8. Interpretation\nThe principal question is whether NVC has reproducible pressure/EUS features against stable filling; PREVOID is not a training negative in V4.","","## 9. Status\n- development_status: COMPLETED_DEVELOPMENT_ONLY\n- deployment_ready: false\n- stimulation_enabled: false"]
    (output_root/"V4_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    from .visualization import generate_plots
    generate_plots(output_root)
    return summary

if __name__=="__main__": run()
