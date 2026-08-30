"""Animal-level LOSO validation for V4 NVC-vs-stable feature learning."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def _weights(frame):
    n_animals=max(frame.subject.nunique(),1); counts=frame.groupby(["subject","target"]).size().to_dict(); classes=frame.groupby("subject")["target"].nunique().to_dict()
    return np.asarray([1.0/(n_animals*max(classes.get(str(s),1),1)*counts.get((s,int(y)),1)) for s,y in zip(frame.subject.astype(str),frame.target)],float)


def _prepare(frame, features, role_filter=("NVC_CORE","STABLE_FILLING")):
    out=frame[frame.teacher_label.isin(role_filter)].copy(); out["target"]=out.teacher_label.map({"NVC_CORE":1,"STABLE_FILLING":0}).astype(int)
    complete=np.isfinite(out[list(features)].to_numpy(float)).all(axis=1)
    out["model_scorable"]=complete; out["model_failure_reason"]=np.where(~complete,"STRUCTURAL_FEATURE_MISSING","")
    return out


def _fit(frame, features, classifier="lr"):
    mask=frame["model_scorable"].astype(bool) if "model_scorable" in frame.columns else np.ones(len(frame),dtype=bool)
    d=frame[mask].copy(); C.assert_safe_feature_schema(features)
    if d.target.nunique()<2: raise ValueError("training fold lacks NVC and stable classes")
    if classifier=="lda":
        model=Pipeline([("scaler",StandardScaler()),("lda",LinearDiscriminantAnalysis(solver="lsqr",shrinkage="auto"))])
        model.fit(d[list(features)],d.target,sample_weight=_weights(d)) if False else model.fit(d[list(features)],d.target)
    else:
        model=Pipeline([("scaler",StandardScaler()),("logistic",LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=2000,class_weight=None,random_state=C.RANDOM_STATE))])
        model.fit(d[list(features)],d.target,logistic__sample_weight=_weights(d))
    model.fit_features_=tuple(features); model.fit_animals_=tuple(sorted(d.subject.astype(str).unique())); model.classifier_=classifier
    return model


def _score(model, frame):
    return model.predict_proba(frame[list(model.fit_features_)])[:,1].astype(float)


def select_balanced_threshold(y, scores):
    y=np.asarray(y,int); s=np.asarray(scores,float); candidates=np.unique(np.r_[s,np.nextafter(s,np.inf),0.5])
    rows=[]
    for t in candidates:
        pred=s>=t; bal=float(balanced_accuracy_score(y,pred)); pos=y==1; neg=y==0
        sens=float(np.mean(pred[pos])) if pos.any() else np.nan; spec=float(np.mean(~pred[neg])) if neg.any() else np.nan
        rows.append({"threshold":float(t),"balanced_accuracy":bal,"sensitivity":sens,"specificity":spec,"youden_j":sens+spec-1 if np.isfinite(sens) and np.isfinite(spec) else np.nan})
    tab=pd.DataFrame(rows); best=tab.sort_values(["balanced_accuracy","youden_j","threshold"],ascending=[False,False,False],na_position="last").iloc[0]; tab["selected"]=np.isclose(tab.threshold,float(best.threshold)); return float(best.threshold),tab


def _metrics(pred, model):
    p=pred[pred.score.notna()].copy(); n=p[p.teacher_label.eq("NVC_CORE")]; st=p[p.teacher_label.eq("STABLE_FILLING")]; p["predicted_nvc"]=p.score>=p.threshold
    per=[]
    for a,g in pred.groupby("subject",sort=True):
        sg=g[g.score.notna()]; sn=sg[sg.teacher_label.eq("NVC_CORE")]; ss=sg[sg.teacher_label.eq("STABLE_FILLING")]; tp=int((sn.score>=sn.threshold).sum()) if len(sn) else 0; tn=int((ss.score<ss.threshold).sum()) if len(ss) else 0
        per.append({"model":model,"animal":str(a),"dataset":str(g.dataset.iloc[0]),"n_frozen_nvc":len(g[g.teacher_label.eq("NVC_CORE")]),"n_scorable_nvc":len(sn),"n_stable":len(g[g.teacher_label.eq("STABLE_FILLING")]),"TP":tp,"TN":tn,"frozen_sensitivity":tp/len(g[g.teacher_label.eq("NVC_CORE")]) if len(g[g.teacher_label.eq("NVC_CORE")]) else np.nan,"scorable_sensitivity":tp/len(sn) if len(sn) else np.nan,"specificity":tn/len(ss) if len(ss) else np.nan,"coverage":len(sn)/len(g[g.teacher_label.eq("NVC_CORE")]) if len(g[g.teacher_label.eq("NVC_CORE")]) else np.nan})
    per_df=pd.DataFrame(per); two=p.target.nunique()==2
    return {"model":model,"AUROC":float(roc_auc_score(p.target,p.score)) if two else np.nan,"AUPRC":float(average_precision_score(p.target,p.score)) if two else np.nan,"macro_sensitivity":float(per_df.loc[per_df.n_frozen_nvc>0,"frozen_sensitivity"].mean()),"pooled_sensitivity":float((n.score>=n.threshold).sum()/len(n)) if len(n) else np.nan,"macro_specificity":float(per_df.loc[per_df.n_stable>0,"specificity"].mean()),"worst_animal_sensitivity":float(per_df.loc[per_df.n_frozen_nvc>0,"frozen_sensitivity"].min()),"zero_hit_animals":int((per_df.loc[per_df.n_frozen_nvc>0,"TP"]==0).sum()),"coverage":float(len(n)/sum(pred.teacher_label.eq("NVC_CORE"))) if sum(pred.teacher_label.eq("NVC_CORE")) else np.nan,"threshold_mean":float(p.threshold.mean()),"nvc_scorable":len(n),"stable_scorable":len(st),"per_animal":per_df}


def _inner_oof(frame, features, subjects, classifier="lr"):
    parts=[]
    for held in subjects:
        fit_s=[s for s in subjects if s!=held]; tr=frame[frame.subject.isin(fit_s)].copy(); te=frame[frame.subject.eq(held)].copy(); te["score"]=np.nan
        try:
            m=_fit(tr,features,classifier); ok=te.model_scorable.astype(bool)
            if ok.any(): te.loc[ok,"score"]=_score(m,te.loc[ok])
        except ValueError: pass
        parts.append(te.assign(inner_held_animal=held,inner_fit_animals="+".join(fit_s),threshold=np.nan))
    return pd.concat(parts,ignore_index=True)


def run_single_loso(frame, model_name, features, classifier="lr", subjects=C.SUBJECTS, challenges=None):
    frame=_prepare(frame,features); preds=[]; audits=[]; thresholds=[]; challenge_parts=[]
    for held in subjects:
        fit_s=[s for s in subjects if s!=held]; train=frame[frame.subject.isin(fit_s)].copy(); inner=_inner_oof(frame[frame.subject.isin(fit_s)],features,fit_s,classifier); io=inner[inner.score.notna()]
        threshold,tt=select_balanced_threshold(io.target,io.score) if len(io) else (np.nan,pd.DataFrame()); model=_fit(train,features,classifier); test=frame[frame.subject.eq(held)].copy(); test["score"]=np.nan; ok=test.model_scorable.astype(bool)
        if ok.any(): test.loc[ok,"score"]=_score(model,test.loc[ok])
        test["threshold"]=threshold; test["model"]=model_name; test["outer_held_out_animal"]=held; test["predicted_nvc"]=test.score>=threshold; preds.append(test)
        audits.append({"model":model_name,"outer_held_out_animal":held,"outer_training_animals":"+".join(fit_s),"threshold_fit_animals":"+".join(fit_s),"scaler_fit_animals":"+".join(model.fit_animals_),"nested_inner_oof":True,"leakage":False,"classifier":classifier,"feature_schema":"|".join(features)})
        if len(tt): tt=tt.assign(model=model_name,outer_held_out_animal=held); thresholds.append(tt)
        if challenges is not None and len(challenges):
            ch=challenges[challenges.subject.eq(held)].copy(); ch["score"]=np.nan; valid=np.isfinite(ch[list(features)].to_numpy(float)).all(axis=1)
            if valid.any(): ch.loc[valid,"score"]=_score(model,ch.loc[valid])
            ch["threshold"]=threshold; ch["model"]=model_name; ch["outer_held_out_animal"]=held; challenge_parts.append(ch)
    pred=pd.concat(preds,ignore_index=True); met=_metrics(pred,model_name); per=met.pop("per_animal"); challenge=pd.concat(challenge_parts,ignore_index=True) if challenge_parts else pd.DataFrame(); return pred,met,per,pd.DataFrame(audits),pd.concat(thresholds,ignore_index=True) if thresholds else pd.DataFrame(),challenge


def _inner_fusion_oof(frame, p_features, e_features, subjects):
    parts=[]
    for held in subjects:
        fit_s=[s for s in subjects if s!=held]; tr=frame[frame.subject.isin(fit_s)].copy(); te=frame[frame.subject.eq(held)].copy()
        te["S_P"]=np.nan; te["S_E"]=np.nan
        try:
            pm=_fit(tr[tr.pm_scorable],p_features,"lr"); em=_fit(tr[tr.em_scorable],e_features,"lr"); po=te.pm_scorable.astype(bool); eo=te.em_scorable.astype(bool)
            if po.any(): te.loc[po,"S_P"]=_score(pm,te.loc[po]);
            if eo.any(): te.loc[eo,"S_E"]=_score(em,te.loc[eo])
        except ValueError: pass
        parts.append(te.assign(inner_held_animal=held,inner_fit_animals="+".join(fit_s)))
    return pd.concat(parts,ignore_index=True)


def run_fusion_loso(frame, p_features=C.P2_FEATURES, e_features=C.E1_FEATURES, classifier="lr", subjects=C.SUBJECTS, challenges=None):
    base=frame[frame.teacher_label.isin(("NVC_CORE","STABLE_FILLING"))].copy(); base["target"]=base.teacher_label.map({"NVC_CORE":1,"STABLE_FILLING":0}).astype(int)
    base["pm_scorable"]=np.isfinite(base[list(p_features)].to_numpy(float)).all(axis=1); base["em_scorable"]=np.isfinite(base[list(e_features)].to_numpy(float)).all(axis=1)
    fparts=[]; audits=[]; thresholds=[]; cparts=[]
    for held in subjects:
        fusion_name = "M3_PE_NVC_LATE_FUSION" if classifier == "lr" else "M4_PE_NVC_SHRINKAGE_LDA"
        fit_s=[s for s in subjects if s!=held]; tr=base[base.subject.isin(fit_s)].copy(); io=_inner_fusion_oof(base[base.subject.isin(fit_s)],p_features,e_features,fit_s)
        fcols=list(C.FUSION_FEATURES); complete=np.isfinite(io[["S_P","S_E"]+list(C.COUPLING_FEATURES)].to_numpy(float)).all(axis=1); io["model_scorable"]=complete; fi=io[io.model_scorable].copy()
        if len(fi):
            fm=_fit(fi,fcols,classifier); threshold,tt=select_balanced_threshold(fi.target,_score(fm,fi))
        else: fm=None; threshold=np.nan; tt=pd.DataFrame()
        pm=_fit(tr[tr.pm_scorable],p_features,"lr"); em=_fit(tr[tr.em_scorable],e_features,"lr"); te=base[base.subject.eq(held)].copy(); te["S_P"]=np.nan;te["S_E"]=np.nan
        po=te.pm_scorable.astype(bool); eo=te.em_scorable.astype(bool)
        if po.any(): te.loc[po,"S_P"]=_score(pm,te.loc[po]);
        if eo.any(): te.loc[eo,"S_E"]=_score(em,te.loc[eo])
        te["score"]=np.nan; ok=np.isfinite(te[["S_P","S_E"]+list(C.COUPLING_FEATURES)].to_numpy(float)).all(axis=1)
        if fm is not None and ok.any(): te.loc[ok,"score"]=_score(fm,te.loc[ok])
        te["threshold"]=threshold;te["model"]=fusion_name;te["outer_held_out_animal"]=held;te["predicted_nvc"]=te.score>=threshold; fparts.append(te)
        audits.append({"model":fusion_name,"outer_held_out_animal":held,"outer_training_animals":"+".join(fit_s),"threshold_fit_animals":"+".join(fit_s),"fusion_fit_animals":"+".join(fit_s),"base_scores_inner_oof":True,"progression_model_independent":False,"leakage":False,"classifier":classifier,"feature_schema":"|".join(fcols)})
        if len(tt): thresholds.append(tt.assign(model=fusion_name,outer_held_out_animal=held))
        if challenges is not None and len(challenges):
            ch=challenges[challenges.subject.eq(held)].copy();ch["S_P"]=np.nan;ch["S_E"]=np.nan;cp=ch[list(p_features)].notna().all(axis=1);ce=ch[list(e_features)].notna().all(axis=1)
            if cp.any(): ch.loc[cp,"S_P"]=_score(pm,ch.loc[cp]);
            if ce.any(): ch.loc[ce,"S_E"]=_score(em,ch.loc[ce]);
            ch["score"]=np.nan; cok=np.isfinite(ch[["S_P","S_E"]+list(C.COUPLING_FEATURES)].to_numpy(float)).all(axis=1)
            if fm is not None and cok.any(): ch.loc[cok,"score"]=_score(fm,ch.loc[cok])
            ch["threshold"]=threshold;ch["model"]=fusion_name;ch["outer_held_out_animal"]=held;cparts.append(ch)
    pred=pd.concat(fparts,ignore_index=True);met=_metrics(pred,fusion_name);per=met.pop("per_animal");challenge=pd.concat(cparts,ignore_index=True) if cparts else pd.DataFrame();return pred,met,per,pd.DataFrame(audits),pd.concat(thresholds,ignore_index=True) if thresholds else pd.DataFrame(),challenge
