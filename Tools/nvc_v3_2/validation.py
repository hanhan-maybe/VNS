"""Nested animal-LOSO validation and safety-first metrics."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .models import fit_classifier, model_scores, animal_equal_weights


def _finite_frame(features, names):
    out = features[features["teacher_label"].isin(C.TARGET_LABELS)].copy()
    out["target"] = out["teacher_label"].map(C.LABEL_TO_TARGET).astype(int)
    complete = np.isfinite(out[list(names)].to_numpy(float)).all(axis=1)
    out["model_scorable"] = out.get("base_eligible", True).astype(bool) & complete
    out["model_failure_reason"] = np.where(~out.get("base_eligible", True).astype(bool), out.get("base_failure_reason", "BASE_EVENT_UNSCORABLE"), np.where(~complete, "STRUCTURAL_FEATURE_MISSING", ""))
    return out


def select_safety_threshold(y, scores, subjects):
    """Lexicographic safety rule: minimum negative FP, then macro NVC sensitivity, PPV, conservative threshold."""
    y = np.asarray(y, int); scores = np.asarray(scores, float); subjects = np.asarray(subjects, str)
    candidates = np.unique(np.r_[scores, np.nextafter(scores, np.inf), np.inf])
    rows = []
    for t in candidates:
        pred = scores >= t
        neg = y == 0; pos = y == 1
        fp = int(np.sum(pred & neg)); tp = int(np.sum(pred & pos))
        sens = []
        for a in np.unique(subjects[pos]):
            m = pos & (subjects == a); sens.append(float(np.mean(pred[m])) if m.any() else np.nan)
        macro = float(np.nanmean(sens)) if sens else np.nan
        ppv = float(tp / (tp + fp)) if tp + fp else 0.0
        rows.append({"threshold": float(t), "PREVOID_FP": fp, "macro_sensitivity": macro, "PPV": ppv})
    table = pd.DataFrame(rows)
    best = table.sort_values(["PREVOID_FP", "macro_sensitivity", "PPV", "threshold"], ascending=[True, False, False, False], na_position="last").iloc[0]
    table["selected"] = np.isclose(table["threshold"], float(best["threshold"]))
    return float(best["threshold"]), table


def _prediction_metrics(pred, model_name):
    p = pred[pred["p_nvc"].notna()].copy(); nvc = pred[pred.teacher_label == "NVC_CORE"]; pv = p[p.teacher_label == "PREVOID_PROGRESSIVE"]
    p["predicted_nvc"] = p["predicted_nvc"].astype(bool)
    tp = int((p.predicted_nvc & p.teacher_label.eq("NVC_CORE")).sum()); fp = int(pv.predicted_nvc.sum())
    per = []
    for animal, g in pred.groupby("subject", sort=True):
        sg = g[g.p_nvc.notna()]; sn = sg[sg.teacher_label.eq("NVC_CORE")]; sp = sg[sg.teacher_label.eq("PREVOID_PROGRESSIVE")]
        t = int(sn.predicted_nvc.sum()); f = int(sp.predicted_nvc.sum())
        nall = int((g.teacher_label == "NVC_CORE").sum())
        per.append({"model": model_name, "animal": str(animal), "dataset": str(g.dataset.iloc[0]),
                    "n_frozen_nvc": nall, "n_scorable_nvc": len(sn), "coverage": len(sn)/nall if nall else np.nan,
                    "n_prevoid": len(sp), "TP": t, "FP": f, "frozen_sensitivity": t/nall if nall else np.nan,
                    "scorable_sensitivity": t/len(sn) if len(sn) else np.nan,
                    "PPV": t/(t+f) if t+f else np.nan,
                    "median_score_nvc": sn.p_nvc.median() if len(sn) else np.nan,
                    "median_score_prevoid": sp.p_nvc.median() if len(sp) else np.nan,
                    "score_separation": (sn.p_nvc.median()-sp.p_nvc.median()) if len(sn) and len(sp) else np.nan,
                    "actionable_hits": int(sn.actionable_hit.sum()) if "actionable_hit" in sn else 0})
    per_df = pd.DataFrame(per)
    two = p.target.nunique() == 2
    return {"model": model_name,
            "animal_macro_frozen_sensitivity": float(per_df.loc[per_df.n_frozen_nvc > 0, "frozen_sensitivity"].mean()),
            "pooled_frozen_sensitivity": tp / len(nvc) if len(nvc) else np.nan,
            "worst_animal_sensitivity": float(per_df.loc[per_df.n_frozen_nvc > 0, "frozen_sensitivity"].min()),
            "zero_hit_animals": int((per_df.loc[per_df.n_frozen_nvc > 0, "TP"] == 0).sum()),
            "PREVOID_FP": fp, "PREVOID_FPR": fp / len(pv) if len(pv) else np.nan,
            "PPV": tp/(tp+fp) if tp+fp else np.nan,
            "AUROC": float(roc_auc_score(p.target, p.p_nvc)) if two else np.nan,
            "AUPRC": float(average_precision_score(p.target, p.p_nvc)) if two else np.nan,
            "actionable_sensitivity": float(p[p.teacher_label.eq("NVC_CORE")].actionable_hit.sum()/len(nvc)) if len(nvc) else np.nan,
            "coverage": float(len(p[p.teacher_label.eq("NVC_CORE")])/len(nvc)) if len(nvc) else np.nan,
            "frozen_NVC": len(nvc), "scorable_NVC": int((p.target == 1).sum()), "scorable_PREVOID": int((p.target == 0).sum()),
            "per_animal": per_df}


def _inner_oof(frame, features, subjects, kind, hyper):
    parts = []
    for held in subjects:
        fit_subjects = [s for s in subjects if s != held]
        train = frame[frame.subject.isin(fit_subjects) & frame.model_scorable]
        test = frame[frame.subject.eq(held)].copy(); test["p_nvc"] = np.nan
        try:
            m = fit_classifier(train, features, kind=kind, allowed_subjects=fit_subjects, **hyper)
            ok = test.model_scorable.astype(bool)
            if ok.any(): test.loc[ok, "p_nvc"] = model_scores(m, test.loc[ok])
        except (ValueError, AssertionError):
            pass
        parts.append(test.assign(inner_held_animal=held, inner_fit_animals="+".join(fit_subjects)))
    return pd.concat(parts, ignore_index=True)


def _choose_hyperparameter(frame, features, subjects, kind, grid):
    best = None
    for hyper in grid:
        oof = _inner_oof(frame, features, subjects, kind, hyper); s = oof[oof.p_nvc.notna()]
        auc = float(roc_auc_score(s.target, s.p_nvc)) if len(s) and s.target.nunique() == 2 else -np.inf
        key = (auc, -float(hyper.get("C", 1.0)))
        if best is None or key > best[0]: best = (key, hyper)
    return best[1] if best else grid[0]


def run_outer_loso(features_frame, model_name, feature_names, kind="lr", subjects=C.SUBJECTS, grid=None):
    frame = _finite_frame(features_frame, feature_names)
    grid = grid or ({"C_value": 1.0} if kind in ("lr",) else [{"C_value": 1.0}])
    if isinstance(grid, dict): grid = [grid]
    preds = []; audits = []; coefs = []; thresholds = []
    for held in subjects:
        train_subjects = [s for s in subjects if s != held]
        train = frame[frame.subject.isin(train_subjects) & frame.model_scorable]
        chosen = _choose_hyperparameter(frame[frame.subject.isin(train_subjects)], feature_names, train_subjects, kind, grid)
        inner = _inner_oof(frame[frame.subject.isin(train_subjects)], feature_names, train_subjects, kind, chosen)
        io = inner[inner.p_nvc.notna()]
        threshold, ttable = select_safety_threshold(io.target, io.p_nvc, io.subject) if len(io) else (np.inf, pd.DataFrame())
        test = frame[frame.subject.eq(held)].copy(); test["p_nvc"] = np.nan
        model = fit_classifier(train, feature_names, kind=kind, allowed_subjects=train_subjects, **chosen)
        ok = test.model_scorable.astype(bool)
        if ok.any(): test.loc[ok, "p_nvc"] = model_scores(model, test.loc[ok])
        test["predicted_nvc"] = test.p_nvc.ge(threshold) & test.p_nvc.notna()
        test["actionable_hit"] = test.predicted_nvc & test.teacher_label.eq("NVC_CORE") & test.actionable.astype(bool)
        test["model"] = model_name; test["outer_held_out_animal"] = held
        test["threshold_train_only"] = threshold; test["training_animals"] = "+".join(train_subjects)
        preds.append(test)
        audits.append({"model": model_name, "outer_held_out_animal": held,
                       "outer_training_animals": "+".join(train_subjects), "inner_selected_threshold": threshold,
                       "hyperparameters": json.dumps(chosen), "scaler_fit_animals": "+".join(train_subjects),
                       "threshold_fit_animals": "+".join(train_subjects), "leakage": False,
                       "feature_schema": "|".join(feature_names)})
        if kind == "svm":
            coef = model.named_steps["svm"].coef_[0]
        else:
            coef = model.named_steps["logistic"].coef_[0]
        for f, c in zip(feature_names, coef):
            coefs.append({"model": model_name, "outer_fold": held, "feature": f,
                          "coefficient": float(c), "sign": int(np.sign(c)), "abs_coefficient": abs(float(c))})
        if len(ttable):
            ttable["model"] = model_name; ttable["outer_held_out_animal"] = held; thresholds.append(ttable)
    prediction = pd.concat(preds, ignore_index=True)
    metrics = _prediction_metrics(prediction, model_name); per = metrics.pop("per_animal")
    return prediction, metrics, per, pd.DataFrame(audits), pd.DataFrame(coefs), (pd.concat(thresholds, ignore_index=True) if thresholds else pd.DataFrame())


def run_outer_loso_m4(features_frame, feature_names, subjects=C.SUBJECTS):
    """Independent progression model; scores are negative progression margins, not 1-p_nvc."""
    frame = _finite_frame(features_frame, feature_names); frame["progression_target"] = 1 - frame.target
    preds=[]; audits=[]; coefs=[]; thresholds=[]
    for held in subjects:
        train_subjects=[s for s in subjects if s!=held]; train=frame[frame.subject.isin(train_subjects)&frame.model_scorable].copy()
        inner_parts=[]
        for ih in train_subjects:
            fit_s=[s for s in train_subjects if s!=ih]; tr=train[train.subject.isin(fit_s)]
            m=fit_classifier(tr, feature_names, kind="lr", target_col="progression_target", allowed_subjects=fit_s, C_value=1.0)
            te=train[train.subject.eq(ih)].copy(); te["p_progression"]=model_scores(m,te); inner_parts.append(te)
        io=pd.concat(inner_parts,ignore_index=True); # threshold means low progression = NVC
        candidates=np.unique(np.r_[io.p_progression, np.nextafter(io.p_progression,np.inf), np.inf]); rows=[]
        for t in candidates:
            nvc=io.p_progression < t; pv=io.progression_target==1; fp=int(np.sum(nvc&pv)); sens=[]
            for a in io.loc[io.target==1,"subject"].unique():
                msk=(io.subject==a)&(io.target==1); sens.append(float(np.mean(nvc[msk])))
            rows.append({"progression_threshold":float(t),"PREVOID_FP":fp,"macro_sensitivity":float(np.mean(sens)) if sens else np.nan})
        tt=pd.DataFrame(rows).sort_values(["PREVOID_FP","macro_sensitivity","progression_threshold"],ascending=[True,False,True]).iloc[0]; threshold=float(tt.progression_threshold)
        model=fit_classifier(train, feature_names, kind="lr", target_col="progression_target", allowed_subjects=train_subjects, C_value=1.0)
        test=frame[frame.subject.eq(held)].copy(); ok=test.model_scorable.astype(bool); test["p_progression"]=np.nan
        if ok.any(): test.loc[ok,"p_progression"]=model_scores(model,test.loc[ok])
        test["p_nvc"]=-test.p_progression; test["predicted_nvc"]=test.p_progression.lt(threshold)&test.p_progression.notna(); test["threshold_train_only"]=threshold
        test["actionable_hit"]=test.predicted_nvc&test.teacher_label.eq("NVC_CORE")&test.actionable.astype(bool); test["model"]="M4_EVENT_PROGRESSION_GUARD"; test["outer_held_out_animal"]=held; test["training_animals"]="+".join(train_subjects)
        preds.append(test); audits.append({"model":"M4_EVENT_PROGRESSION_GUARD","outer_held_out_animal":held,"outer_training_animals":"+".join(train_subjects),"inner_selected_threshold":threshold,"threshold_fit_animals":"+".join(train_subjects),"progression_model_independent":True,"leakage":False})
        for f,c in zip(feature_names,model.named_steps["logistic"].coef_[0]): coefs.append({"model":"M4_EVENT_PROGRESSION_GUARD","outer_fold":held,"feature":f,"coefficient":float(c),"sign":int(np.sign(c)),"abs_coefficient":abs(float(c))})
        tt["model"]="M4_EVENT_PROGRESSION_GUARD";tt["outer_held_out_animal"]=held;thresholds.append(pd.DataFrame([tt]))
    prediction=pd.concat(preds,ignore_index=True); metrics=_prediction_metrics(prediction,"M4_EVENT_PROGRESSION_GUARD"); per=metrics.pop("per_animal")
    return prediction,metrics,per,pd.DataFrame(audits),pd.DataFrame(coefs),pd.concat(thresholds,ignore_index=True)
