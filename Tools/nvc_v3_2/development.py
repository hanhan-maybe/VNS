"""Executable V3.2 parallel development run."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .data_adapter import load_v32_inputs, build_v32_features, load_frozen_v31_features, augment_frozen_features
from .spectral_features import stft_frequency_names
from .validation import run_outer_loso, run_outer_loso_m4
from Tools.dsd_feature_extraction.data_io import write_json


MODEL_DEFS = (
    ("B0-primary", C.PE_FEATURES, "lr", [{"C_value": 1.0}]),
    ("M1_P_SPEC_SHORT", C.M1_FEATURES, "lr", [{"C_value": 1.0}]),
    # M2's frequency grid is determined once from acquisition metadata, before any fold runs.
    ("M2_PE_EUS_STFT_SPARSE", None, "elasticnet", [{"C_value": c, "l1_ratio": r} for c in (0.01, 0.1, 1.0, 10.0) for r in (0.25, 0.5, 0.75)]),
    ("M3_PE_TF_COMPACT_LR", C.M3_FEATURES, "lr", [{"C_value": 1.0}]),
    ("M5_PE_TF_COMPACT_SVM", C.M5_FEATURES, "svm", [{"C_value": c} for c in (0.01, 0.1, 1.0, 10.0)]),
)


def _status(row, reference):
    if row.get("coverage", 0.0) < 0.95: return "COVERAGE_FAILURE"
    if (row.get("animal_macro_frozen_sensitivity", np.nan) > reference.get("animal_macro_frozen_sensitivity", np.nan)
            and row.get("PREVOID_FPR", np.inf) <= reference.get("PREVOID_FPR", np.inf)
            and row.get("worst_animal_sensitivity", -np.inf) >= reference.get("worst_animal_sensitivity", -np.inf)
            and row.get("zero_hit_animals", np.inf) <= reference.get("zero_hit_animals", np.inf)):
        return "SAFE_PARETO_IMPROVEMENT"
    if row.get("animal_macro_frozen_sensitivity", 0.0) > reference.get("animal_macro_frozen_sensitivity", 0.0) and row.get("PREVOID_FPR", 0.0) > reference.get("PREVOID_FPR", 0.0):
        return "RECALL_SAFETY_TRADEOFF"
    return "NO_MEANINGFUL_GAIN"


def _challenge(predictions):
    rows=[]
    for model, g in predictions.groupby("model"):
        for animal in ("STxF26", "STxF33", "STxF34", "STxF37"):
            x=g[g.subject.eq(animal)]; n=x[x.teacher_label.eq("NVC_CORE")]; pv=x[x.teacher_label.eq("PREVOID_PROGRESSIVE")]
            rows.append({"model":model,"animal":animal,"frozen_NVC":len(n),"scorable_NVC":int(n.p_nvc.notna().sum()),"TP":int(n.predicted_nvc.sum()),"sensitivity":float(n.predicted_nvc.sum()/len(n)) if len(n) else np.nan,"PREVOID_FP":int(pv.predicted_nvc.sum()),"median_NVC_score":n.p_nvc.median(),"median_PREVOID_score":pv.p_nvc.median(),"score_separation":n.p_nvc.median()-pv.p_nvc.median() if len(n) and len(pv) else np.nan})
    return pd.DataFrame(rows)


def _causal_replay(predictions):
    rows=[]
    for model, g in predictions.groupby("model"):
        for animal, x in g.sort_values(["subject","decision_time_s"]).groupby("subject"):
            last=-np.inf
            for _, r in x.iterrows():
                trigger=bool(r.get("predicted_nvc",False)); t=float(r.get("decision_time_s",np.nan)); allowed=trigger and (t-last>=15.0)
                if allowed: last=t
                rows.append({"model":model,"subject":animal,"event_uid":r.event_uid,"decision_time_s":t,"predicted_nvc":trigger,"lockout_s":15.0,"trigger_allowed":allowed,"urine_used":False})
    return pd.DataFrame(rows)


def run(output_root: Path = C.DEFAULT_OUTPUT_ROOT, overwrite: bool = False):
    output_root=Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    # Reuse the immutable V3.1 delayed feature artifact and source descriptors;
    # only native EUS/pressure spectra are loaded cycle-by-cycle.
    base_features, events, manifest, cycle_paths = load_frozen_v31_features()
    features, meta=augment_frozen_features(base_features, events, cycle_paths, delays=C.DIAGNOSTIC_DELAYS_S)
    # M2 feature names are fixed globally from common acquisition capability.
    m2_features=tuple(C.PE_FEATURES)+tuple(meta["m2_frequency_features"])
    model_defs=[(n, (m2_features if n.startswith("M2") else f), k, grid) for n,f,k,grid in MODEL_DEFS]

    predictions=[]; metric_rows=[]; per_rows=[]; audit_rows=[]; coef_rows=[]; threshold_rows=[]
    for name, feats, kind, grid in model_defs:
        frame=features[features.decision_delay_s.eq(C.PRIMARY_DELAY_S)].copy()
        pred, metrics, per, audit, coefs, thresholds=run_outer_loso(frame, name, feats, kind=kind, subjects=C.SUBJECTS, grid=grid)
        predictions.append(pred); metric_rows.append({k:v for k,v in metrics.items() if k!="per_animal"}); per_rows.append(per)
        audit_rows.append(audit); coef_rows.append(coefs); threshold_rows.append(thresholds)
    m4_pred,m4_metrics,m4_per,m4_audit,m4_coefs,m4_thresholds=run_outer_loso_m4(features[features.decision_delay_s.eq(C.PRIMARY_DELAY_S)].copy(), C.M3_FEATURES, subjects=C.SUBJECTS)
    predictions.append(m4_pred); metric_rows.append({k:v for k,v in m4_metrics.items() if k!="per_animal"}); per_rows.append(m4_per); audit_rows.append(m4_audit); coef_rows.append(m4_coefs); threshold_rows.append(m4_thresholds)
    all_pred=pd.concat(predictions,ignore_index=True); comparison=pd.DataFrame(metric_rows); per_animal=pd.concat(per_rows,ignore_index=True)
    ref=comparison[comparison.model.eq("B0-primary")].iloc[0].to_dict(); comparison["status"]=comparison.apply(lambda r: "REFERENCE" if r.model=="B0-primary" else _status(r,ref),axis=1)
    # Legacy reference is read-only evidence from V3.1, never overwritten.
    legacy_path=C.DEFAULT_V31_ROOT/"model_comparison_v31.csv"; legacy=pd.read_csv(legacy_path) if legacy_path.exists() else pd.DataFrame()
    # Required artifacts.
    events.to_csv(output_root/"v32_source_events.csv",index=False)
    comparison.to_csv(output_root/"v32_primary_model_comparison.csv",index=False)
    per_animal.to_csv(output_root/"v32_per_animal_metrics.csv",index=False)
    all_pred.to_csv(output_root/"v32_event_predictions.csv",index=False)
    coverage=per_animal.groupby("model",as_index=False).agg(coverage=("coverage","mean"),scorable_NVC=("n_scorable_nvc","sum"),frozen_NVC=("n_frozen_nvc","sum")); coverage.to_csv(output_root/"v32_model_coverage.csv",index=False)
    failures=features.melt(id_vars=["model"],value_vars=[]) if False else features[["subject","event_uid","teacher_label","m1_failure_reason","m2_failure_reason","m3_failure_reason"]].copy(); failures.to_csv(output_root/"v32_failure_reasons.csv",index=False)
    pd.concat(audit_rows,ignore_index=True).to_csv(output_root/"v32_outer_fold_audit.csv",index=False)
    pd.concat(threshold_rows,ignore_index=True).to_csv(output_root/"v32_threshold_audit.csv",index=False)
    schema=[]
    for n,f,_,_ in model_defs+[('M4_EVENT_PROGRESSION_GUARD',C.M3_FEATURES,'lr',[])]:
        schema += [{"model":n,"feature_order":i,"feature":x,"preregistered":True,"allowed":True} for i,x in enumerate(f)]
    pd.DataFrame(schema).to_csv(output_root/"v32_feature_schema.csv",index=False)
    pd.concat(coef_rows,ignore_index=True).to_csv(output_root/"v32_lr_coefficients.csv",index=False)
    c=pd.concat(coef_rows,ignore_index=True); m2f=c[(c.model=="M2_PE_EUS_STFT_SPARSE") & c.feature.str.startswith("eus_stft_bin_")].copy(); m2f.to_csv(output_root/"m2_frequency_selection_by_fold.csv",index=False)
    if not m2f.empty:
        summ=m2f.groupby("feature",as_index=False).agg(selection_frequency_across_outer_folds=("abs_coefficient",lambda x: int(np.sum(x>0))),median_coefficient=("coefficient","median"),coefficient_sign_consistency=("sign",lambda x: abs(float(np.mean(np.sign(x)))))); summ.to_csv(output_root/"m2_frequency_selection_summary.csv",index=False)
    else: pd.DataFrame(columns=["feature","selection_frequency_across_outer_folds","median_coefficient","coefficient_sign_consistency"]).to_csv(output_root/"m2_frequency_selection_summary.csv",index=False)
    m3=comparison[comparison.model=="M3_PE_TF_COMPACT_LR"].iloc[0]; m5=comparison[comparison.model=="M5_PE_TF_COMPACT_SVM"].iloc[0]
    pd.DataFrame([m3,m5]).to_csv(output_root/"m3_vs_m5_classifier_comparison.csv",index=False)
    m4_pred[[c for c in ["subject","event_uid","teacher_label","p_progression","p_nvc","predicted_nvc","threshold_train_only","outer_held_out_animal"] if c in m4_pred]].to_csv(output_root/"m4_progression_predictions.csv",index=False)
    m4_coefs.to_csv(output_root/"m4_progression_coefficients.csv",index=False); m4_thresholds.to_csv(output_root/"m4_guard_threshold_audit.csv",index=False)
    _challenge(all_pred).to_csv(output_root/"v32_challenge_animals.csv",index=False)
    # Diagnostic-only delays: descriptive score separation, never used for selection.
    drows=[]
    for d in C.DIAGNOSTIC_DELAYS_S:
        x=features[features.decision_delay_s.eq(d)]
        drows.append({"delay_s":d,"status":"DIAGNOSTIC_ONLY","n_rows":len(x),"n_nvc":int((x.teacher_label=="NVC_CORE").sum()),"n_prevoid":int((x.teacher_label=="PREVOID_PROGRESSIVE").sum()),"m1_scorable":int(x.m1_scorable.sum()),"m3_scorable":int(x.m3_scorable.sum())})
    pd.DataFrame(drows).to_csv(output_root/"v32_delay_diagnostic.csv",index=False)
    _causal_replay(all_pred).to_csv(output_root/"v32_causal_replay.csv",index=False)
    legacy_pass=bool(not legacy.empty and legacy.get("match",pd.Series(dtype=bool)).all())
    dataset_summary=manifest.groupby("dataset",as_index=False).agg(animals=("subject","nunique"),cycles=("cycle_id","nunique"))
    counts=events[events.teacher_label.isin(C.TARGET_LABELS)].groupby(["dataset","teacher_label"]).size().unstack(fill_value=0).reset_index()
    dataset_summary=dataset_summary.merge(counts,on="dataset",how="left")
    dataset_summary.to_csv(output_root/"v32_dataset_summary.csv",index=False)
    summary={"v31_reproduction":"PASS" if legacy_pass else "FAIL_OR_UNAVAILABLE","subjects":list(C.SUBJECTS),"subjects_338":list(C.SUBJECTS_338),"subjects_164":list(C.SUBJECTS_164),"primary_delay_s":C.PRIMARY_DELAY_S,"common_eus_high_hz":meta["common_eus_high_hz"],"included_eus_bands":meta["included_eus_bands"],"excluded_eus_bands":meta["excluded_eus_bands"],"models_parallel":True,"model_results":comparison.to_dict(orient="records"),"m3_m5_same_features":tuple(C.M3_FEATURES)==tuple(C.M5_FEATURES),"spectral_coverage":meta,"development_status":"COMPLETED_WITH_DEVELOPMENT_ONLY_EVALUATION","deployment_ready":False,"stimulation_enabled":False}
    write_json(output_root/"v3_2_summary.json",summary)
    report=["# 338 + 164 NVC V3.2 parallel mechanism development","",f"## 1. V3.1 reproduction\n- {summary['v31_reproduction']}","",f"## 2. Cohort\n- subjects: {', '.join(C.SUBJECTS)}\n- datasets: 338 + 164 development cohort\n- primary delay: {C.PRIMARY_DELAY_S} s","",f"## 3. Common EUS spectral capability\n- native rates: {meta['native_eus_rates_hz']} Hz\n- common reliable bandwidth: {meta['common_eus_high_hz']} Hz\n- included bands: {meta['included_eus_bands']}\n- excluded bands: {meta['excluded_eus_bands']}","","## 4. B0-primary\nPE_TIME @ confirm +2.0 s","","## 5. Parallel model results",comparison.to_markdown(index=False),"","## 6. Safety Pareto comparison\nSafety-first thresholding was applied in inner animal-LOSO folds.","","## 7. Per-animal results\nSee `v32_per_animal_metrics.csv`.","","## 8. Challenge animals\nSee `v32_challenge_animals.csv`; phenotype generalization remains unresolved if F37 remains low.","","## 9. Spectral mechanism findings\nM1/M2 outputs are mechanism diagnostics; frequency selection is fold-specific and train-only.","","## 10. M3 vs M5\nSee `m3_vs_m5_classifier_comparison.csv`.","","## 11. Independent progression guard\nM4 fits a separate progression target, scaler, model, and threshold.","","## 12. Coverage and failure reasons\nStructural missingness is explicit; no future fill or NaN=0.","","## 13. Delay diagnostic\nAll delay rows are DIAGNOSTIC_ONLY; no delay selection.","","## 14. Mechanistic conclusion\nDevelopment-only evidence does not establish cross-animal deployment safety.","","## 15. V3.3 decision\nHOLD_FEATURE_NONSEPARABILITY / HOLD_PHENOTYPE_HETEROGENEITY pending independent animals.",""]
    (output_root/"V3_2_REPORT.md").write_text("\n".join(report),encoding="utf-8")
    return summary


if __name__ == "__main__":
    run()
