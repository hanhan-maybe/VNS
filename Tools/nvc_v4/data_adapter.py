"""Memory-bounded V4 dataset construction and stable-filling sampling."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .source_adapter import load_frozen_v31_features, _load_pair
from . import config as C
from .features import extract_v4_features


def _abs_time(cycle, idx):
    t=np.asarray(cycle.get("t_abs_s",[]),float)
    return float(t[int(idx)]) if 0 <= int(idx) < t.size else np.nan


def _stable_candidates(cycle, events, subject, cycle_id, used):
    p=np.asarray(cycle.get("bladder_pressure_mmHg",[]),float); t=np.asarray(cycle.get("t_abs_s",[]),float)
    if p.size==0 or t.size!=p.size: return []
    valid=np.asarray(cycle.get("cmg_valid_100hz",np.ones(p.size,bool)),bool)&np.isfinite(p)
    event_times=[]
    for r in events.itertuples(index=False):
        for key in ("start_s","confirm_time_s","end_s","local_recovery_time_s"):
            v=getattr(r,key,np.nan)
            if np.isfinite(v): event_times.append(float(v))
    manifest_times=[]
    for key in ("void_start_s","urine_output_onset_s","terminal_urine_episode_onset_s"):
        if key in cycle:
            v=np.asarray(cycle[key]).item() if np.asarray(cycle[key]).ndim==0 else np.nan
            if np.isfinite(v): manifest_times.append(float(v))
    candidates=[]; min_idx=int(30*C.DP_FS_HZ); step=int(2.5*C.DP_FS_HZ)
    for idx in range(min_idx, max(min_idx,p.size-int(1*C.DP_FS_HZ)), step):
        if idx in used or not valid[idx]: continue
        ts=_abs_time(cycle,idx)
        if any(abs(ts-v)<5.0 for v in event_times+manifest_times): continue
        a=max(0,idx-int(2*C.DP_FS_HZ)); x=p[a:idx+1]; ok=valid[a:idx+1]
        if ok.mean()<.98 or x[ok].size<50: continue
        # Stable means no large pressure excursion in the preceding causal window.
        if np.ptp(x[ok])>3.0 or abs(float(np.mean(np.diff(x[ok]))*C.DP_FS_HZ))>0.5: continue
        candidates.append((idx,ts))
    return candidates


def _select_stable_for_nvc(nvc_row, candidates, used, needed=3):
    target=float(nvc_row.confirm_time_s); ordered=sorted(candidates,key=lambda q:abs(q[1]-target)); picked=[]
    for idx,ts in ordered:
        if idx in used or any(abs(ts-x[1])<5.0 for x in picked): continue
        picked.append((idx,ts)); used.add(idx)
        if len(picked)>=needed: break
    return picked


def build_v4_dataset(v31_root: Path=C.V31_ROOT):
    """Return trainable NVC-vs-STABLE rows and separate PREVOID/VOID challenges."""
    base, events, manifest, paths=load_frozen_v31_features(v31_root)
    event_groups={(str(s),str(c)):g.copy() for (s,c),g in events.groupby(["subject","cycle_id"])}
    nvc=events[events.teacher_label.eq("NVC_CORE")].copy()
    rows=[]; stable_rows=[]; challenge_rows=[]; stable_audit=[]
    by_cycle={}; used_by_cycle={}
    for key,path in paths.items():
        try: item=_load_pair(path)
        except (FileNotFoundError, OSError): continue
        cycle=item["cycle"]; cg=event_groups.get(key,pd.DataFrame()); by_cycle[key]=cycle
        candidates=_stable_candidates(cycle,cg,key[0],key[1],set()); used_by_cycle[key]=set()
        positives=nvc[(nvc.subject==key[0])&(nvc.cycle_id==key[1])]
        for r in positives.itertuples(index=False):
            idx=int(r.confirm_index) if np.isfinite(r.confirm_index) else -1
            f,reason=extract_v4_features(cycle,idx,int(r.start_index),float(r.confirm_time_s))
            row={"sample_uid":str(r.event_uid),"subject":str(r.subject),"cycle_id":str(r.cycle_id),"dataset":str(r.dataset),"sample_role":"NVC_CORE","teacher_label":"NVC_CORE","target":1,"decision_index":idx,"decision_time_s":float(r.confirm_time_s),"source_event_uid":str(r.event_uid),"feature_failure_reason":reason,**f}
            rows.append(row)
            picked=_select_stable_for_nvc(r,candidates,used_by_cycle[key],needed=3)
            for j,(sidx,sts) in enumerate(picked):
                # A stable window has no event onset; use the beginning of its
                # causal two-second feature window as the baseline boundary.
                sf,sreason=extract_v4_features(cycle,sidx,max(0,sidx-int(2*C.DP_FS_HZ)),sts)
                uid=f"STABLE::{key[0]}::{key[1]}::{sidx}"
                stable_rows.append({"sample_uid":uid,"subject":key[0],"cycle_id":key[1],"dataset":str(r.dataset),"sample_role":"STABLE_FILLING","teacher_label":"STABLE_FILLING","target":0,"decision_index":sidx,"decision_time_s":sts,"source_event_uid":str(r.event_uid),"feature_failure_reason":sreason,**sf})
                stable_audit.append({"sample_uid":uid,"matched_nvc_event_uid":str(r.event_uid),"subject":key[0],"cycle_id":key[1],"distance_to_nvc_s":abs(sts-float(r.confirm_time_s)),"stable_sampling_rule":"same_cycle_causal_stable_window"})
        # PREVOID challenge: not part of training.
        for r in cg[cg.teacher_label.eq("PREVOID_PROGRESSIVE")].itertuples(index=False):
            idx=int(r.confirm_index) if np.isfinite(r.confirm_index) else -1
            f,reason=extract_v4_features(cycle,idx,int(r.start_index),float(r.confirm_time_s))
            challenge_rows.append({"challenge_type":"PREVOID_CHALLENGE","sample_uid":str(r.event_uid),"subject":key[0],"cycle_id":key[1],"teacher_label":"PREVOID_PROGRESSIVE","decision_index":idx,"decision_time_s":float(r.confirm_time_s),"feature_failure_reason":reason,**f})
        # VOID challenge uses the frozen cycle void onset only for evaluation metadata.
        void=np.asarray(cycle.get("void_start_s",np.nan)); void=float(void.item()) if void.ndim==0 else np.nan
        if np.isfinite(void):
            idx=int(np.searchsorted(np.asarray(cycle["t_abs_s"],float),void,side="right")-1); f,reason=extract_v4_features(cycle,idx,max(0,idx-int(2*C.DP_FS_HZ)),void)
            challenge_rows.append({"challenge_type":"VOID_CHALLENGE","sample_uid":f"VOID::{key[0]}::{key[1]}","subject":key[0],"cycle_id":key[1],"teacher_label":"VOID_CONFIRMED","decision_index":idx,"decision_time_s":void,"feature_failure_reason":reason,**f})
    train=pd.DataFrame(rows+stable_rows); challenges=pd.DataFrame(challenge_rows); audit=pd.DataFrame(stable_audit)
    if train.empty: raise RuntimeError("V4 dataset construction produced no rows")
    # Remove invalid structural rows only at model fit time; preserve them in the artifact.
    train["target"]=pd.to_numeric(train["target"],errors="coerce").astype("Int64")
    return train,challenges,audit,manifest,events,paths
