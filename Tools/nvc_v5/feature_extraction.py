"""Causal V4 pressure/EUS feature extraction and stable-window sampling."""
from __future__ import annotations
import numpy as np
from scipy.signal import detrend
from .spectral_features import causal_pressure_spectral_features, causal_eus_compact_bands
from . import config as C


def _finite_mad(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size < 2: return np.nan, np.nan
    med = float(np.median(x)); mad = float(1.4826 * np.median(np.abs(x-med)))
    return med, max(mad, np.finfo(float).eps)


def _band_power(x, fs, lo, hi):
    x = np.asarray(x, float)
    if x.size < 8 or not np.isfinite(x).all(): return np.nan
    x = detrend(x - np.mean(x)); x *= np.hanning(x.size)
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size, 1.0 / fs); m = (f >= lo) & (f <= hi)
    return float(np.trapz(p[m], f[m])) if m.any() else np.nan


def _spectral_entropy(x, fs, lo=0.2, hi=20.0):
    x=np.asarray(x,float)
    if x.size<8 or not np.isfinite(x).all(): return np.nan
    x=detrend(x-np.mean(x)); p=np.abs(np.fft.rfft(x*np.hanning(x.size)))**2; f=np.fft.rfftfreq(x.size,1/fs); m=(f>=lo)&(f<=hi)
    q=p[m]; total=float(q.sum())
    if total<=0 or not np.isfinite(total): return np.nan
    q=q/total; return float(-np.sum(q*np.log(q+C.EPSILON))/np.log(max(q.size,2)))


def _slow_features(env, valid, index, fs=C.DP_FS_HZ):
    n=int(round(C.EUS_SLOW_HISTORY_S*fs)); a=int(index)-n+1
    if a<0 or index>=len(env): return {}
    x=np.asarray(env[a:index+1],float); ok=np.asarray(valid[a:index+1],bool)&np.isfinite(x)
    if ok.mean()<0.95: return {}
    # Downsample to 2 Hz after causal extraction; the history endpoint remains index.
    y=x[ok]; step=max(1,int(round(fs/2.0))); y=y[::step]
    if y.size<60: return {}
    y=detrend(y-np.mean(y)); p=np.abs(np.fft.rfft(y*np.hanning(y.size)))**2; f=np.fft.rfftfreq(y.size,1/2.0); m=(f>=0.017)&(f<=0.133)
    if not m.any(): return {}
    band=p[m]; total=float(p[1:].sum()) if p.size>1 else np.nan
    return {"eus_slow_modulation_power_0p017_0p133":float(np.trapz(band,f[m])),
            "eus_slow_modulation_depth":float((np.percentile(y,95)-np.percentile(y,5))/2.0),
            "eus_slow_periodicity":float(np.max(band)/max(total,C.EPSILON)) if np.isfinite(total) else np.nan}


def extract_v4_features(cycle: dict, index: int, onset_index: int | None = None,
                        decision_time_s: float | None = None) -> tuple[dict, str]:
    """Return all causal V4 features at one pressure sample."""
    p=np.asarray(cycle.get("bladder_pressure_mmHg",[]),float); t=np.asarray(cycle.get("t_abs_s",[]),float); idx=int(index)
    if idx<1 or idx>=len(p) or t.size!=p.size: return {}, "OUTSIDE_CYCLE"
    valid=np.asarray(cycle.get("cmg_valid_100hz",np.ones(p.size,bool)),bool)&np.isfinite(p)
    if not valid[idx]: return {}, "DATA_INVALID"
    onset=idx if onset_index is None else max(0,min(idx,int(onset_index)))
    base_end=min(idx,onset); base_start=base_end-int(round(C.BASELINE_WINDOW_S*C.DP_FS_HZ))
    if base_start<0: return {}, "PRESSURE_HISTORY_INSUFFICIENT"
    base=p[base_start:base_end]; base=base[valid[base_start:base_end]]
    if base.size<int(round(0.8*C.BASELINE_WINDOW_S*C.DP_FS_HZ)): return {}, "PRESSURE_HISTORY_INSUFFICIENT"
    med,scale=_finite_mad(base); delta=p-med; h=delta[max(0,onset):idx+1];
    if h.size<2 or not np.isfinite(h).all(): return {}, "PRESSURE_HISTORY_INSUFFICIENT"
    fs=C.DP_FS_HZ; d=np.diff(delta[max(0,idx-int(fs)):idx+1])*fs; d05=d[-int(.5*fs):]
    if d.size<fs: return {}, "PRESSURE_HISTORY_INSUFFICIENT"
    above=h>3.68; threshold_duration=float(np.sum(above)/fs)
    auc=float(np.trapz(np.maximum(h,0),dx=1/fs)); prev=h[:-int(.5*fs)] if h.size>int(.5*fs) else h
    pfeat={"p_current_delta":float(delta[idx]/scale),"p_peak_delta":float(np.max(h)/scale),"p_threshold_above_duration":threshold_duration,
           "p_slope_0p5s":float(np.mean(d05)/scale),"p_slope_1s":float(np.mean(d)/scale),"p_max_positive_dpdt":float(np.max(d)/scale),
           "p_positive_dpdt_occupancy":float(np.mean(d>0)),"p_auc":float(auc/scale),"p_auc_growth":float(auc/max((h.size-1)/fs*scale,C.EPSILON)),
           "pressure_curvature":float((np.mean(d05)-np.mean(d))/scale),"peak_to_current_drop":float((np.max(h)-h[-1])/scale),
           "p_local_variability":float(np.std(h[-int(fs):])/scale) if h.size>=fs else np.nan}
    event={"start_index":onset,"confirm_index":idx}
    ps,pr=causal_pressure_spectral_features(delta,idx,event,fs)
    pfeat.update({"pressure_power_0p2_0p6_rel":ps.get("pressure_power_0p2_0p6_rel",np.nan),"pressure_auc_0p2_20_rel":ps.get("pressure_auc_0p2_20_rel",np.nan)})
    cur=delta[idx-int(round(C.PRESSURE_SPEC_WINDOW_S*fs))+1:idx+1] if idx>=int(round(C.PRESSURE_SPEC_WINDOW_S*fs))-1 else np.array([])
    pfeat["pressure_spectral_entropy"]=_spectral_entropy(cur,fs)
    env=np.asarray(cycle.get("eus_envelope_100hz",cycle.get("eus_envelope_mV",[])),float); ev=np.asarray(cycle.get("eus_valid_100hz",np.ones(env.size,bool)),bool)
    efeat={k:np.nan for k in C.E0_FEATURES+C.E_FAST_FEATURES+C.E_SLOW_FEATURES+C.COUPLING_FEATURES}
    if env.size==p.size:
        eb=env[base_start:base_end]; eok=ev[base_start:base_end]&np.isfinite(eb); ec=env[max(0,idx-int(2*fs)+1):idx+1]; ecur=ev[max(0,idx-int(2*fs)+1):idx+1]&np.isfinite(ec)
        if eok.sum()>=int(10*fs) and ecur.mean()>=.95:
            em,es=_finite_mad(eb[eok]); z=(ec[ecur]-em)/es; efeat.update({"eus_relative_rms":float(np.sqrt(np.mean(z*z))),"eus_relative_amplitude":float((env[idx]-em)/es),"eus_envelope_slope":float(np.polyfit(np.arange(z.size)/fs,z,1)[0]) if z.size>1 else 0.0,"eus_tonic_occupancy":float(np.mean(z>3)),"eus_burst_occupancy":float(np.mean(z>5)),"eus_short_term_variability":float(np.std(z))})
        slow=_slow_features(env,ev,idx,fs); efeat.update(slow)
    if decision_time_s is None: decision_time_s=float(t[idx])
    rawfeat,rr=causal_eus_compact_bands(cycle,decision_time_s,event,C.EUS_FAST_BANDS,C.COMMON_EUS_HIGH_HZ)
    efeat.update({k:v for k,v in rawfeat.items() if k in efeat})
    # Coupling and latency use only the final two causal seconds.
    if env.size==p.size:
        a=max(1,idx-int(2*fs)+1); pp=p[a:idx+1]; ee=env[a:idx+1]; ok=ev[a:idx+1]&np.isfinite(pp)&np.isfinite(ee)
        if ok.sum()>5:
            em,es=_finite_mad(env[base_start:base_end][ev[base_start:base_end]&np.isfinite(env[base_start:base_end])]); ez=(ee[ok]-em)/es; dp=np.r_[0,np.diff(pp[ok])*fs]
            if np.std(ez)>0 and np.std(dp)>0: efeat["causal_pressure_eus_corr"]=float(np.corrcoef(ez,dp)[0,1])
            efeat["pressure_eus_coactivation"]=float(np.mean((dp>0)&(ez>3)))
            ppos=np.flatnonzero(dp>0); epos=np.flatnonzero(ez>3)
            if ppos.size and epos.size: efeat["eus_activation_latency_s"]=float((epos[0]-ppos[0])/fs)
    out={**pfeat,**efeat,"feature_window_start_s":float(t[max(0,idx-int(2*fs)+1)]),"feature_window_end_s":float(t[idx]),"baseline_window_start_s":float(t[base_start]),"baseline_window_end_s":float(t[base_end-1])}
    if not np.isfinite(np.asarray([out.get(k,np.nan) for k in C.P2_FEATURES],float)).all():
        # P2 can still be used independently if spectral history is short; only
        # return a reason when core pressure morphology is unavailable.
        if not np.isfinite(np.asarray([out.get(k,np.nan) for k in C.P1_FEATURES],float)).all(): return out,"PRESSURE_HISTORY_INSUFFICIENT"
    return out, ""
