"""V3.2 input adapter built on the frozen V3.1 raw-cycle adapter."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .mechanism_adapter import (
    load_development_streams, build_delayed_features,
)
from Tools.dsd_feature_extraction.data_io import load_cycle
from . import config as C
from .spectral_features import (
    causal_pressure_spectral_features, causal_raw_eus_stft_features,
    causal_eus_compact_bands, stft_frequency_names,
)


def load_v32_inputs(cycles_338: Path = C.DEFAULT_338_CYCLES,
                    reference_338: Path = C.DEFAULT_338_REFERENCE,
                    cycles_164: Path = C.DEFAULT_164_CYCLES,
                    labels_164: Path = C.DEFAULT_164_LABELS):
    """Load the exact registered 338 + 164 cohort and frozen event labels."""
    cache, events, manifest, files = load_development_streams(
        Path(cycles_338), Path(reference_338), Path(cycles_164), Path(labels_164))
    return cache, events, manifest, files


def _row_event(row):
    event = row.to_dict()
    # v31 rows use these detector fields; all are causal descriptors.
    return event


def build_v32_features(cache: dict, events: pd.DataFrame,
                       delays=C.DIAGNOSTIC_DELAYS_S) -> tuple[pd.DataFrame, dict]:
    """Build frozen V3.1 time-domain fields plus preregistered V3.2 spectra."""
    delayed = build_delayed_features(cache, events, delays=delays)
    native_rates = [float(item["cycle"].get("eus_fs_native", np.nan)) for item in cache.values()]
    high, bands, excluded = C.common_eus_bands(native_rates)
    stft_names = stft_frequency_names(high)
    event_map = events.set_index("event_uid").to_dict(orient="index") if "event_uid" in events else {}
    rows = []
    for row in delayed.to_dict(orient="records"):
        subject, cycle_id = str(row["subject"]), str(row["cycle_id"])
        item = cache[(subject, cycle_id)]
        cycle = item["cycle"]
        idx = int(row["decision_index"]) if np.isfinite(row.get("decision_index", np.nan)) else -1
        event = dict(event_map.get(str(row.get("event_uid", "")), {})); event.update(_row_event(row))
        row.update({name: np.nan for name in C.PRESSURE_SPECTRAL_FEATURES + stft_names + C.EUS_COMPACT_FEATURES})
        row["m1_scorable"] = False; row["m2_scorable"] = False; row["m3_scorable"] = False
        row["spectral_failure_reason_v32"] = "BASE_EVENT_UNSCORABLE" if not bool(row.get("base_eligible", False)) else ""
        if bool(row.get("base_eligible", False)) and idx >= 0:
            p, reason = causal_pressure_spectral_features(item["delta"], idx, event)
            if p:
                row.update({k: v for k, v in p.items() if k in row or k in C.PRESSURE_SPECTRAL_FEATURES})
                row["m1_scorable"] = all(np.isfinite(row.get(k, np.nan)) for k in C.PRESSURE_SPECTRAL_FEATURES)
            else:
                row["spectral_failure_reason_v32"] = reason
            # Native EUS is intentionally used only for the spectral models.
            if np.isclose(float(row["decision_delay_s"]), C.PRIMARY_DELAY_S):
                eus, ereason = causal_raw_eus_stft_features(cycle, float(row["decision_time_s"]), event, high)
                compact, creason = causal_eus_compact_bands(cycle, float(row["decision_time_s"]), event, bands, high)
                if eus:
                    row.update(eus)
                    row["m2_scorable"] = np.isfinite(np.asarray([row.get(k, np.nan) for k in stft_names], dtype=float)).all()
                if compact:
                    row.update(compact)
                if not eus and not row["spectral_failure_reason_v32"]:
                    row["spectral_failure_reason_v32"] = ereason
                if not compact and not row["spectral_failure_reason_v32"]:
                    row["spectral_failure_reason_v32"] = creason
            row["m3_scorable"] = row["m1_scorable"] and np.isfinite(
                np.asarray([row.get(k, np.nan) for k in C.M3_FEATURES], dtype=float)).all()
        rows.append(row)
    out = pd.DataFrame(rows)
    out["target"] = out["teacher_label"].map(C.LABEL_TO_TARGET)
    # Explicit failure reason hierarchy; structural missingness is never hidden by imputation.
    for name, cols in (("M1", C.M1_FEATURES), ("M3", C.M3_FEATURES)):
        complete = np.isfinite(out[list(cols)].to_numpy(float)).all(axis=1)
        flag = out["base_eligible"].astype(bool) & complete
        out[f"{name.lower()}_scorable"] = flag
        out[f"{name.lower()}_failure_reason"] = np.where(
            ~out["base_eligible"].astype(bool), out["base_failure_reason"],
            np.where(~complete, "PRESSURE_SPECTRAL_BASELINE_INSUFFICIENT", ""))
    m2_complete = np.isfinite(out[list(tuple(C.PE_FEATURES) + stft_names)].to_numpy(float)).all(axis=1)
    out["m2_scorable"] = out["base_eligible"].astype(bool) & m2_complete
    out["m2_failure_reason"] = np.where(~out["base_eligible"].astype(bool), out["base_failure_reason"],
                                         np.where(~m2_complete, "EUS_INVALID", ""))
    metadata = {
        "native_eus_rates_hz": sorted(set(round(x, 6) for x in native_rates if np.isfinite(x))),
        "pressure_rates_hz": [C.DP_FS_HZ], "common_eus_high_hz": high,
        "included_eus_bands": [list(x) for x in bands],
        "excluded_eus_bands": [list(x) for x in excluded],
        "m2_frequency_features": list(stft_names),
        "m3_features": list(C.M3_FEATURES), "m5_features": list(C.M5_FEATURES),
    }
    return out, metadata


def load_frozen_v31_features(v31_root: Path = C.DEFAULT_V31_ROOT):
    """Load the immutable V3.1 feature artifact and source manifests.

    This keeps the development run memory-bounded: native EUS is loaded one
    cycle at a time by :func:`augment_frozen_features`.
    """
    root = Path(v31_root)
    base = pd.read_csv(root / "event_features_delayed_v31.csv")
    events = pd.read_csv(root / "source_events_v31.csv")
    manifest = pd.read_csv(root / "source_manifest_v31.csv")
    # Absolute paths are already present in the frozen manifest; retain only
    # registered cohort rows.
    manifest = manifest[manifest.subject.isin(C.SUBJECTS)].copy()
    events = events[events.subject.isin(C.SUBJECTS)].copy()
    paths = {}
    for r in manifest.itertuples(index=False):
        p100 = getattr(r, "cycle_100hz_path", np.nan); peus = getattr(r, "cycle_native_eus_path", np.nan)
        if not isinstance(p100, str) or p100.casefold() == "nan":
            p100 = str(C.ROOT / "data" / "SPARC164_cycles" / str(r.subject) / str(r.cycle_id) / "cycle_100Hz.npz")
        if not isinstance(peus, str) or peus.casefold() == "nan":
            peus = str(C.ROOT / "data" / "SPARC164_cycles" / str(r.subject) / str(r.cycle_id) / "cycle_native_eus.npz")
        paths[(str(r.subject), str(r.cycle_id))] = {"cycle_100hz_path": Path(p100), "cycle_native_eus_path": Path(peus)}
    return base, events, manifest, paths


def _load_pair(paths):
    p = Path(paths["cycle_100hz_path"]); q = Path(paths["cycle_native_eus_path"])
    with np.load(p, allow_pickle=False) as z:
        cycle = {k: z[k].copy() for k in z.files}
    with np.load(q, allow_pickle=False) as z:
        cycle.update({"eus_raw_native": z["eus_raw"].copy(), "eus_fs_native": float(np.asarray(z["eus_fs"]).item()),
                      "t_eus_abs_native": z["t_eus_abs_s"].copy()})
    # A causal baseline-centered pressure stream is sufficient for spectral
    # ratios and does not depend on future labels or recovery information.
    pressure = np.asarray(cycle["bladder_pressure_mmHg"], dtype=float)
    valid = np.asarray(cycle.get("cmg_valid_100hz", np.ones(pressure.size, bool)), bool) & np.isfinite(pressure)
    base = pressure[valid][:min(valid.sum(), int(C.BASELINE_WINDOW_S * C.DP_FS_HZ))]
    delta = pressure - (float(np.median(base)) if base.size else 0.0)
    return {"cycle": cycle, "delta": delta}


def augment_frozen_features(base: pd.DataFrame, events: pd.DataFrame, paths: dict,
                            delays=C.DIAGNOSTIC_DELAYS_S) -> tuple[pd.DataFrame, dict]:
    """Augment immutable V3.1 rows with V3.2 spectra, loading one cycle at a time."""
    out = base[base.decision_delay_s.isin(tuple(float(x) for x in delays))].copy()
    rates=[]
    for v in paths.values():
        try:
            with np.load(v["cycle_native_eus_path"], allow_pickle=False) as z: rates.append(float(np.asarray(z["eus_fs"]).item()))
        except Exception: pass
    high,bands,excluded=C.common_eus_bands(rates); stft_names=stft_frequency_names(high)
    event_map=events.set_index("event_uid").to_dict(orient="index")
    for name in C.PRESSURE_SPECTRAL_FEATURES+stft_names+C.EUS_COMPACT_FEATURES: out[name]=np.nan
    out["m1_scorable"]=False; out["m2_scorable"]=False; out["m3_scorable"]=False; out["spectral_failure_reason_v32"]=""
    for key, inds in out.groupby(["subject","cycle_id"]).groups.items():
        item=_load_pair(paths[(str(key[0]),str(key[1]))]); cycle=item["cycle"]; delta=item["delta"]
        for i in inds:
            r=out.loc[i]; idx=int(r.decision_index) if np.isfinite(r.decision_index) else -1
            ev=dict(event_map.get(str(r.event_uid),{})); ev.update(r.to_dict())
            if not bool(r.base_eligible) or idx<0: out.loc[i,"spectral_failure_reason_v32"]="BASE_EVENT_UNSCORABLE"; continue
            p,pr=causal_pressure_spectral_features(delta,idx,ev); out.loc[i,list(C.PRESSURE_SPECTRAL_FEATURES)]=[p.get(k,np.nan) for k in C.PRESSURE_SPECTRAL_FEATURES]
            out.loc[i,"m1_scorable"]=bool(p)
            if np.isclose(float(r.decision_delay_s),C.PRIMARY_DELAY_S):
                e,er=causal_raw_eus_stft_features(cycle,float(r.decision_time_s),ev,high); c,cr=causal_eus_compact_bands(cycle,float(r.decision_time_s),ev,bands,high)
                for k,v in e.items():
                    if k in out.columns: out.loc[i,k]=v
                for k,v in c.items():
                    if k in out.columns: out.loc[i,k]=v
                out.loc[i,"m2_scorable"]=bool(e) and np.isfinite([e.get(k,np.nan) for k in stft_names]).all()
                if not e: out.loc[i,"spectral_failure_reason_v32"]=er
            vals=np.asarray([out.loc[i].get(k,np.nan) for k in C.M3_FEATURES],float)
            out.loc[i,"m3_scorable"]=bool(np.isfinite(vals).all())
    out["target"]=out.teacher_label.map(C.LABEL_TO_TARGET)
    for name,cols in (("M1",C.M1_FEATURES),("M3",C.M3_FEATURES)):
        complete=np.isfinite(out[list(cols)].to_numpy(float)).all(axis=1); flag=out.base_eligible.astype(bool)&complete; out[f"{name.lower()}_scorable"]=flag; out[f"{name.lower()}_failure_reason"]=np.where(~out.base_eligible.astype(bool),out.base_failure_reason,np.where(~complete,"STRUCTURAL_FEATURE_MISSING",""))
    m2_complete=np.isfinite(out[list(tuple(C.PE_FEATURES)+stft_names)].to_numpy(float)).all(axis=1); out["m2_scorable"]=out.base_eligible.astype(bool)&m2_complete; out["m2_failure_reason"]=np.where(~out.base_eligible.astype(bool),out.base_failure_reason,np.where(~m2_complete,"EUS_INVALID",""))
    return out,{"native_eus_rates_hz":sorted(set(rates)),"pressure_rates_hz":[C.DP_FS_HZ],"common_eus_high_hz":high,"included_eus_bands":[list(x) for x in bands],"excluded_eus_bands":[list(x) for x in excluded],"m2_frequency_features":list(stft_names),"m3_features":list(C.M3_FEATURES),"m5_features":list(C.M5_FEATURES)}
