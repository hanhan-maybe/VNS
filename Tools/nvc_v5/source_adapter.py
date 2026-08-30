"""V4-local reader for frozen V3.1 artifacts and raw cycle pairs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def load_frozen_v31_features(v31_root: Path = C.V31_ROOT):
    root = Path(v31_root)
    base = pd.read_csv(root / "event_features_delayed_v31.csv")
    events = pd.read_csv(root / "source_events_v31.csv")
    manifest = pd.read_csv(root / "source_manifest_v31.csv")
    manifest = manifest[manifest.subject.isin(C.SUBJECTS)].copy()
    events = events[events.subject.isin(C.SUBJECTS)].copy()
    paths = {}
    for row in manifest.itertuples(index=False):
        p100 = getattr(row, "cycle_100hz_path", np.nan)
        peus = getattr(row, "cycle_native_eus_path", np.nan)
        if not isinstance(p100, str) or p100.casefold() == "nan":
            p100 = str(C.ROOT / "data" / "SPARC164_cycles" / str(row.subject) /
                       str(row.cycle_id) / "cycle_100Hz.npz")
        if not isinstance(peus, str) or peus.casefold() == "nan":
            peus = str(C.ROOT / "data" / "SPARC164_cycles" / str(row.subject) /
                       str(row.cycle_id) / "cycle_native_eus.npz")
        paths[(str(row.subject), str(row.cycle_id))] = {
            "cycle_100hz_path": Path(p100),
            "cycle_native_eus_path": Path(peus),
        }
    return base, events, manifest, paths


def _load_pair(paths):
    with np.load(Path(paths["cycle_100hz_path"]), allow_pickle=False) as archive:
        cycle = {key: archive[key].copy() for key in archive.files}
    with np.load(Path(paths["cycle_native_eus_path"]), allow_pickle=False) as archive:
        cycle.update({
            "eus_raw_native": archive["eus_raw"].copy(),
            "eus_fs_native": float(np.asarray(archive["eus_fs"]).item()),
            "t_eus_abs_native": archive["t_eus_abs_s"].copy(),
        })
    pressure = np.asarray(cycle["bladder_pressure_mmHg"], dtype=float)
    valid = np.asarray(cycle.get("cmg_valid_100hz", np.ones(pressure.size, bool)), bool)
    valid &= np.isfinite(pressure)
    baseline = pressure[valid][:min(valid.sum(), int(C.BASELINE_WINDOW_S * C.DP_FS_HZ))]
    delta = pressure - (float(np.median(baseline)) if baseline.size else 0.0)
    return {"cycle": cycle, "delta": delta}
