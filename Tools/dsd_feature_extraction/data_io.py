"""Strict readers for the frozen 32-cycle cohort and native auxiliary channels."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import SUBJECT_CYCLES


def validate_manifest(input_root: Path) -> pd.DataFrame:
    manifest_path = input_root / "cycle_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(str(manifest_path))
    all_rows = pd.read_csv(manifest_path)
    rows = all_rows[all_rows["subject"].isin(SUBJECT_CYCLES)].copy()
    actual = rows.groupby("subject").size().to_dict()
    extras = sorted(set(rows["subject"]) - set(SUBJECT_CYCLES))
    missing_files = []
    for subject, expected in SUBJECT_CYCLES.items():
        subject_rows = rows[rows["subject"] == subject]
        if len(subject_rows) != expected:
            listing = sorted(str(p.relative_to(input_root)) for p in (input_root / subject).glob("*/cycle_100Hz.npz"))
            raise RuntimeError(f"Frozen cohort mismatch for {subject}: expected={expected}, actual={len(subject_rows)}, files={listing}")
        for _, row in subject_rows.iterrows():
            folder = input_root / subject / str(row["dsd_cycle_id"])
            for name in ("cycle_100Hz.npz", "cycle_native_eus.npz"):
                if not (folder / name).exists():
                    missing_files.append(str(folder / name))
    if extras or actual != SUBJECT_CYCLES or missing_files:
        raise RuntimeError(f"Frozen cohort invalid: actual={actual}, extras={extras}, missing={missing_files}")
    return rows.sort_values(["subject", "cycle_start_s"]).reset_index(drop=True)


def _scalar(data, key):
    return data[key].item() if np.asarray(data[key]).ndim == 0 else data[key]


def load_cycle(input_root: Path, row: pd.Series) -> Dict:
    subject, cycle_id = str(row["subject"]), str(row["dsd_cycle_id"])
    folder = input_root / subject / cycle_id
    with np.load(folder / "cycle_100Hz.npz", allow_pickle=False) as z:
        required = {"t_abs_s", "bladder_pressure_mmHg", "cmg_valid_100hz", "eus_valid_100hz"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{subject}/{cycle_id}: missing 100Hz fields {sorted(missing)}")
        cycle = {k: z[k].copy() for k in z.files}
    with np.load(folder / "cycle_native_eus.npz", allow_pickle=False) as z:
        required = {"eus_raw", "eus_fs", "t_eus_abs_s", "source_file"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{subject}/{cycle_id}: missing native EUS fields {sorted(missing)}")
        native = {k: z[k].copy() for k in z.files}
    if float(_scalar(cycle, "sample_rate_hz")) != 100.0:
        raise RuntimeError(f"{subject}/{cycle_id}: DP sample rate is not 100 Hz")
    cycle.update({"eus_raw_native": native["eus_raw"], "eus_fs_native": float(_scalar(native, "eus_fs")),
                  "t_eus_abs_native": native["t_eus_abs_s"], "native_source_file": str(_scalar(native, "source_file")),
                  "folder": folder, "manifest_row": row.to_dict()})
    return cycle


def load_pressure_cycle(input_root: Path, row: pd.Series) -> Dict:
    subject, cycle_id = str(row["subject"]), str(row["dsd_cycle_id"])
    path = input_root / subject / cycle_id / "cycle_100Hz.npz"
    with np.load(path, allow_pickle=False) as z:
        keys = ("subject", "dsd_cycle_id", "cycle_start_s", "cycle_end_s", "cycle_duration_s",
                "t_abs_s", "bladder_pressure_mmHg", "cmg_valid_100hz", "eus_valid_100hz", "sample_rate_hz")
        return {k: z[k].copy() for k in keys}


def resolve_native_volume(input_root: Path, subject: str, sample_cycle: Dict) -> Path:
    local = sample_cycle["folder"] / "native_volume.npz"
    candidates = [local]
    source_file = Path(sample_cycle["native_source_file"])
    candidates.append(source_file.parent / "pre_stim_urine_output.npz")
    candidates.append(input_root.parent / "baseline" / subject / "pre_stim_urine_output.npz")
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{subject}: native Volume unavailable; checked: {[str(p) for p in candidates]}")


def load_native_volume(path: Path, subject: str) -> Dict:
    with np.load(path, allow_pickle=False) as z:
        required = {"urine_output_raw", "sample_rate_hz", "units"}
        missing = required - set(z.files)
        if missing:
            raise RuntimeError(f"{path}: missing Volume fields {sorted(missing)}")
        raw = z["urine_output_raw"].astype(np.float64)
        fs = float(z["sample_rate_hz"])
        units = str(z["units"])
        time_s = z["time_s"].astype(np.float64) if "time_s" in z.files else np.arange(raw.size) / fs
    if fs <= 0 or raw.size != time_s.size:
        raise RuntimeError(f"{path}: invalid native Volume timebase")
    return {"subject": subject, "raw": raw, "time_s": time_s, "fs": fs, "units": units, "path": str(path)}


def write_json(path: Path, obj) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, (np.integer, np.bool_)):
            return value.item()
        if isinstance(value, np.floating):
            value = float(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    path.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
