"""Subject-aware urine evidence sources shared by cycle extraction and event census."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SUBJECT_REGISTRY


URINE_EVIDENCE_SOURCE = {
    subject: row["urine_source"]
    for subject, row in SUBJECT_REGISTRY.items()
    if row.get("dsd_confirmed")
}


@dataclass(frozen=True)
class UrineEvidence:
    subject: str
    source_type: str
    event_times_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    continuous_time_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    continuous_value: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    metadata: dict = field(default_factory=dict)
    model_input: bool = False

    @property
    def continuous_available(self) -> bool:
        return bool(self.continuous_time_s.size and self.continuous_value.size)


def _read_events(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _find_external_contract(subject: str, subject_dir: Path, explicit: Path | None = None) -> Path | None:
    candidates = [explicit, Path(subject_dir) / "urine_evidence_contract.json",
                  Path(subject_dir).parent.parent / "SPARC164_reaudit" / subject / "urine_evidence_contract.json"]
    return next((Path(path) for path in candidates if path is not None and Path(path).is_file()), None)


def load_urine_evidence(subject: str, subject_dir: Path, volume_display,
                        evidence_contract_path: Path | None = None) -> UrineEvidence:
    """Load the exact baseline-verified source without re-detecting or interpolating events."""
    if subject not in URINE_EVIDENCE_SOURCE:
        contract_path = _find_external_contract(subject, subject_dir, evidence_contract_path)
        if contract_path is None:
            raise ValueError(f"{subject}: external urine evidence contract is required")
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        source_type = str(contract.get("urine_source_type", contract.get("evidence_type", "UNRESOLVED")) or "UNRESOLVED")
        metadata = dict(contract.get("channel_metadata", {}))
        metadata.update({"contract_path": str(contract_path), "reason": contract.get("reason", ""),
                         "qc_status": contract.get("cycle_generation_allowed", False)})
        if source_type == "UNRESOLVED" or not contract.get("cycle_generation_allowed", False):
            raise ValueError(f"{subject}: urine evidence is {source_type}; cycle extraction is stopped")
        if source_type == "CONTINUOUS_WEIGHT":
            with np.load(Path(subject_dir) / "pre_stim_urine_output.npz", allow_pickle=False) as data:
                raw = data["urine_output_raw"].copy(); fs_hz = float(data["sample_rate_hz"])
                time_s = data["time_s"].copy()
            return UrineEvidence(subject, source_type, continuous_time_s=time_s,
                                 continuous_value=raw.astype(np.float32), metadata=metadata)
        episode_path = contract_path.parent / "discrete_volume_episodes.csv"
        transition_path = contract_path.parent / "urine_transition_events.csv"
        rows = _read_events(episode_path) if episode_path.exists() else _read_events(transition_path) if transition_path.exists() else []
        times = np.asarray([float(row.get("onset_s", row.get("transition_time_s"))) for row in rows
                            if row.get("match_status", "MATCHED") == "MATCHED"
                            and row.get("onset_s", row.get("transition_time_s", "")) not in {"", "nan"}], dtype=np.float64)
        return UrineEvidence(subject, source_type, event_times_s=times, metadata=metadata)
    expected = URINE_EVIDENCE_SOURCE[subject]
    info = json.loads((subject_dir / "urine_output_info.json").read_text(encoding="utf-8"))
    if expected == "LEAK_BUTTON_EVENT":
        rows = _read_events(subject_dir / "pre_stim_events.csv")
        drop_rows = [row for row in rows if row["event_type"] == "LEAK"
                     and row.get("value") == "drop_button_rise"]
        metadata_rows = [row for row in rows if row["event_type"] == "KEYBOARD"]
        times = np.asarray([float(row["time_s"]) for row in drop_rows], dtype=np.float64)
        if info.get("qc_status") != "PASS" or not times.size:
            raise ValueError(f"{subject}: baseline button/drop evidence is unavailable")
        return UrineEvidence(
            subject=subject,
            source_type="LEAK_BUTTON_EVENT",
            event_times_s=times,
            metadata={
                "physical_source_channel": int(info["source_channel"]),
                "physical_source_title": info["source_title"],
                "physical_source_type": "Adc",
                "physical_source_comment": "button",
                "event_derivation": "BASELINE_PARSE_DROP_BUTTON_RISING_EDGE",
                "baseline_event_type": "LEAK",
                "raw_event_count": int(times.size),
                "keyboard_metadata_channel": 30,
                "keyboard_metadata_event_count": len(metadata_rows),
                "keyboard_metadata_times_s": [float(row["time_s"]) for row in metadata_rows],
                "baseline_confirmation_rule": "CMG_CONTRACTION_PLUS_AT_LEAST_TWO_DROPS_IN_START_MINUS_1_TO_END_PLUS_5",
                "qc_status": info["qc_status"],
            },
            model_input=False,
        )

    if info.get("qc_status") != "CONFIRMED_URINE_OUTPUT":
        raise ValueError(f"{subject}: continuous urine source has not passed baseline QC")
    with np.load(subject_dir / "pre_stim_urine_output.npz", allow_pickle=False) as data:
        raw = data["urine_output_raw"].copy()
        fs_hz = float(data["sample_rate_hz"])
        units = str(data["units"])
    time_s, trace, _ = volume_display(raw, fs_hz)
    return UrineEvidence(
        subject=subject,
        source_type="CONTINUOUS_WEIGHT",
        continuous_time_s=time_s,
        continuous_value=trace,
        metadata={
            "physical_source_channel": int(info["source_channel"]),
            "physical_source_title": info["source_title"],
            "physical_source_type": "Adc",
            "source_units": units,
            "sample_rate_hz": fs_hz,
            "qc_status": info["qc_status"],
        },
        model_input=False,
    )


def stable_phase_inputs(evidence: UrineEvidence) -> tuple[str, dict]:
    if evidence.source_type == "LEAK_BUTTON_EVENT":
        return "LEAK", {"drop_times": evidence.event_times_s}
    if evidence.source_type == "CONTINUOUS_WEIGHT":
        return "VOLUME", {
            "volume_time": evidence.continuous_time_s,
            "volume_trace": evidence.continuous_value,
        }
    if evidence.source_type in {"DISCRETE_STABLE_VOLUME", "VOID_MARKER_EVENT"}:
        return "DISCRETE_VOID_EVENT", {"marker_times": evidence.event_times_s}
    raise ValueError(f"Unsupported urine source: {evidence.source_type}")
