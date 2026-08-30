"""Shared configuration and subject provenance for the SPARC338 two-stage pipeline.

Paths can be overridden with environment variables so that the same code runs on
the current Windows workstation and on a validation machine without source edits.
"""
from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SPARC338_DATA_ROOT", PROJECT_ROOT / "data"))
RAW_ROOT = Path(os.environ.get("SPARC338_RAW_ROOT", r"D:\Sparc"))
BASELINE_ROOT = Path(os.environ.get("SPARC338_BASELINE_ROOT", DATA_ROOT / "baseline"))
DSD_CYCLES_ROOT = Path(os.environ.get("SPARC338_DSD_CYCLES_ROOT", DATA_ROOT / "DSD_cycles"))
VALIDATION_ROOT = Path(os.environ.get("SPARC338_VALIDATION_ROOT", DATA_ROOT / "dsd_validation"))

SCI_SUBJECTS = (
    "STxF14", "STxF21", "STxF22", "STxF23", "STxF24",
    "STxF26", "STxF27", "STxF29", "STxF30",
)
DSD_SUBJECTS = ("STxF21", "STxF26", "STxF27", "STxF29")

DISPLAY_FS_HZ = 100.0
STABLE_ONSET_SUPPORT_CYCLES = 3
REFERENCE_BASELINE_MAX_CYCLES = 5
REFERENCE_BASELINE_MIN_CYCLES = 3
BOUNDARY_METHOD = "PREVIOUS_SETTLED_VOID_END_TO_CURRENT_SETTLED_VOID_END"


# Audited dataset-specific decisions are configuration/provenance, not algorithm code.
# Keep the exact existing decisions so refactoring does not silently relabel Dataset338.
SUBJECT_REGISTRY = {
    "STxF14": {
        "dsd_confirmed": False,
        "urine_source": "LEAK_BUTTON_EVENT",
        "urine_review_status": "PASS",
        "review_note": "Visual review supports discrete button/drop pulses near repeated CMG contractions.",
    },
    "STxF21": {
        "dsd_confirmed": True,
        "urine_source": "LEAK_BUTTON_EVENT",
        "urine_review_status": "PASS",
        "review_note": "Visual review supports discrete button/drop pulses near repeated CMG contractions.",
        "keyboard_role": "METADATA_ONLY",
        "void_evidence_policy": "CMG_PLUS_CHANNEL5_LEAK_BUTTON_EVENTS",
    },
    "STxF22": {
        "dsd_confirmed": False,
        "urine_source": "LEAK_BUTTON_EVENT",
        "urine_review_status": "PASS",
        "review_note": "Visual review supports discrete button/drop pulses near repeated CMG contractions.",
    },
    "STxF23": {
        "dsd_confirmed": False,
        "urine_source": "UNRESOLVED",
        "urine_review_status": "UNRESOLVED",
        "review_note": "No separated button high state; drop timing remains unresolved.",
    },
    "STxF24": {
        "dsd_confirmed": False,
        "urine_source": "NOT_URINE_OUTPUT",
        "urine_review_status": "NOT_URINE_OUTPUT",
        "review_note": "Visual review: 0/5 contraction-linked changes; long monotonic change is independent of CMG contractions.",
    },
    "STxF26": {
        "dsd_confirmed": True,
        "urine_source": "CONTINUOUS_WEIGHT",
        "urine_review_status": "CONFIRMED_URINE_OUTPUT",
        "review_note": "Visual review: repeated staircase increases aligned with CMG contractions.",
    },
    "STxF27": {
        "dsd_confirmed": True,
        "urine_source": "CONTINUOUS_WEIGHT",
        "urine_review_status": "CONFIRMED_URINE_OUTPUT",
        "review_note": "Visual review: repeated contraction-linked ramps; one selected early CMG artifact had no change.",
    },
    "STxF29": {
        "dsd_confirmed": True,
        "urine_source": "CONTINUOUS_WEIGHT",
        "urine_review_status": "CONFIRMED_URINE_OUTPUT",
        "review_note": "Visual review: 5/5 contraction-linked staircase/ramp increases.",
    },
    "STxF30": {
        "dsd_confirmed": False,
        "urine_source": "CONTINUOUS_WEIGHT",
        "urine_review_status": "CONFIRMED_URINE_OUTPUT",
        "review_note": "Visual review: repeated contraction-linked staircase increases; one early CMG artifact had no change.",
    },
}


def validate_registry() -> None:
    missing = set(SCI_SUBJECTS) - set(SUBJECT_REGISTRY)
    configured_dsd = {
        subject for subject, row in SUBJECT_REGISTRY.items() if row.get("dsd_confirmed")
    }
    if missing:
        raise ValueError(f"Subject registry is missing: {sorted(missing)}")
    if configured_dsd != set(DSD_SUBJECTS):
        raise ValueError(
            f"DSD registry mismatch: configured={sorted(configured_dsd)} expected={sorted(DSD_SUBJECTS)}"
        )


validate_registry()
