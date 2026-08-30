"""Acceptance tests for the audited STxF21 subject-specific urine correction."""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "Tools"
for root in (PROJECT_ROOT, TOOLS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from sparc338_urine_output import parse_drop_button  # noqa: E402
from Tools.dsd_cycle_extraction.urine_evidence_adapter import (  # noqa: E402
    URINE_EVIDENCE_SOURCE, load_urine_evidence,
)
from Tools.dsd_prevoid_event_census.config import INPUT_ROOT, OUTPUT_ROOT  # noqa: E402
from Tools.dsd_prevoid_event_census.data_loader import (  # noqa: E402
    audit_schema, discover_cycle_records, load_cycle,
)
from Tools.dsd_prevoid_event_census.dp_contraction_detector import detect_dp_contractions  # noqa: E402
from Tools.dsd_prevoid_event_census.event_features import characterize_events  # noqa: E402
from Tools.dsd_prevoid_event_census.fvol_outcome_classifier import associate_prevoid_outcomes  # noqa: E402


BASELINE = PROJECT_ROOT / "data" / "baseline" / "STxF21"
DSD = PROJECT_ROOT / "data" / "DSD_cycles"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class STxF21KeyboardUrineFixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = discover_cycle_records()
        cls.schema = audit_schema(cls.records)
        cls.stx_records = [row for row in cls.records if row["subject"] == "STxF21"]
        cls.cycle = load_cycle(cls.stx_records[3], cls.schema["resolved_field_mapping"])

    def test_01_STxF21_uses_keyboard_urine_source(self) -> None:
        self.assertEqual(URINE_EVIDENCE_SOURCE["STxF21"], "LEAK_BUTTON_EVENT")
        self.assertEqual(self.cycle["urine_source_type"], "LEAK_BUTTON_EVENT")

    def test_02_STxF21_does_not_use_continuous_fvol_classifier(self) -> None:
        search_end = self.cycle["time_s"].searchsorted(float(self.cycle["record"]["void_start_s"]))
        detection = detect_dp_contractions(self.cycle["dp_clean"][:search_end], self.cycle["fs_hz"])
        events = characterize_events(detection, self.cycle)
        with patch("Tools.dsd_prevoid_event_census.fvol_outcome_classifier.prepare_continuous_fvol",
                   side_effect=AssertionError("continuous classifier called")):
            _, prepared = associate_prevoid_outcomes(events, self.cycle)
        self.assertIsNone(prepared)

    def test_03_STxF21_keyboard_matches_baseline_logic(self) -> None:
        with np.load(BASELINE / "pre_stim_raw.npz", allow_pickle=False) as data:
            expected, status, _ = parse_drop_button(data["leak_raw"], float(data["leak_fs_hz"]))
        evidence = load_urine_evidence("STxF21", BASELINE, lambda *_: None)
        self.assertEqual(status, "PASS")
        np.testing.assert_array_equal(evidence.event_times_s, expected)

    def test_04_STxF21_no_fake_fvol_interpolation(self) -> None:
        for row in self.stx_records:
            with np.load(row["npz_path"], allow_pickle=False) as data:
                self.assertFalse(bool(data["fvol_continuous_available"]))
                self.assertFalse(np.isfinite(data["urine_output_auxiliary_100hz"]).any())

    def test_05_STxF21_quicklook_uses_keyboard_panel(self) -> None:
        schema = json.loads((OUTPUT_ROOT / "plot_schema_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["STxF21"]["cycle_overview_panels"][2], "Cumulative urine drops")
        self.assertTrue((DSD / "STxF21" / "B01" / "quicklook.png").is_file())

    def test_06_STxF21_census_overview_uses_keyboard_panel(self) -> None:
        schema = json.loads((OUTPUT_ROOT / "plot_schema_audit.json").read_text(encoding="utf-8"))
        self.assertFalse(schema["STxF21"]["continuous_fvol_panel"])
        self.assertTrue((OUTPUT_ROOT / "quicklook" / "STxF21_B01_event_overview.png").is_file())

    def test_07_STxF21_event_atlas_no_dfvol_panel(self) -> None:
        schema = json.loads((OUTPUT_ROOT / "plot_schema_audit.json").read_text(encoding="utf-8"))
        self.assertFalse(schema["STxF21"]["dfvol_panel"])
        self.assertNotIn("dFVol/dt", schema["STxF21"]["event_atlas_panels"])

    def test_08_dp_candidate_independent_of_keyboard(self) -> None:
        end = self.cycle["time_s"].searchsorted(float(self.cycle["record"]["void_start_s"]))
        first = detect_dp_contractions(self.cycle["dp_clean"][:end], self.cycle["fs_hz"])
        changed = dict(self.cycle, fvol_events=np.array([0.0, 1.0e9]))
        second = detect_dp_contractions(changed["dp_clean"][:end], changed["fs_hz"])
        project = lambda result: [(row["onset_index"], row["peak_index"], row["end_index"])
                                  for row in result["events"]]
        self.assertEqual(project(first), project(second))

    def test_09_keyboard_only_changes_outcome_association(self) -> None:
        end = self.cycle["time_s"].searchsorted(float(self.cycle["record"]["void_start_s"]))
        detection = detect_dp_contractions(self.cycle["dp_clean"][:end], self.cycle["fs_hz"])
        actual = characterize_events(detection, self.cycle)
        empty = characterize_events(detection, self.cycle)
        actual, _ = associate_prevoid_outcomes(actual, self.cycle)
        empty, _ = associate_prevoid_outcomes(empty, dict(self.cycle, fvol_events=np.empty(0)))
        boundaries = lambda rows: [(row["onset_index"], row["peak_index"], row["end_index"])
                                   for row in rows]
        self.assertEqual(boundaries(actual), boundaries(empty))
        self.assertTrue(all(row["urine_source_type"] == "LEAK_BUTTON_EVENT" for row in actual))

    def test_10_STxF21_final_void_keyboard_sanity(self) -> None:
        rows = [row for row in read_csv(OUTPUT_ROOT / "final_void_urine_sanity.csv")
                if row["subject"] == "STxF21"]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["urine_source_type"] == "LEAK_BUTTON_EVENT" for row in rows))
        self.assertTrue(all(row["keyboard_evidence_confirmed"] == "True" for row in rows))
        self.assertTrue(all(row["fvol_step_confirmed"] == "False" for row in rows))

    def test_11_confirmed_void_before_after_report_exists(self) -> None:
        rows = read_csv(DSD / "STxF21_confirmed_void_comparison.csv")
        self.assertEqual(len(rows), 11)
        self.assertTrue(all(row["matched"] == "True" for row in rows))

    def test_12_cycle_boundary_propagation(self) -> None:
        current = [row for row in read_csv(INPUT_ROOT / "cycle_manifest.csv")
                   if row["subject"] == "STxF21"]
        comparison = read_csv(DSD / "STxF21_confirmed_void_comparison.csv")
        self.assertEqual(len(current), 8)
        self.assertTrue(all(float(row["cycle_start_s"]) < float(row["cycle_end_s"])
                            for row in current))
        self.assertEqual(len(comparison), 11)
        self.assertTrue(all(row["matched"] == "True" for row in comparison))

    def test_13_non_STxF21_regression(self) -> None:
        rows = read_csv(OUTPUT_ROOT / "non_STxF21_regression_check.csv")
        self.assertGreaterEqual(len(rows), 7)
        self.assertTrue(all(row["status"] == "PASS" for row in rows))

    def test_14_urine_source_metadata_saved(self) -> None:
        with np.load(self.stx_records[0]["npz_path"], allow_pickle=False) as data:
            self.assertEqual(str(data["urine_source_type"]), "LEAK_BUTTON_EVENT")
            self.assertIn("urine_event_times_abs_s", data.files)
            self.assertIn("urine_event_times_cycle_s", data.files)
            metadata = json.loads(str(data["urine_source_metadata_json"]))
            self.assertEqual(metadata["event_derivation"], "BASELINE_PARSE_DROP_BUTTON_RISING_EDGE")

    def test_15_model_input_false_for_all_urine_sources(self) -> None:
        for row in self.records:
            with np.load(row["npz_path"], allow_pickle=False) as data:
                self.assertFalse(bool(data["urine_output_model_input"]))
        events = read_csv(OUTPUT_ROOT / "all_prevoid_dp_contractions.csv")
        self.assertTrue(all(row["urine_model_input"] == "False" for row in events))

    def test_16_protected_raw_files_unchanged(self) -> None:
        rerun_audit = json.loads(
            (OUTPUT_ROOT / "three_stage_rerun_integrity.json").read_text(encoding="utf-8")
        )
        census_audit = json.loads((OUTPUT_ROOT / "protected_files_sha256.json").read_text(encoding="utf-8"))
        self.assertTrue(rerun_audit["validation_sha256_all_identical"])
        self.assertTrue(rerun_audit["raw_size_mtime_all_identical"])
        self.assertTrue(census_audit["sha256_all_identical"])
        self.assertEqual(rerun_audit["validation_changed_files"], [])
        self.assertEqual(rerun_audit["raw_changed_files"], [])
        self.assertEqual(census_audit["changed_files"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
