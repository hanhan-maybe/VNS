"""Synthetic tests for the refactored two-stage contracts; no Dataset338 files required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dsd_cycle_extraction.cycle_qc import assign_cycle_statuses, nvc_eligible_cycles
from dsd_cycle_extraction.stable_cycle_extractor import cycle_arrays
from sparc338_common import commit_directory, make_staging_directory
from sparc338_config import DSD_SUBJECTS, SUBJECT_REGISTRY
from sparc338_preprocessing import preprocess_bladder


def stable_cycle(index: int, artifact: bool = False) -> dict:
    start = float(index * 10)
    return {
        "complete_cycle": True,
        "stability_candidate": "STABLE_CANDIDATE",
        "artifact_overlap": artifact,
        "data_gap_flag": False,
        "cycle_start_s": start,
        "cycle_end_s": start + 9.0,
        "first_stim_s": 200.0,
        "exclusion_reason": "",
    }


class TwoStageRefactorTests(unittest.TestCase):
    def test_nvc_eligibility_is_independent_of_sustained_baseline(self) -> None:
        cycles = [stable_cycle(index) for index in range(3)]
        cycles[1]["stability_candidate"] = "TRANSITIONAL"
        for row in cycles:
            row["confirmed_void"] = True
            row["cycle_duration_s"] = 45.0
        assigned, first = assign_cycle_statuses(cycles)
        self.assertIsNone(first)
        self.assertEqual(len(nvc_eligible_cycles(assigned)), 3)
        self.assertEqual(assigned[1]["nvc_quality_status"],
                         "NVC_ELIGIBLE_STATISTICAL_REVIEW")

    def test_nvc_hard_qc_still_excludes_artifact_and_gap(self) -> None:
        cycles = [stable_cycle(index) for index in range(3)]
        for row in cycles:
            row["confirmed_void"] = True
            row["cycle_duration_s"] = 45.0
        cycles[0]["artifact_overlap"] = True
        cycles[1]["data_gap_flag"] = True
        assigned, _ = assign_cycle_statuses(cycles)
        self.assertEqual(len(nvc_eligible_cycles(assigned)), 1)
        self.assertFalse(assigned[0]["nvc_eligible"])
        self.assertFalse(assigned[1]["nvc_eligible"])
        self.assertTrue(assigned[2]["nvc_eligible"])

    def test_preprocessing_prefix_is_future_independent(self) -> None:
        fs = 1000.0
        rng = np.random.default_rng(3)
        prefix = rng.normal(size=5000)
        first, _ = preprocess_bladder(np.r_[prefix, np.zeros(5000)], fs)
        second, _ = preprocess_bladder(np.r_[prefix, np.full(5000, 100.0)], fs)
        prefix_outputs = int(np.ceil(len(prefix) / fs * 100.0))
        np.testing.assert_allclose(first[:prefix_outputs], second[:prefix_outputs], atol=0, rtol=0)

    def test_analysis_and_reference_cycles_are_distinct(self) -> None:
        cycles = [stable_cycle(index) for index in range(10)]
        assigned, first = assign_cycle_statuses(cycles)
        self.assertEqual(first, 0)
        self.assertEqual(sum(row["cycle_status"] == "PASS_STABLE" for row in assigned), 10)
        reference = [row for row in assigned if row["reference_baseline"]]
        self.assertEqual(len(reference), 5)
        self.assertEqual([row["dsd_cycle_id"] for row in reference], ["B06", "B07", "B08", "B09", "B10"])

    def test_latest_stable_run_controls_reference_only(self) -> None:
        cycles = [stable_cycle(index, artifact=(index == 5)) for index in range(10)]
        assigned, _ = assign_cycle_statuses(cycles)
        self.assertEqual(sum(row["cycle_status"] == "PASS_STABLE" for row in assigned), 9)
        reference = [row for row in assigned if row["reference_baseline"]]
        self.assertEqual(len(reference), 4)
        self.assertTrue(all(row["stable_run_id"] == "R02" for row in reference))

    def test_atomic_directory_commit_removes_stale_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "DSD_cycles"
            (target / "STxF21" / "B99").mkdir(parents=True)
            (target / "STxF21" / "B99" / "stale.txt").write_text("old")
            staging = make_staging_directory(target)
            (staging / "STxF21" / "B01").mkdir(parents=True)
            (staging / "STxF21" / "B01" / "new.txt").write_text("new")
            commit_directory(staging, target)
            self.assertFalse((target / "STxF21" / "B99").exists())
            self.assertTrue((target / "STxF21" / "B01" / "new.txt").is_file())

    def test_cycle_native_source_uses_actual_baseline_root(self) -> None:
        time_s = np.arange(0.0, 2.0, 0.01)
        cache = {
            "time_100hz": time_s,
            "pressure_100hz": np.ones(time_s.size),
            "envelope_100hz": np.ones(time_s.size),
            "bladder_valid_100hz": np.ones(time_s.size, dtype=bool),
            "eus_valid_100hz": np.ones(time_s.size, dtype=bool),
            "display_fs_hz": 100.0,
            "bladder_raw": np.ones(2000), "bladder_fs_hz": 1000.0,
            "eus_raw": np.ones(4000), "eus_fs_hz": 2000.0,
            "urine": {"source_type": "LEAK_BUTTON_EVENT", "continuous_available": False,
                      "drop_times": np.empty(0), "metadata": {}},
            "first_stim_s": 2.0,
            "baseline_root": Path("X:/custom_baseline"),
        }
        row = {
            "subject": "STxF21", "global_cycle_id": "C02", "dsd_cycle_id": "B01",
            "cycle_start_s": 0.5, "cycle_end_s": 1.5, "cycle_duration_s": 1.0,
            "void_start_s": 1.0, "cmg_peak_s": 1.1, "urine_output_onset_s": 1.1,
            "void_end_s": 1.2, "first_stim_s": 2.0, "cycle_boundary_method": "TEST",
            "urine_evidence_type": "LEAK_BUTTON_EVENT", "reference_baseline": True,
            "reference_baseline_id": "RB01",
        }
        _, native = cycle_arrays(cache, row)
        self.assertIn("custom_baseline", str(native["source_file"]))

    def test_registry_and_dsd_cohort_agree(self) -> None:
        configured = {subject for subject, row in SUBJECT_REGISTRY.items() if row["dsd_confirmed"]}
        self.assertEqual(configured, set(DSD_SUBJECTS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
