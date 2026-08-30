"""Acceptance tests for the independent stable-cycle extraction output."""
from __future__ import annotations

import csv
import json
import re
import unittest
from pathlib import Path

import numpy as np

from .config import OUTPUT_ROOT, SUBJECTS


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


class CycleExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = read_csv(OUTPUT_ROOT / "cycle_manifest.csv")
        cls.candidates = read_csv(OUTPUT_ROOT / "all_candidate_cycles.csv")
        cls.summaries = read_csv(OUTPUT_ROOT / "subject_summary.csv")

    def test_01_only_requested_subjects(self) -> None:
        found = {row["subject"] for row in self.candidates}
        self.assertEqual(found, set(SUBJECTS))
        self.assertEqual({row["subject"] for row in self.manifest}, set(SUBJECTS))

    def test_02_all_samples_are_strictly_pre_stim(self) -> None:
        for row in self.manifest:
            path = OUTPUT_ROOT / row["subject"] / row["dsd_cycle_id"] / "cycle_100Hz.npz"
            with np.load(path, allow_pickle=False) as data:
                self.assertTrue(np.all(data["t_abs_s"] < float(data["first_stim_s"])))
            native_path = path.with_name("cycle_native_eus.npz")
            with np.load(native_path, allow_pickle=False) as data:
                self.assertTrue(np.all(data["t_eus_abs_s"] < float(data["first_stim_s"])))

    def test_03_cycle_boundaries_are_ordered(self) -> None:
        for row in self.candidates:
            if row["cycle_start_s"].lower() != "nan":
                self.assertLess(float(row["cycle_start_s"]), float(row["cycle_end_s"]))

    def test_04_pass_cycles_are_complete(self) -> None:
        for row in self.manifest:
            self.assertEqual(row["cycle_status"], "PASS_STABLE")
            self.assertTrue(as_bool(row["complete_cycle"]))

    def test_05_pass_cycles_have_confirmed_current_void(self) -> None:
        self.assertTrue(self.manifest)
        self.assertTrue(all(as_bool(row["confirmed_void"]) for row in self.manifest))

    def test_06_subject_cycles_strictly_increase(self) -> None:
        for subject in SUBJECTS:
            rows = [row for row in self.candidates if row["subject"] == subject]
            ends = [float(row["cycle_end_s"]) for row in rows]
            self.assertTrue(all(right > left for left, right in zip(ends, ends[1:])))

    def test_07_dataset_ids_are_contiguous(self) -> None:
        for subject in SUBJECTS:
            rows = [row for row in self.manifest if row["subject"] == subject]
            self.assertEqual([row["dsd_cycle_id"] for row in rows],
                             [f"B{index:02d}" for index in range(1, len(rows) + 1)])

    def test_08_global_ids_preserve_original_order(self) -> None:
        for subject in SUBJECTS:
            candidates = [row for row in self.candidates if row["subject"] == subject]
            expected = [f"C{index:02d}" for index in range(1, len(candidates) + 1)]
            self.assertEqual([row["global_cycle_id"] for row in candidates], expected)
            included = [int(row["global_cycle_id"][1:]) for row in self.manifest if row["subject"] == subject]
            self.assertEqual(included, sorted(included))

    def test_09_no_fixed_duration_cycle_logic(self) -> None:
        durations = np.asarray([float(row["cycle_duration_s"]) for row in self.manifest])
        self.assertGreater(float(np.ptp(durations)), 1.0)
        source_dir = Path(__file__).resolve().parent
        production = "\n".join(
            path.read_text(encoding="utf-8") for path in source_dir.glob("*.py")
            if path.name != Path(__file__).name
        )
        self.assertIsNone(re.search(r"cycle_duration\s*=\s*200(?:\.0)?", production))
        self.assertNotIn("peak_plus_minus_fixed_window", production)

    def test_10_stability_does_not_use_prohibited_fields(self) -> None:
        source_dir = Path(__file__).resolve().parent
        production = "\n".join(
            path.read_text(encoding="utf-8") for path in source_dir.glob("*.py")
            if path.name != Path(__file__).name
        ).upper()
        prohibited = ("PHENOTYPE_V1", "PHENOTYPE_V2", "TONIC_HIGH", "TONIC_LOW",
                      "PHASIC_CANDIDATE", "PHASIC_CONFIRMED", "DSD_SCORE", "PARTIAL_SCORE")
        for token in prohibited:
            self.assertNotIn(token, production)

    def test_11_protected_sources_were_not_overwritten(self) -> None:
        integrity = json.loads((OUTPUT_ROOT / "source_integrity.json").read_text(encoding="utf-8"))
        self.assertTrue(integrity["sha256_all_identical"])
        self.assertEqual(integrity["changed_files"], [])

    def test_12_quicklooks_match_manifest_boundaries(self) -> None:
        for row in self.manifest:
            cycle_dir = OUTPUT_ROOT / row["subject"] / row["dsd_cycle_id"]
            self.assertTrue((cycle_dir / "quicklook.png").is_file())
            with np.load(cycle_dir / "cycle_100Hz.npz", allow_pickle=False) as data:
                self.assertAlmostEqual(float(data["cycle_start_s"]), float(row["cycle_start_s"]), places=8)
                self.assertAlmostEqual(float(data["cycle_end_s"]), float(row["cycle_end_s"]), places=8)
                self.assertEqual(str(data["global_cycle_id"]), row["global_cycle_id"])
                self.assertEqual(str(data["dsd_cycle_id"]), row["dsd_cycle_id"])

    def test_13_reference_baseline_is_manifest_subset(self) -> None:
        reference = read_csv(OUTPUT_ROOT / "reference_baseline_manifest.csv")
        manifest_keys = {(row["subject"], row["dsd_cycle_id"]) for row in self.manifest}
        reference_keys = {(row["subject"], row["dsd_cycle_id"]) for row in reference}
        self.assertTrue(reference_keys.issubset(manifest_keys))
        for subject in SUBJECTS:
            count = sum(row["subject"] == subject for row in reference)
            self.assertIn(count, {0, 3, 4, 5})


if __name__ == "__main__":
    unittest.main()
