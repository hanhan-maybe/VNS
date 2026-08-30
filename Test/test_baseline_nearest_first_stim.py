import unittest

from sparc338_stable_phase import select_nearest_stable_run


class NearestRunTests(unittest.TestCase):
    def test_latest_qualifying_run_is_selected(self):
        cycles = [
            {"original_cycle_number": i, "stability_candidate": status}
            for i, status in enumerate(("STABLE_CANDIDATE", "STABLE_CANDIDATE", "TRANSITIONAL",
                                        "STABLE_CANDIDATE", "STABLE_CANDIDATE", "STABLE_CANDIDATE"), 1)
        ]
        _, run, _ = select_nearest_stable_run(cycles)
        self.assertEqual([x["original_cycle_number"] for x in run], [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
