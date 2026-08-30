import unittest

from sparc338_stable_phase import select_nearest_stable_run


class NoBackfillTests(unittest.TestCase):
    def test_four_cycle_latest_run_stays_four(self):
        cycles = [
            *[{"original_cycle_number": i, "stability_candidate": "STABLE_CANDIDATE"} for i in range(1, 7)],
            {"original_cycle_number": 7, "stability_candidate": "TRANSITIONAL"},
            *[{"original_cycle_number": i, "stability_candidate": "STABLE_CANDIDATE"} for i in range(8, 12)],
        ]
        _, _, selected = select_nearest_stable_run(cycles)
        self.assertEqual([x["original_cycle_number"] for x in selected], [8, 9, 10, 11])


if __name__ == "__main__":
    unittest.main()
