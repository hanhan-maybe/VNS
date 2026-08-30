import unittest

from sparc338_stable_phase import select_nearest_stable_run


class IncompleteEndTests(unittest.TestCase):
    def test_incomplete_terminal_cycle_cannot_be_selected(self):
        cycles = [
            {"original_cycle_number": i, "stability_candidate": "STABLE_CANDIDATE"} for i in range(1, 6)
        ] + [{"original_cycle_number": 6, "stability_candidate": "TRANSITIONAL"}]
        _, _, selected = select_nearest_stable_run(cycles)
        self.assertEqual([x["original_cycle_number"] for x in selected], [1, 2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main()
