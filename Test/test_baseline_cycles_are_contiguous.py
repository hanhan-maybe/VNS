import unittest

from sparc338_stable_phase import select_nearest_stable_run


def row(number, status="STABLE_CANDIDATE"):
    return {"original_cycle_number": number, "stability_candidate": status}


class ContiguousBaselineTests(unittest.TestCase):
    def test_selected_cycles_are_from_one_contiguous_run(self):
        cycles = [row(1), row(2), row(3, "TRANSITIONAL"), row(4), row(5), row(6)]
        _, run, selected = select_nearest_stable_run(cycles)
        self.assertEqual([x["original_cycle_number"] for x in run], [4, 5, 6])
        self.assertEqual([x["original_cycle_number"] for x in selected], [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
