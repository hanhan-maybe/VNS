import unittest
import numpy as np
from sparc338_preprocessing import align_100hz, build_phase_segments, group_stim_trains


class BoundaryTests(unittest.TestCase):
    def test_strict_boundary_and_post_stim_off(self):
        t, b, e = align_100hz(np.zeros(10001), np.zeros(10001), 100.0)
        self.assertLess(t[-1], 100.0)
        self.assertEqual(len(t), 10000)
        trains = group_stim_trains([100.0, 100.1, 200.0, 200.1])
        phases = build_phase_segments(300.0, trains)
        pre = [p for p in phases if p["phase_type"] == "PRE_STIM"]
        self.assertEqual(len(pre), 1)
        self.assertEqual(pre[0]["end_s"], 100.0)
        self.assertTrue(all(p["phase_type"] != "PRE_STIM" for p in phases[1:]))


if __name__ == "__main__": unittest.main()
