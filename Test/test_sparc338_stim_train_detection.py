import unittest
from sparc338_preprocessing import group_stim_trains


class StimTrainTests(unittest.TestCase):
    def test_two_trains(self):
        times = [100.0, 100.1, 100.2, 101.0, 200.0, 200.1]
        trains = group_stim_trains(times)
        self.assertEqual(len(trains), 2)
        self.assertEqual(trains[0]["start_s"], 100.0)
        self.assertEqual(trains[1]["start_s"], 200.0)


if __name__ == "__main__": unittest.main()
