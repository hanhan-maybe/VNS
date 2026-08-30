import unittest
from sparc338_smrx_reader import match_channels


class ChannelMatchingTests(unittest.TestCase):
    def test_raw_eus_preferred_to_ceus(self):
        rows = [
            {"channel": 3, "type": "RealWave", "title": "cEUS", "units": "mV", "selected_role": "OTHER"},
            {"channel": 1, "type": "Adc", "title": "EUS", "units": "mV", "selected_role": "OTHER"},
            {"channel": 0, "type": "Adc", "title": "CMG pres", "units": "mmHg", "selected_role": "OTHER"},
            {"channel": 6, "type": "EventRise", "title": "Stim", "units": "", "selected_role": "OTHER"},
        ]
        selected, _ = match_channels(rows)
        self.assertEqual(selected["EUS_RAW"]["channel"], 1)
        self.assertEqual(selected["EUS_FILTERED"]["channel"], 3)
        self.assertEqual(selected["BLADDER"]["channel"], 0)


if __name__ == "__main__": unittest.main()
