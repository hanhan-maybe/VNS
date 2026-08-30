import unittest
import numpy as np
from sparc338_preprocessing import causal_downsample


class ResamplingLengthTests(unittest.TestCase):
    def test_duration_error(self):
        for fs in (200.0, 2000.0, 10000.0, 1.0 / (21 * 5e-6)):
            duration = 12.345
            x = np.zeros(int(np.ceil(duration * fs)), dtype=np.float32)
            y, _ = causal_downsample(x, fs, 100.0, 40.0)
            self.assertLessEqual(abs(len(y) / 100.0 - len(x) / fs), 0.01 + 1e-9)


if __name__ == "__main__": unittest.main()
