import inspect
import unittest

import sparc338_stable_phase


class StableBaselineIndependenceTests(unittest.TestCase):
    def test_phase_a_source_has_no_phenotype_inputs(self):
        source = inspect.getsource(sparc338_stable_phase).lower()
        for forbidden in ("tonic", "phasic", "silent_fraction", "occupancy", "phenotype"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
