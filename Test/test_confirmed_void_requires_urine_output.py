import unittest
from sparc338_stable_phase import confirm_with_urine


class ConfirmedVoidTests(unittest.TestCase):
    def test_cmg_peak_alone_is_not_confirmed(self):
        contractions=[{"void_start_s":10.0,"cmg_peak_s":15.0,"void_end_s":20.0}]
        confirmed,excluded=confirm_with_urine(contractions,"NONE")
        self.assertEqual(confirmed,[])
        self.assertEqual(len(excluded),1)


if __name__ == "__main__": unittest.main()
