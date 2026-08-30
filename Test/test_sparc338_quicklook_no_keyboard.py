import inspect
import unittest
from sparc338_pre_stim_qc import make_quicklooks


class QuicklookContentTests(unittest.TestCase):
    def test_third_panel_has_no_keyboard_path(self):
        source = inspect.getsource(make_quicklooks).casefold()
        self.assertNotIn("key" + "board", source)
        self.assertIn("urine output / voiding evidence", source)


if __name__ == "__main__": unittest.main()
