import unittest
from pathlib import Path


class NoFutureFilteringTests(unittest.TestCase):
    def test_production_sources_use_only_forward_filters(self):
        root = Path(__file__).parent
        banned = ["filt" + "filt", "sosfilt" + "filt", "centered" + " moving average"]
        production = [root / "sparc338_smrx_reader.py", root / "sparc338_preprocessing.py",
                      root / "sparc338_pre_stim_qc.py", root / "sparc338_pre_stim_extract.py",
                      root / "sparc338_urine_output.py"]
        for path in production:
            text = path.read_text(encoding="utf-8").casefold()
            for token in banned:
                self.assertNotIn(token, text, f"{token} found in {path.name}")


if __name__ == "__main__": unittest.main()
