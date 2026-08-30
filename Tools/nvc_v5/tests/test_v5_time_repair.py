import numpy as np
import pandas as pd

from ..data_adapter import event_confirm_time


def test_missing_confirm_time_is_reconstructed():
    cycle = {"t_abs_s": np.arange(100, dtype=float) + 10.0}
    row = pd.Series({"confirm_time_s": np.nan, "confirm_index": 25})
    assert event_confirm_time(cycle, row) == 35.0

