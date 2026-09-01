"""Export full test-cycle pressure with frozen registered P-EARLY rows."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tools.nvc_v5 import config as C
from Tools.nvc_v5.data_adapter import build_v5_dataset
from Tools.nvc_v5.source_adapter import _load_pair


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("Test/v5_results/generated/feature_golden.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    replay = pd.read_csv(C.OUTPUT_ROOT / "v5_final_validation/m1_full_cycle_replay.csv")
    replay = replay[replay.decision_index.astype(int) % 25 == 0].copy()
    _, _, _, _, paths, _ = build_v5_dataset()
    fields = ["animal", "cycle_id", "sample_index", "pressure", "signal_valid",
              "expected_available", *C.P_EARLY_FEATURES]
    total = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject in C.SUBJECTS:
            for cycle_id in C.SPLITS[subject]["test"]:
                cycle = _load_pair(paths[(subject, cycle_id)])["cycle"]
                pressure = np.asarray(cycle["bladder_pressure_mmHg"], np.float32)
                valid = np.asarray(cycle.get("cmg_valid_100hz", np.ones(len(pressure), bool)), bool)
                expected_rows = replay[(replay.animal == subject) &
                                       (replay.cycle_id.astype(str) == cycle_id)]
                expected = {int(r.decision_index): r for r in expected_rows.itertuples(index=False)}
                for i, value in enumerate(pressure):
                    source = expected.get(i)
                    available = bool(source is not None and bool(source.feature_available))
                    row = {"animal": subject, "cycle_id": cycle_id, "sample_index": i,
                           "pressure": repr(float(value)), "signal_valid": str(bool(valid[i])),
                           "expected_available": str(available)}
                    for name in C.P_EARLY_FEATURES:
                        row[name] = repr(float(getattr(source, name))) if available else ""
                    writer.writerow(row)
                    total += 1
    print(f"generated {total} samples: {args.output}")


if __name__ == "__main__":
    main()
