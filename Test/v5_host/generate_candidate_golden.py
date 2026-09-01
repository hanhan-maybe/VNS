"""Export pressure-only candidate traces from the frozen Python detector."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tools.nvc_v5 import config as C
from Tools.nvc_v5.data_adapter import build_v5_dataset
from Tools.nvc_v5.final_validation import _adaptive_priors, _candidate_trace_for_subject


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("Test/v5_results/generated/candidate_golden.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, _, _, _, paths, _ = build_v5_dataset()
    fields = (
        "animal", "cycle_id", "is_test", "sample_index", "pressure", "signal_valid",
        "expected_data_valid", "candidate_active", "candidate_event_id",
        "recovery_active", "candidate_ended", "residual", "adaptive_start",
        "adaptive_confirm", "adaptive_recovery", "prior_sigma_p", "prior_sigma_dpdt",
    )
    count = 0
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for subject in C.SUBJECTS:
            split = C.SPLITS[subject]
            cycle_ids = list(split["train"]) + list(split["test"])
            traces = _candidate_trace_for_subject(subject, cycle_ids, paths)
            prior_p, prior_d = _adaptive_priors(subject)
            for cycle_id in cycle_ids:
                trace = traces[str(cycle_id)]
                cycle = trace["cycle"]
                pressure = np.asarray(cycle["bladder_pressure_mmHg"], float)
                supplied_valid = np.asarray(cycle.get("cmg_valid_100hz", np.ones(len(pressure), bool)), bool)
                ids = trace["candidate_event_id"]
                active = np.asarray(trace["candidate_active"], bool)
                for i, value in enumerate(pressure):
                    uid = str(ids[i]) if active[i] else ""
                    numeric_id = int(uid.rsplit("_", 1)[-1]) if uid else 0
                    ended = bool(i > 0 and active[i - 1] and not active[i])
                    writer.writerow({
                        "animal": subject, "cycle_id": cycle_id,
                        "is_test": str(cycle_id in split["test"]), "sample_index": i,
                        "pressure": repr(float(value)), "signal_valid": str(bool(supplied_valid[i])),
                        "expected_data_valid": str(bool(trace["adaptive"]["valid"][i])),
                        "candidate_active": str(bool(active[i])), "candidate_event_id": numeric_id,
                        "recovery_active": str(bool(trace["recovery_active"][i])),
                        "candidate_ended": str(ended),
                        "residual": repr(float(trace["residual"][i])),
                        "adaptive_start": repr(float(trace["adaptive"]["adaptive_start"][i])),
                        "adaptive_confirm": repr(float(trace["adaptive"]["adaptive_confirm"][i])),
                        "adaptive_recovery": repr(float(trace["adaptive"]["adaptive_recovery"][i])),
                        "prior_sigma_p": repr(prior_p), "prior_sigma_dpdt": repr(prior_d),
                    })
                    count += 1
    print(f"generated {count} causal pressure samples: {args.output}")


if __name__ == "__main__":
    main()
