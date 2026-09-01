"""Combine raw causal pressure with frozen Python registered replay outputs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path,
                        default=Path("Test/v5_results/generated/candidate_golden.csv"))
    parser.add_argument("--replay", type=Path,
                        default=Path("data/NVC_V5/v5_final_validation/m1_full_cycle_replay.csv"))
    parser.add_argument("--output", type=Path,
                        default=Path("Test/v5_results/generated/full_replay_golden.csv"))
    args = parser.parse_args()
    replay = pd.read_csv(args.replay)
    replay = replay[replay.decision_index.astype(int) % 25 == 0]
    expected = {(str(r.animal), str(r.cycle_id), int(r.decision_index)): r
                for r in replay.itertuples(index=False)}
    output_fields = [
        "animal", "cycle_id", "is_test", "sample_index", "pressure", "signal_valid",
        "prior_sigma_p", "prior_sigma_dpdt", "expected_candidate_active",
        "expected_candidate_event_id", "expected_registered", "expected_feature_available",
        "expected_score", "expected_score_positive", "expected_t0_state", "expected_t0_trigger",
    ]
    with args.candidate.open("r", encoding="utf-8", newline="") as source, \
         args.output.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=output_fields)
        writer.writeheader()
        count = 0
        for row in reader:
            key = (row["animal"], row["cycle_id"], int(row["sample_index"]))
            ref = expected.get(key)
            registered = ref is not None
            event_text = str(ref.candidate_event_id) if registered else ""
            event_numeric = int(event_text.rsplit("_", 1)[-1]) if event_text not in ("", "nan", "None") else 0
            writer.writerow({
                "animal": row["animal"], "cycle_id": row["cycle_id"], "is_test": row["is_test"],
                "sample_index": row["sample_index"], "pressure": row["pressure"],
                "signal_valid": row["signal_valid"], "prior_sigma_p": row["prior_sigma_p"],
                "prior_sigma_dpdt": row["prior_sigma_dpdt"],
                "expected_candidate_active": row["candidate_active"],
                "expected_candidate_event_id": row["candidate_event_id"],
                "expected_registered": str(registered),
                "expected_feature_available": str(bool(ref.feature_available)) if registered else "False",
                "expected_score": repr(float(ref.score)) if registered and pd.notna(ref.score) else "",
                "expected_score_positive": str(bool(ref.score_positive)) if registered else "False",
                "expected_t0_state": str(bool(ref.t0_state)) if registered else "False",
                "expected_t0_trigger": str(bool(ref.t0_trigger)) if registered else "False",
            })
            count += 1
    print(f"generated {count} full replay samples: {args.output}")


if __name__ == "__main__":
    main()
