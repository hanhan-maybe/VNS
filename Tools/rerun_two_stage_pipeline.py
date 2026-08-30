"""Run PRE_STIM extraction followed by DSD cycle extraction; event census is excluded."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from sparc338_common import source_record, write_json_atomic
    from sparc338_config import BASELINE_ROOT, DSD_CYCLES_ROOT, RAW_ROOT, SCI_SUBJECTS
except ImportError:
    from Tools.sparc338_common import source_record, write_json_atomic
    from Tools.sparc338_config import BASELINE_ROOT, DSD_CYCLES_ROOT, RAW_ROOT, SCI_SUBJECTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_command(arguments: list[str]) -> None:
    print("RUN:", " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=PROJECT_ROOT, env=os.environ.copy(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DSD_CYCLES_ROOT)
    parser.add_argument("--hash-source", action="store_true")
    args = parser.parse_args()

    started = datetime.now(timezone.utc).isoformat()
    raw_before = {
        subject: source_record(args.raw_root / f"{subject}.smrx")
        for subject in SCI_SUBJECTS
        if (args.raw_root / f"{subject}.smrx").is_file()
    }
    if not args.skip_baseline:
        command = [
            sys.executable, str(PROJECT_ROOT / "Tools" / "sparc338_pre_stim_extract.py"),
            "--raw-dir", str(args.raw_root), "--output-dir", str(args.baseline_root),
        ]
        if args.hash_source:
            command.append("--hash-source")
        run_command(command)
    run_command([
        sys.executable, "-m", "Tools.dsd_cycle_extraction.pipeline",
        "--baseline-root", str(args.baseline_root), "--output-root", str(args.output_root),
    ])
    raw_after = {
        subject: source_record(args.raw_root / f"{subject}.smrx")
        for subject in SCI_SUBJECTS
        if (args.raw_root / f"{subject}.smrx").is_file()
    }
    changed = sorted(subject for subject in raw_before if raw_before[subject] != raw_after.get(subject))
    audit = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "stages": ["baseline", "DSD_cycles"] if not args.skip_baseline else ["DSD_cycles"],
        "event_census_run": False,
        "raw_stat_all_identical": not changed,
        "raw_changed_subjects": changed,
        "raw_before": raw_before,
        "raw_after": raw_after,
    }
    write_json_atomic(args.output_root / "two_stage_rerun_integrity.json", audit)
    if changed:
        raise RuntimeError(f"Raw SMRX source metadata changed during run: {changed}")


if __name__ == "__main__":
    main()
