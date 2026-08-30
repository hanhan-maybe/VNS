"""Regenerate and audit all V3-V5 reporting figures without retraining."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from Tools.nvc_v3.visualization import generate_plots as plot_v3
from Tools.nvc_v3_1.visualization import generate_plots as plot_v31
from Tools.nvc_v3_2.visualization import generate_plots as plot_v32
from Tools.nvc_v4.visualization import generate_plots as plot_v4
from Tools.nvc_v5.visualization import generate_plots as plot_v5


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REPORTING_ROOT = DATA_ROOT / "NVC_REPORTING"


def main(strict: bool = True) -> None:
    groups = (
        ("V3", plot_v3, DATA_ROOT / "NVC_V3", 4),
        ("V3.1", plot_v31, DATA_ROOT / "NVC_V3_1", 4),
        ("V3.2", plot_v32, DATA_ROOT / "NVC_V3_2", 5),
        ("V4", plot_v4, DATA_ROOT / "NVC_V4", 5),
        ("V5", plot_v5, DATA_ROOT / "NVC_V5", 7),
    )
    total = 0
    manifest = []
    failures = []
    for version, function, output_root, minimum in groups:
        paths = function(output_root)
        total += len(paths)
        status = "PASS" if len(paths) >= minimum else "INCOMPLETE"
        if status != "PASS":
            failures.append(f"{version}: {len(paths)}/{minimum}")
        print(f"{version}: generated {len(paths)} figures [{status}, minimum={minimum}]")
        for path in paths:
            print(f"  {path}")
            manifest.append({
                "version": version,
                "figure": path.name,
                "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "status": "PASS" if path.stat().st_size > 10_000 else "CHECK_SMALL_FILE",
            })

    REPORTING_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTING_ROOT / "FIGURE_MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("version", "figure", "relative_path", "bytes", "status"))
        writer.writeheader(); writer.writerows(manifest)
    status_path = REPORTING_ROOT / "FIGURE_STATUS.md"
    rows = [
        "# V3–V5 图片生成状态",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}",
        "",
        "| 版本 | 图片数 | 最低要求 | 状态 |",
        "|---|---:|---:|---|",
    ]
    counts = {version: sum(item["version"] == version for item in manifest) for version, *_ in groups}
    for version, _, _, minimum in groups:
        count = counts[version]
        rows.append(f"| {version} | {count} | {minimum} | {'PASS' if count >= minimum else 'INCOMPLETE'} |")
    rows.extend(("", f"总计：{total} 张图片。", "", "逐图路径与文件大小见 `FIGURE_MANIFEST.csv`。", ""))
    status_path.write_text("\n".join(rows), encoding="utf-8")
    print(f"Total: {total} figures")
    print(f"Manifest: {manifest_path}")
    if strict and failures:
        raise RuntimeError("Figure generation incomplete: " + "; ".join(failures))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-strict", action="store_true", help="Do not fail when a version produces too few figures")
    args = parser.parse_args()
    main(strict=not args.no_strict)
