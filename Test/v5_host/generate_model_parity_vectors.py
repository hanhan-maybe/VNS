"""Generate same-row model-parity vectors from frozen V5 replay CSV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

FEATURES = (
    "p_current_delta", "p_peak_delta", "p_threshold_above_duration",
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt",
    "p_positive_dpdt_occupancy", "p_auc", "p_auc_growth",
    "pressure_curvature", "peak_to_current_drop",
    "p_trailing_variability_1s", "pressure_power_0p2_0p6_rel",
    "pressure_auc_0p2_20_rel", "pressure_spectral_entropy",
)


def c_float(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def generate(source: Path, subject: str, destination: Path) -> int:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = [r for r in csv.DictReader(handle) if r["animal"] == subject]
    rows = []
    for row in source_rows:
        try:
            values = [float(row[name]) for name in FEATURES]
            score = float(row["python_score"])
            threshold = float(row["threshold"])
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(x) for x in values + [score, threshold]):
            rows.append((row, values, score, threshold))
    if not rows:
        raise RuntimeError(f"No usable rows for {subject}")

    short = subject.lower().replace("stx", "")
    upper = short.upper()
    ctype = f"{upper}ParityRow"
    symbol = f"g_{short}_parity"
    lines = [
        f"#ifndef {upper}_PARITY_VECTORS_H", f"#define {upper}_PARITY_VECTORS_H", "",
        "#include <stddef.h>", "#include <stdint.h>", "", "typedef struct {",
        "    float x[15];", "    float expected_score;", "    float expected_threshold;",
        "    uint8_t expected_positive;", "    const char *cycle_id;",
        "    int decision_index;", f"}} {ctype};", "",
        f"static const {ctype} {symbol}_rows[] = {{",
    ]
    for row, values, score, threshold in rows:
        xs = ", ".join(c_float(x) for x in values)
        positive = 1 if score >= threshold else 0
        lines.append(
            f'    {{ {{ {xs} }}, {c_float(score)}, {c_float(threshold)}, '
            f'{positive}u, "{row["cycle_id"]}", {int(float(row["decision_index"]))} }},'
        )
    lines.extend([
        "};", "", f"static const size_t {symbol}_count =",
        f"    sizeof({symbol}_rows) / sizeof({symbol}_rows[0]);", "",
        f"#endif /* {upper}_PARITY_VECTORS_H */", "",
    ])
    destination.write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", choices=("STxF26", "STxF37"))
    parser.add_argument("--source", type=Path,
                        default=root / "data/NVC_V5/v5_final_validation/F37_F26_streaming_test_vectors.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    short = args.subject.lower().replace("stx", "")
    output = args.output or Path(__file__).with_name(f"{short}_parity_vectors.h")
    count = generate(args.source, args.subject, output)
    print(f"generated {count} {args.subject} rows: {output}")


if __name__ == "__main__":
    main()
