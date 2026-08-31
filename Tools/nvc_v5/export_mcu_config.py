"""Export frozen V5 M1 P-EARLY parameters to a C header.

Run current V5 final validation first:

    python -m Tools.nvc_v5.run_final_validation

Then:

    python -m Tools.nvc_v5.export_mcu_config STxF26
    python -m Tools.nvc_v5.export_mcu_config STxF37

This script does NOT retrain the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config as C


FEATURES = tuple(C.P_EARLY_FEATURES)


def _first(d, *names):
    for name in names:
        if name in d:
            return d[name]

    raise KeyError(
        f"None of the expected keys {names} were found. "
        f"Available keys: {sorted(d.keys())}"
    )


def _fmt(xs):
    return ", ".join(f"{float(x):.9g}f" for x in xs)


def _flatten_vector(x):
    """Accept sklearn-style [[...]] or simple [...]."""
    if isinstance(x, (tuple, list)) and len(x) == 1:
        if isinstance(x[0], (tuple, list)):
            return list(x[0])

    return list(x)


def _flatten_scalar(x):
    """Accept sklearn-style [x] or scalar x."""
    if isinstance(x, (tuple, list)):
        if len(x) != 1:
            raise RuntimeError(
                f"Expected scalar or one-element list, got {x}"
            )
        return float(x[0])

    return float(x)


def export_one(src: Path, dst: Path):
    if not src.exists():
        raise FileNotFoundError(
            f"Frozen V5 configuration not found:\n{src}\n\n"
            "Run first:\n"
            "python -m Tools.nvc_v5.run_final_validation"
        )

    payload = json.loads(src.read_text(encoding="utf-8"))

    # ---------------------------------------------------------
    # Support both:
    # 1. current final_validation frozen config schema
    # 2. older/nested model_params schema
    # ---------------------------------------------------------
    p = payload.get(
        "model_params",
        payload.get("params", payload)
    )

    feature_order = tuple(
        _first(
            p,
            "feature_order",
            "features",
        )
    )

    if feature_order != FEATURES:
        raise RuntimeError(
            "V5 P-EARLY feature-order mismatch.\n"
            f"Expected:\n{FEATURES}\n\n"
            f"Frozen config:\n{feature_order}"
        )

    mean = list(
        _first(
            p,
            "scaler_center",
            "center",
            "mean",
            "scaler_mean",
        )
    )

    scale = list(
        _first(
            p,
            "scaler_scale",
            "scale",
        )
    )

    coef = _flatten_vector(
        _first(
            p,
            "lr_coefficients",
            "coef",
            "coefficients",
            "lr_coef",
        )
    )

    intercept = _flatten_scalar(
        _first(
            p,
            "lr_intercept",
            "intercept",
        )
    )

    threshold = float(
        _first(
            p,
            "probability_threshold",
            "threshold",
        )
    )

    n = len(FEATURES)

    if len(mean) != n:
        raise RuntimeError(
            f"Scaler center length mismatch: "
            f"expected {n}, got {len(mean)}"
        )

    if len(scale) != n:
        raise RuntimeError(
            f"Scaler scale length mismatch: "
            f"expected {n}, got {len(scale)}"
        )

    if len(coef) != n:
        raise RuntimeError(
            f"LR coefficient length mismatch: "
            f"expected {n}, got {len(coef)}"
        )

    if not (0.0 < threshold < 1.0):
        raise RuntimeError(
            f"Invalid probability threshold: {threshold}"
        )

    subject = payload.get(
        "subject",
        payload.get(
            "animal",
            src.name.split("_m1_")[0],
        ),
    )

    symbol = subject.lower().replace("-", "_")
    guard = (
        f"V5_MODEL_{subject.upper()}_H"
        .replace("-", "_")
    )

    text = f"""\
#ifndef {guard}
#define {guard}

#include "v5_model.h"

/*
 * AUTO-GENERATED FILE.
 *
 * Source:
 *   {src.as_posix()}
 *
 * Model:
 *   V5 individualized M1 P-EARLY
 *
 * DO NOT manually tune these values on MCU.
 */

static const V5ModelConfig g_v5_model_{symbol} = {{
    .mean = {{
        {_fmt(mean)}
    }},
    .scale = {{
        {_fmt(scale)}
    }},
    .coef = {{
        {_fmt(coef)}
    }},
    .intercept = {intercept:.9g}f,
    .threshold = {threshold:.9g}f
}};

#endif /* {guard} */
"""

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dst.write_text(
        text,
        encoding="utf-8",
    )

    print("V5 MCU model export PASS")
    print(f"subject    : {subject}")
    print(f"features   : {n}")
    print(f"threshold  : {threshold:.9g}")
    print(f"input      : {src}")
    print(f"output     : {dst}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "subject",
        help="e.g. STxF26 or STxF37",
    )

    ap.add_argument(
        "--input",
        type=Path,
        default=None,
    )

    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = ap.parse_args()

    src = args.input or (
        C.OUTPUT_ROOT
        / "v5_final_validation"
        / f"{args.subject}_m1_frozen_config.json"
    )

    dst = args.output or (
        C.ROOT
        / "Modules"
        / "Inc"
        / f"v5_model_{args.subject.lower()}.h"
    )

    export_one(src, dst)


if __name__ == "__main__":
    main()
    