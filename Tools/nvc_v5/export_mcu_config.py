"""Export frozen V5 M1 P-EARLY parameters to a C header.

Run current V5 final validation first. This script does not retrain the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from . import config as C

FEATURES = tuple(C.P_EARLY_FEATURES)


def _first(d, *names):
    for n in names:
        if n in d:
            return d[n]
    raise KeyError(names)


def _fmt(xs):
    return ", ".join(f"{float(x):.9g}f" for x in xs)


def export_one(src: Path, dst: Path):
    payload = json.loads(src.read_text(encoding="utf-8"))
    p = payload.get("model_params", payload.get("params", payload))

    order = tuple(p.get("features", payload.get("features", FEATURES)))
    if order != FEATURES:
        raise RuntimeError(f"feature order mismatch: {order} != {FEATURES}")

    mean = list(_first(p, "center", "mean", "scaler_mean"))
    scale = list(_first(p, "scale", "scaler_scale"))
    coef = _first(p, "coef", "coefficients", "lr_coef")
    if coef and isinstance(coef[0], list):
        coef = coef[0]
    coef = list(coef)
    intercept = _first(p, "intercept", "lr_intercept")
    if isinstance(intercept, list):
        intercept = intercept[0]
    threshold = float(_first(p, "threshold"))

    if len(mean) != 15 or len(scale) != 15 or len(coef) != 15:
        raise RuntimeError("V5 P-EARLY expects 15 features")

    subject = payload.get("subject", payload.get("animal", src.name.split("_m1_")[0]))
    symbol = subject.lower().replace("-", "_")
    guard = f"V5_MODEL_{subject.upper()}_H".replace("-", "_")

    text = f'''#ifndef {guard}\n#define {guard}\n\n#include "v5_model.h"\n\nstatic const V5ModelConfig g_v5_model_{symbol} = {{\n    .mean = {{ {_fmt(mean)} }},\n    .scale = {{ {_fmt(scale)} }},\n    .coef = {{ {_fmt(coef)} }},\n    .intercept = {float(intercept):.9g}f,\n    .threshold = {threshold:.9g}f\n}};\n\n#endif\n'''
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="e.g. STxF26")
    ap.add_argument("--input", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    src = args.input or (C.OUTPUT_ROOT / "v5_final_validation" /
                         f"{args.subject}_m1_frozen_config.json")
    dst = args.output or (C.ROOT / "Modules" / "Inc" /
                          f"v5_model_{args.subject.lower()}.h")
    export_one(src, dst)


if __name__ == "__main__":
    main()
