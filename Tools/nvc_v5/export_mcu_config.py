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
import hashlib
import json
from pathlib import Path
import struct
import zlib

from . import config as C


FEATURES = tuple(C.P_EARLY_FEATURES)
SUBJECT_CONFIG_MAGIC = 0x43533556
SUBJECT_CONFIG_VERSION = 1
FEATURE_ORDER_HASH = zlib.crc32("|".join(FEATURES).encode("utf-8")) & 0xFFFFFFFF


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


def _generic_values(src: Path):
    payload = json.loads(src.read_text(encoding="utf-8"))
    p = payload.get("model_params", payload.get("params", payload))
    feature_order = tuple(_first(p, "feature_order", "features"))
    if feature_order != FEATURES:
        raise RuntimeError("V5 SubjectConfig feature-order mismatch")
    mean = list(_first(p, "scaler_center", "center", "mean", "scaler_mean"))
    scale = list(_first(p, "scaler_scale", "scale"))
    coef = _flatten_vector(_first(p, "lr_coefficients", "coef", "coefficients", "lr_coef"))
    intercept = _flatten_scalar(_first(p, "lr_intercept", "intercept"))
    threshold = float(_first(p, "probability_threshold", "threshold"))
    if not all(len(x) == len(FEATURES) for x in (mean, scale, coef)):
        raise RuntimeError("V5 SubjectConfig model vector length mismatch")
    subject = str(payload.get("subject", payload.get("animal", src.name.split("_m1_")[0])))
    model_hash = str(payload.get("model_hash", ""))
    if len(model_hash) != 64:
        canonical = json.dumps({"features": FEATURES, "mean": mean, "scale": scale,
                                "coef": coef, "intercept": intercept,
                                "threshold": threshold}, sort_keys=True).encode("utf-8")
        model_hash = hashlib.sha256(canonical).hexdigest()
    from .final_validation import _adaptive_priors
    prior_p, prior_dpdt = _adaptive_priors(subject)
    return {"subject": subject, "mean": mean, "scale": scale, "coef": coef,
            "intercept": intercept, "threshold": threshold, "model_hash": model_hash,
            "prior_sigma_p": prior_p, "prior_sigma_dpdt": prior_dpdt}


def export_generic(src: Path, output_dir: Path, header_dir: Path):
    values = _generic_values(src)
    subject = values["subject"]
    symbol = subject.lower().replace("-", "_")
    model_hash_bytes = bytes.fromhex(values["model_hash"])
    floats = [*values["mean"], *values["scale"], *values["coef"],
              values["intercept"], values["threshold"], values["prior_sigma_p"],
              values["prior_sigma_dpdt"]]
    prefix = struct.pack("<IHHI32s49f", SUBJECT_CONFIG_MAGIC, SUBJECT_CONFIG_VERSION,
                         len(FEATURES), FEATURE_ORDER_HASH, model_hash_bytes, *floats)
    crc = zlib.crc32(prefix) & 0xFFFFFFFF
    binary = prefix + struct.pack("<I", crc)
    if len(binary) != 244:
        raise AssertionError(f"unexpected SubjectConfig size: {len(binary)}")
    audit = {
        "schema": "V5SubjectConfig", "version": SUBJECT_CONFIG_VERSION,
        "subject": subject, "magic": f"0x{SUBJECT_CONFIG_MAGIC:08X}",
        "feature_count": len(FEATURES), "feature_order": list(FEATURES),
        "feature_order_hash_crc32": f"0x{FEATURE_ORDER_HASH:08X}",
        "model_hash_sha256": values["model_hash"], "scaler_mean": values["mean"],
        "scaler_scale": values["scale"], "lr_coef": values["coef"],
        "lr_intercept": values["intercept"], "probability_threshold": values["threshold"],
        "candidate_prior_sigma_p": values["prior_sigma_p"],
        "candidate_prior_sigma_dpdt": values["prior_sigma_dpdt"],
        "binary_size": len(binary), "crc32": f"0x{crc:08X}",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{subject}_v5_subject_config.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / f"{subject}_v5_subject_config.bin").write_bytes(binary)
    hash_values = ", ".join(f"0x{x:02X}u" for x in model_hash_bytes)
    header = f'''#ifndef V5_SUBJECT_CONFIG_{subject.upper()}_H
#define V5_SUBJECT_CONFIG_{subject.upper()}_H
#include "v5_subject_config.h"
static const V5SubjectConfig g_v5_subject_config_{symbol} = {{
    .magic = V5_SUBJECT_CONFIG_MAGIC,
    .version = V5_SUBJECT_CONFIG_VERSION,
    .feature_count = V5_MODEL_FEATURE_COUNT,
    .feature_order_hash = V5_SUBJECT_CONFIG_FEATURE_HASH,
    .model_hash = {{ {hash_values} }},
    .model = {{
        .mean = {{ {_fmt(values["mean"])} }},
        .scale = {{ {_fmt(values["scale"])} }},
        .coef = {{ {_fmt(values["coef"])} }},
        .intercept = {values["intercept"]:.9g}f,
        .threshold = {values["threshold"]:.9g}f
    }},
    .candidate_prior_sigma_p = {values["prior_sigma_p"]:.9g}f,
    .candidate_prior_sigma_dpdt = {values["prior_sigma_dpdt"]:.9g}f,
    .crc32 = 0x{crc:08X}u
}};
#endif
'''
    header_dir.mkdir(parents=True, exist_ok=True)
    (header_dir / f"v5_subject_config_{symbol}.h").write_text(header, encoding="utf-8")
    print(f"generic SubjectConfig PASS: {subject}, crc=0x{crc:08X}")


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
    ap.add_argument("--generic-dir", type=Path, default=C.OUTPUT_ROOT / "mcu_config")
    ap.add_argument("--model-only", action="store_true")

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
    if not args.model_only:
        export_generic(src, args.generic_dir, C.ROOT / "Modules" / "Inc")


if __name__ == "__main__":
    main()
