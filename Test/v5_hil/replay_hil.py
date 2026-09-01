"""Replay frozen 100 Hz pressure to STM32 and audit returned V5 telemetry."""
from __future__ import annotations

import argparse
import csv
import struct
import time
from pathlib import Path
import zlib

PRESSURE_MAGIC = 0x49503556
TELEMETRY_MAGIC = 0x4F543556
INPUT_SIZE = 20
OUTPUT_SIZE = 112
FEATURES = (
    "p_current_delta", "p_peak_delta", "p_threshold_above_duration",
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt",
    "p_positive_dpdt_occupancy", "p_auc", "p_auc_growth",
    "pressure_curvature", "peak_to_current_drop", "p_trailing_variability_1s",
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
    "pressure_spectral_entropy",
)


def flag(value: str) -> bool:
    return value.casefold() in {"true", "1"}


def pressure_frame(index: int, pressure: float, valid: bool, reset: bool) -> bytes:
    prefix = struct.pack("<IIfBBH", PRESSURE_MAGIC, index, pressure, valid, reset, 0)
    return prefix + struct.pack("<I", zlib.crc32(prefix) & 0xFFFFFFFF)


def read_exact(port, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = port.read(length - len(data))
        if not chunk:
            raise TimeoutError(f"STM32 telemetry timeout at {len(data)}/{length} bytes")
        data.extend(chunk)
    return bytes(data)


def telemetry(data: bytes) -> dict:
    if len(data) != OUTPUT_SIZE or zlib.crc32(data[:108]) & 0xFFFFFFFF != struct.unpack_from("<I", data, 108)[0]:
        raise ValueError("telemetry size/CRC failure")
    magic, index = struct.unpack_from("<II", data, 0)
    if magic != TELEMETRY_MAGIC:
        raise ValueError("telemetry magic failure")
    pressure, baseline = struct.unpack_from("<ff", data, 8)
    event_id, flags = struct.unpack_from("<II", data, 16)
    values = struct.unpack_from("<15f", data, 24)
    score, threshold = struct.unpack_from("<ff", data, 84)
    latched, stim_state, processing_us = struct.unpack_from("<III", data, 92)
    return {"sample_index": index, "pressure": pressure, "baseline": baseline,
            "candidate_event_id": event_id, "flags": flags,
            **dict(zip(FEATURES, values)), "score": score, "threshold": threshold,
            "latched_event_id": latched, "stim_state": stim_state,
            "processing_time_us": processing_us,
            "candidate_active": bool(flags & (1 << 1)),
            "feature_available": bool(flags & (1 << 2)),
            "score_positive": bool(flags & (1 << 3)), "t0_trigger": bool(flags & (1 << 4)),
            "shadow_mode": bool(flags & (1 << 5)),
            "stimulation_request": bool(flags & (1 << 6)),
            "stim_output_on": bool(flags & (1 << 7)),
            "config_valid": bool(flags & (1 << 8)),
            "stimulation_enabled": bool(flags & (1 << 9))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--subject", choices=("STxF37", "STxF26"), required=True)
    parser.add_argument("--baud", type=int, default=3_000_000)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--input", type=Path,
                        default=Path("Test/v5_results/generated/full_replay_golden.csv"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial before HIL: python -m pip install pyserial") from exc
    output = args.output or Path(f"Test/v5_hil/{args.subject}_hil_telemetry.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["animal"] == args.subject]
    results, prior_cycle = [], None
    with serial.Serial(args.port, args.baud, timeout=args.timeout) as port:
        for source in rows:
            cycle = source["cycle_id"]
            reset = cycle != prior_cycle
            index = int(source["sample_index"])
            port.write(pressure_frame(index, float(source["pressure"]), flag(source["signal_valid"]), reset))
            decoded = telemetry(read_exact(port, OUTPUT_SIZE))
            decoded.update({"animal": args.subject, "cycle_id": cycle,
                            "expected_registered": source["expected_registered"],
                            "expected_score": source["expected_score"],
                            "expected_t0_trigger": source["expected_t0_trigger"]})
            results.append(decoded)
            prior_cycle = cycle
            if args.realtime:
                time.sleep(0.01)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader(); writer.writerows(results)
    test = [r for r in results if r["expected_registered"].casefold() in {"true", "1"}]
    errors = [abs(r["score"] - float(r["expected_score"])) for r in test
              if r["candidate_active"] and r["feature_available"] and r["expected_score"]]
    expected_triggers = sum(r["expected_t0_trigger"].casefold() in {"true", "1"} for r in test)
    actual_triggers = sum(r["t0_trigger"] for r in test)
    max_us = max(r["processing_time_us"] for r in results)
    print(f"subject={args.subject} max_score_error={max(errors, default=0):.9g}")
    print(f"T0 triggers Python/STM32={expected_triggers}/{actual_triggers}")
    print(f"max_processing_time_us={max_us}")
    print(f"telemetry={output}")
    if any(not r["shadow_mode"] or r["stimulation_enabled"] or
           r["stimulation_request"] or r["stim_output_on"] for r in results):
        raise SystemExit("FAIL: HIL was not safely shadow-only")


if __name__ == "__main__":
    main()
