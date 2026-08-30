"""Batch extraction entry point for SPARC Dataset 338 SCI PRE_STIM data."""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import numpy as np

try:
    from sparc338_smrx_reader import (EVENT_TYPES, WAVE_TYPES, close_smrx, match_channels, open_smrx,
                                      read_event_times, read_keyboard, read_wave_before, scan_inventory,
                                      validate_wave_time_axis)
    from sparc338_preprocessing import (FS_TARGET, align_100hz, build_phase_segments,
                                        group_stim_trains, preprocess_bladder, preprocess_eus,
                                        resample_valid_100hz)
    from sparc338_pre_stim_qc import build_summary, make_quicklooks
    from sparc338_urine_output import (assess_volume_correspondence, candidate_contractions,
                                       LEAK_VISUAL_REVIEW, VOLUME_VISUAL_REVIEW,
                                       make_candidate_plots, parse_drop_button,
                                       volume_display, write_volume_qc_csv)
    from sparc338_common import (cleanup_staging, commit_directory, make_staging_directory,
                                 read_csv, source_record, write_csv_atomic, write_json_atomic)
    from sparc338_config import (BASELINE_ROOT, RAW_ROOT, SCI_SUBJECTS, SUBJECT_REGISTRY)
except ImportError:  # Package import for tests and programmatic use.
    from Tools.sparc338_smrx_reader import (EVENT_TYPES, WAVE_TYPES, close_smrx, match_channels, open_smrx,
                                            read_event_times, read_keyboard, read_wave_before, scan_inventory,
                                            validate_wave_time_axis)
    from Tools.sparc338_preprocessing import (FS_TARGET, align_100hz, build_phase_segments,
                                              group_stim_trains, preprocess_bladder, preprocess_eus,
                                              resample_valid_100hz)
    from Tools.sparc338_pre_stim_qc import build_summary, make_quicklooks
    from Tools.sparc338_urine_output import (assess_volume_correspondence, candidate_contractions,
                                             LEAK_VISUAL_REVIEW, VOLUME_VISUAL_REVIEW,
                                             make_candidate_plots, parse_drop_button,
                                             volume_display, write_volume_qc_csv)
    from Tools.sparc338_common import (cleanup_staging, commit_directory, make_staging_directory,
                                       read_csv, source_record, write_csv_atomic, write_json_atomic)
    from Tools.sparc338_config import (BASELINE_ROOT, RAW_ROOT, SCI_SUBJECTS, SUBJECT_REGISTRY)


SUBJECTS = list(SCI_SUBJECTS)


def resolve_subject_provenance(subject: str, source_dataset_id: str | int) -> dict:
    """Resolve frozen Dataset338 provenance or an external dataset subject."""
    dataset_id = str(source_dataset_id)
    if subject in SUBJECT_REGISTRY:
        return {"source_dataset_id": dataset_id, **dict(SUBJECT_REGISTRY[subject])}
    if dataset_id == "338":
        raise KeyError(
            f"Unregistered subject {subject!r} cannot be declared as Dataset338"
        )
    return {
        "source_dataset_id": dataset_id,
        "local_subject_id": subject,
        "dsd_confirmed": None,
        "urine_source": "AUTO_DISCOVER_FROM_SMRX",
        "urine_review_status": "AUTOMATIC_ONLY",
        "manual_review_used": False,
    }


def write_csv(path: Path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    write_csv_atomic(path, rows, fieldnames)


def _process_subject_to_directory(source: Path, out_dir: Path,
                                  include_source_sha256: bool = False,
                                  source_dataset_id: str = "338"):
    subject = source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open_smrx(source)
    try:
        return _process_open_subject(source, out_dir, f, include_source_sha256, source_dataset_id)
    finally:
        close_smrx(f)


def _process_open_subject(source: Path, out_dir: Path, f,
                          include_source_sha256: bool = False,
                          source_dataset_id: str = "338"):
    subject = source.stem
    provenance = resolve_subject_provenance(subject, source_dataset_id)
    tb = float(f.GetTimeBase())
    record_duration_s = float(f.MaxTime() * tb)
    inventory = scan_inventory(f, subject)
    for row in inventory:
        row["source_dataset_id"] = str(source_dataset_id)
    selected, warnings = match_channels(inventory)
    write_csv(out_dir / "channel_inventory.csv", inventory)
    missing = [role for role in ("BLADDER", "EUS_RAW", "STIM") if selected[role] is None]
    if missing:
        candidates = {
            role: [row["channel"] for row in inventory
                   if row.get("selected_role") == "OTHER" and (
                       (role == "BLADDER" and ("pressure" in str(row.get("title", "")).casefold()
                                                or "cmg" in str(row.get("title", "")).casefold()))
                       or (role == "EUS_RAW" and str(row.get("title", "")).casefold() == "eus")
                       or (role == "STIM" and str(row.get("title", "")).casefold() == "stim")
                   )]
            for role in missing
        }
        raise RuntimeError(f"Missing or ambiguous required channels: {missing}; candidates={candidates}")

    stim_times = read_event_times(f, selected["STIM"]["channel"], tb)
    trains = group_stim_trains(stim_times)
    if not trains:
        raise RuntimeError("No stimulation train found")
    for train in trains:
        train["subject"] = subject
    train_fields = ["subject", "train_id", "start_s", "end_s", "duration_s", "pulse_count", "mean_frequency_hz"]
    write_csv(out_dir / "stim_trains.csv", trains, train_fields)
    first_stim_s = float(trains[0]["start_s"])

    bladder = selected["BLADDER"]
    eus = selected["EUS_RAW"]
    volume = selected["VOLUME"]
    time_meta = {
        "bladder_start_s": float(bladder.get("start_s", 0.0)),
        "eus_start_s": float(eus.get("start_s", 0.0)),
        "volume_start_s": float(volume.get("start_s", 0.0)) if volume is not None else None,
        "wave_time_origin": ";".join(sorted({str(x.get("wave_time_origin", "UNKNOWN"))
                                               for x in (bladder, eus, volume) if x is not None})),
    }
    if (not bool(bladder.get("time_axis_reliable", False))
            or not bool(eus.get("time_axis_reliable", False))):
        raise RuntimeError(
            "DATA_INVALID: unreliable BLADDER/EUS time origin; "
            f"time_metadata={time_meta}"
        )
    for row in inventory:
        if row.get("type") in WAVE_TYPES:
            row.update(time_meta)
    write_csv(out_dir / "channel_inventory.csv", inventory)

    phases = build_phase_segments(record_duration_s, trains)
    for phase in phases:
        phase["subject"] = subject
    phase_fields = ["subject", "phase_id", "phase_type", "start_s", "end_s", "duration_s", "stim_train_id"]
    write_csv(out_dir / "phase_segments.csv", phases, phase_fields)

    bladder_raw = read_wave_before(f, bladder, first_stim_s, start_s=time_meta["bladder_start_s"])
    eus_raw = read_wave_before(f, eus, first_stim_s, start_s=time_meta["eus_start_s"])
    if (not validate_wave_time_axis(f, bladder, len(bladder_raw), time_meta["bladder_start_s"], first_stim_s, float(bladder["sample_rate_hz"]))
            or not validate_wave_time_axis(f, eus, len(eus_raw), time_meta["eus_start_s"], first_stim_s, float(eus["sample_rate_hz"]))):
        raise RuntimeError("DATA_INVALID: waveform end time does not match SMRX absolute sample axis")
    bladder["time_axis_reliable"] = True
    eus["time_axis_reliable"] = True
    leak_raw = np.empty(0, dtype=np.float32)
    leak_fs = np.nan
    leak_event_times = np.empty(0, dtype=np.float64)
    leak = selected["LEAK"]
    if leak is not None:
        if leak["type"] in WAVE_TYPES:
            leak_raw = read_wave_before(f, leak, first_stim_s, start_s=float(leak.get("start_s", 0.0)))
            leak_fs = float(leak["sample_rate_hz"])
        elif leak["type"] in EVENT_TYPES:
            leak_event_times = read_event_times(f, leak["channel"], tb)
            leak_event_times = leak_event_times[leak_event_times < first_stim_s]

    keyboard_times = np.empty(0, dtype=np.float64)
    keyboard_codes = np.empty((0, 4), dtype=np.int16)
    keyboard_text = np.empty(0, dtype="U1")
    keyboard = selected["KEYBOARD"]
    if keyboard is not None:
        keyboard_times, keyboard_codes, keyboard_text = read_keyboard(f, keyboard["channel"], tb, first_stim_s)
        strict = keyboard_times < first_stim_s
        keyboard_times, keyboard_codes, keyboard_text = keyboard_times[strict], keyboard_codes[strict], keyboard_text[strict]

    bladder_native_valid = np.isfinite(bladder_raw)
    eus_native_valid = np.isfinite(eus_raw)
    raw_payload = {
        "subject_id": np.array(subject),
        "source_dataset_id": np.array(str(source_dataset_id)),
        "bladder_raw": bladder_raw, "bladder_fs_hz": np.array(bladder["sample_rate_hz"]),
        "bladder_units": np.array(bladder["units"]),
        "eus_raw": eus_raw, "eus_fs_hz": np.array(eus["sample_rate_hz"]), "eus_units": np.array(eus["units"]),
        "pre_stim_start_s": np.array(0.0), "pre_stim_end_s": np.array(first_stim_s),
        "bladder_start_s": np.array(time_meta["bladder_start_s"]),
        "eus_start_s": np.array(time_meta["eus_start_s"]),
        "volume_start_s": np.array(time_meta["volume_start_s"] if time_meta["volume_start_s"] is not None else np.nan),
        "wave_time_origin": np.array(time_meta["wave_time_origin"]),
        "keyboard_event_times_s": keyboard_times, "keyboard_event_codes": keyboard_codes,
        "keyboard_event_text": keyboard_text,
        "bladder_native_valid": bladder_native_valid,
        "eus_native_valid": eus_native_valid,
    }
    if leak_raw.size:
        raw_payload.update({"leak_raw": leak_raw, "leak_fs_hz": np.array(leak_fs)})
    if leak_event_times.size:
        raw_payload["leak_event_times_s"] = leak_event_times
    # Raw arrays are intentionally stored without compression: this keeps the
    # export lossless and avoids CPU-heavy recompression of tens of millions of samples.
    np.savez(out_dir / "pre_stim_raw.npz", **raw_payload)

    bladder_100, _, bladder_valid_100, bladder_processing_qc = preprocess_bladder(
        bladder_raw, float(bladder["sample_rate_hz"]), return_qc=True
    )
    eus_100, _, eus_valid_100, eus_processing_qc = preprocess_eus(
        eus_raw, float(eus["sample_rate_hz"]), return_qc=True
    )
    for signal_name, qc in (("bladder", bladder_processing_qc), ("EUS", eus_processing_qc)):
        if qc["longest_nonfinite_gap_s"] > 0.10:
            warnings.append(
                f"{signal_name} contains a nonfinite acquisition gap of "
                f"{qc['longest_nonfinite_gap_s']:.3f} s; affected cycles require exclusion"
            )
    time_s, bladder_100, eus_100 = align_100hz(
        bladder_100, eus_100, first_stim_s,
        bladder_start_s=time_meta["bladder_start_s"], eus_start_s=time_meta["eus_start_s"],
    )
    bladder_valid_100 = resample_valid_100hz(bladder_valid_100, time_meta["bladder_start_s"], time_s)
    eus_valid_100 = resample_valid_100hz(eus_valid_100, time_meta["eus_start_s"], time_s)
    np.savez_compressed(
        out_dir / "pre_stim_100Hz.npz", subject_id=np.array(subject), source_dataset_id=np.array(str(source_dataset_id)), time_s=time_s,
        bladder_pressure_mmHg=bladder_100, eus_envelope_mV=eus_100,
        bladder_valid_100hz=bladder_valid_100, eus_valid_100hz=eus_valid_100,
        sample_rate_hz=np.array(FS_TARGET), stim_state=np.zeros(time_s.size, dtype=np.uint8),
        wave_time_origin=np.array(time_meta["wave_time_origin"]),
    )

    candidates = candidate_contractions(time_s, bladder_100, n_candidates=5)
    urine_mode = "NONE"
    urine_time = np.empty(0, dtype=np.float64)
    urine_trace = np.empty(0, dtype=np.float32)
    urine_derivative = np.empty(0, dtype=np.float32)
    drop_times = np.empty(0, dtype=np.float64)
    urine_status = "UNRESOLVED"
    urine_info = {
        "subject": subject, "acquisition_mode": "UNRESOLVED",
        "physiological_correspondence_status": "UNRESOLVED",
        "acquisition_semantics": "UNRESOLVED",
        "source_channel": None, "source_title": None, "source_units": None,
        "sample_rate_hz": None, "interpretation": "No synchronized urine-output channel available",
        "qc_status": "UNRESOLVED",
    }
    if volume is not None:
        volume_raw = read_wave_before(f, volume, first_stim_s, start_s=time_meta["volume_start_s"])
        volume_fs = float(volume["sample_rate_hz"])
        urine_time_native = time_meta["volume_start_s"] + np.arange(len(volume_raw), dtype=np.float64) / volume_fs
        if not validate_wave_time_axis(f, volume, len(volume_raw), time_meta["volume_start_s"], first_stim_s, volume_fs):
            raise RuntimeError("DATA_INVALID: Volume waveform end time does not match SMRX absolute sample axis")
        volume["time_axis_reliable"] = True
        if (len(urine_time_native) != len(volume_raw)
                or (len(urine_time_native) > 1 and not np.all(np.diff(urine_time_native) > 0))
                or (len(urine_time_native) and not np.all(urine_time_native < first_stim_s))):
            raise RuntimeError("DATA_INVALID: Volume time axis is not strictly increasing absolute pre-stim time")
        np.savez(
            out_dir / "pre_stim_urine_output.npz", subject_id=np.array(subject),
            time_s=urine_time_native, urine_output_raw=volume_raw,
            units=np.array(volume["units"]), sample_rate_hz=np.array(volume_fs),
            source_channel=np.array(volume["channel"]), channel_type=np.array(volume.get("type", "")),
            channel_title=np.array(volume.get("title", "")), channel_comment=np.array(volume.get("comment", "")),
            start_s=np.array(time_meta["volume_start_s"]), time_origin=np.array("absolute"),
        )
        urine_time, urine_trace, urine_derivative = volume_display(volume_raw, volume_fs)
        urine_time = time_meta["volume_start_s"] + urine_time
        n = min(len(urine_time), len(time_s)); urine_time = urine_time[:n]
        urine_trace = urine_trace[:n]; urine_derivative = urine_derivative[:n]
        count, fraction, classification, metrics, notes = assess_volume_correspondence(
            urine_time, urine_trace, candidates)
        if subject in VOLUME_VISUAL_REVIEW:
            classification, review_note = VOLUME_VISUAL_REVIEW[subject]
            notes = f"{notes} {review_note}"
        volume_qc = {
            "subject": subject, "volume_channel": volume["channel"], "type": volume["type"],
            "units": volume["units"], "sample_rate_hz": volume_fs,
            "n_candidate_voids": len(candidates), "n_corresponding_volume_changes": count,
            "correspondence_fraction": fraction, "classification": classification, "notes": notes,
        }
        write_volume_qc_csv(out_dir / "volume_channel_qc.csv", volume_qc)
        urine_mode = "VOLUME" if classification in {"CONFIRMED_URINE_OUTPUT", "LIKELY_URINE_OUTPUT"} else "VOLUME_REJECTED"
        urine_status = classification
        urine_info = {
            "subject": subject,
            "source_dataset_id": str(source_dataset_id),
            "physiological_correspondence_status": classification,
            "acquisition_semantics": "URINE_SIGNAL_CANDIDATE",
            "acquisition_mode": "UNRESOLVED",
            "source_channel": volume["channel"], "source_title": volume["title"],
            "source_type": volume["type"], "source_comment": volume.get("comment", ""),
            "start_s": time_meta["volume_start_s"], "time_origin": "absolute",
            "source_units": volume["units"], "sample_rate_hz": volume_fs,
            "interpretation": "Volume-labelled urine signal candidate; continuity and acquisition semantics require evidence audit",
            "qc_status": classification, "candidate_metrics": metrics, "notes": notes,
        }
    elif leak_raw.size:
        drop_times, drop_status, drop_notes = parse_drop_button(
            leak_raw, float(leak_fs), start_s=float(leak.get("start_s", 0.0))
        )
        if subject in LEAK_VISUAL_REVIEW:
            drop_notes = f"{drop_notes} {LEAK_VISUAL_REVIEW[subject]}"
        urine_mode = "DROP_EVENTS" if drop_status == "PASS" else "DROP_UNRESOLVED"
        urine_status = drop_status
        if drop_status != "PASS":
            warnings.append("Leak/drop signal could not be reliably parsed into button events")
        urine_info = {
            "subject": subject,
            "physiological_correspondence_status": "LEAK_EVENT_CANDIDATE" if drop_status == "PASS" else "UNRESOLVED",
            "acquisition_semantics": "LEAK_BUTTON_EVENT" if drop_status == "PASS" else "UNRESOLVED",
            "acquisition_mode": "EARLY_DROP_MARKING" if drop_status == "PASS" else "UNRESOLVED",
            "source_channel": leak["channel"], "source_title": leak["title"],
            "source_type": leak["type"], "source_comment": leak.get("comment", ""),
            "start_s": float(leak.get("start_s", 0.0)), "time_origin": "absolute",
            "source_units": leak["units"], "sample_rate_hz": float(leak_fs),
            "interpretation": "Adaptive rising-edge parsing of button/drop signal; events are not VOID labels",
            "qc_status": drop_status, "drop_event_count": int(len(drop_times)), "notes": drop_notes,
        }
    urine_info["source_dataset_id"] = str(source_dataset_id)
    urine_info["subject_provenance"] = provenance
    urine_info["subject_registry"] = dict(SUBJECT_REGISTRY[subject]) if subject in SUBJECT_REGISTRY else None
    urine_info.update(time_meta)
    write_json_atomic(out_dir / "urine_output_info.json", urine_info)
    make_candidate_plots(
        out_dir, subject, time_s, bladder_100, eus_100, candidates, urine_mode,
        urine_time=urine_time, urine_trace=urine_trace, urine_derivative=urine_derivative,
        drop_times=drop_times, source_units=(volume["units"] if volume else (leak["units"] if leak else "")),
    )

    events = []
    source_keyboard = keyboard["channel"] if keyboard else ""
    for t, code, char in zip(keyboard_times, keyboard_codes, keyboard_text):
        events.append({"subject": subject, "event_type": "KEYBOARD", "time_s": float(t),
                       "value": char if char else int(code[0]), "code1": int(code[0]),
                       "code2": int(code[1]), "code3": int(code[2]), "code4": int(code[3]),
                       "text": str(char), "source_channel": source_keyboard})
    source_leak = leak["channel"] if leak else ""
    for t in leak_event_times:
        events.append({"subject": subject, "event_type": "LEAK", "time_s": float(t),
                       "value": 1, "code1": "", "code2": "", "code3": "", "code4": "",
                       "text": "", "source_channel": source_leak})
    for t in drop_times:
        if t < first_stim_s:
            events.append({"subject": subject, "event_type": "LEAK", "time_s": float(t),
                           "value": "drop_button_rise", "code1": "", "code2": "", "code3": "", "code4": "",
                           "text": "drop_button_rise", "source_channel": source_leak})
    events.sort(key=lambda r: r["time_s"])
    write_csv(out_dir / "pre_stim_events.csv", events,
              ["subject", "event_type", "time_s", "value", "code1", "code2", "code3", "code4", "text", "source_channel"])

    summary = build_summary(
        subject, record_duration_s, first_stim_s, bladder_raw, float(bladder["sample_rate_hz"]),
        eus_raw, float(eus["sample_rate_hz"]), time_s, bladder_100, eus_100,
        len(leak_event_times) + len(drop_times), len(keyboard_times), warnings,
    )
    status = "PASS" if not summary["warnings"] else "PASS_WITH_WARNING"
    if not all(summary["checks"].values()):
        status = "FAIL"
    summary["status"] = status
    summary["source_dataset_id"] = str(source_dataset_id)
    summary["subject_provenance"] = provenance
    summary["subject_registry"] = dict(SUBJECT_REGISTRY[subject]) if subject in SUBJECT_REGISTRY else None
    summary.update(time_meta)
    summary["signal_preprocessing_qc"] = {
        "bladder": bladder_processing_qc,
        "eus": eus_processing_qc,
    }
    summary["source"] = source_record(source, include_sha256=include_source_sha256)
    summary["algorithm_contract"] = {
        "sample_interval": "0 <= t < first_stim_s",
        "filtering": "CAUSAL_FORWARD_ONLY",
        "raw_native_preserved": True,
        "urine_role": "OFFLINE_CONFIRMATION_NOT_ONLINE_TRIGGER",
    }
    write_json_atomic(out_dir / "pre_stim_summary.json", summary)
    make_quicklooks(out_dir, subject, first_stim_s, time_s, bladder_100, eus_100,
                    urine_mode=urine_mode, urine_time=urine_time, urine_trace=urine_trace,
                    drop_times=drop_times, urine_units=(volume["units"] if volume else ""),
                    urine_status=urine_status)
    return {
        "subject": subject, "source_dataset_id": str(source_dataset_id), "source_file": str(source), "status": status,
        "record_duration_s": record_duration_s, "first_stim_s": first_stim_s,
        "pre_stim_duration_s": first_stim_s,
        "bladder_channel": bladder["channel"], "bladder_fs_hz": bladder["sample_rate_hz"], "bladder_units": bladder["units"],
        "eus_channel": eus["channel"], "eus_fs_hz": eus["sample_rate_hz"], "eus_units": eus["units"],
        "stim_channel": selected["STIM"]["channel"],
        **time_meta,
        "leak_available": leak is not None, "keyboard_available": keyboard is not None,
        "n_stim_trains": len(trains), "n_leak_events": len(leak_event_times) + len(drop_times),
        "n_keyboard_events": len(keyboard_times), "samples_100Hz": len(time_s),
        "setup_warning": summary["setup_warning"], "warnings": " | ".join(summary["warnings"]),
    }


def process_subject(source: Path, output_root: Path,
                    include_source_sha256: bool = False,
                    source_dataset_id: str = "338"):
    """Build one subject in staging and replace its folder only after success."""
    target = output_root / source.stem
    staging = make_staging_directory(target)
    try:
        result = _process_subject_to_directory(source, staging, include_source_sha256, source_dataset_id)
        commit_directory(staging, target)
        return result
    except Exception:
        cleanup_staging(staging)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--subjects", nargs="*", default=SUBJECTS)
    parser.add_argument("--source-dataset-id", default="338")
    parser.add_argument("--hash-source", action="store_true",
                        help="Also compute the potentially slow SHA256 of each raw SMRX file")
    args = parser.parse_args()
    subjects = list(dict.fromkeys(args.subjects))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in subjects:
        source = args.raw_dir / f"{subject}.smrx"
        try:
            row = process_subject(source, args.output_dir, args.hash_source, str(args.source_dataset_id))
            print(f"{subject}: {row['status']} first_stim_s={row['first_stim_s']:.6f}", flush=True)
        except Exception as exc:
            row = {"subject": subject, "source_dataset_id": str(args.source_dataset_id),
                   "source_file": str(source), "status": "FAIL",
                   "warnings": f"{type(exc).__name__}: {exc}"}
            print(f"{subject}: FAIL {row['warnings']}", flush=True)
            traceback.print_exc()
        rows.append(row)
    fieldnames = ["subject", "source_dataset_id", "source_file", "status", "record_duration_s", "first_stim_s", "pre_stim_duration_s",
                  "bladder_channel", "bladder_fs_hz", "bladder_units", "eus_channel", "eus_fs_hz", "eus_units",
                  "bladder_start_s", "eus_start_s", "volume_start_s", "wave_time_origin",
                  "stim_channel", "leak_available", "keyboard_available", "n_stim_trains", "n_leak_events",
                  "n_keyboard_events", "samples_100Hz", "setup_warning", "warnings"]
    normalized = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    write_csv(args.output_dir / "pre_stim_inventory.csv", normalized, fieldnames)
    volume_rows = []
    for subject in subjects:
        path = args.output_dir / subject / "volume_channel_qc.csv"
        if path.exists():
            volume_rows.extend(read_csv(path))
    volume_fields = ["subject", "volume_channel", "type", "units", "sample_rate_hz", "n_candidate_voids",
                     "n_corresponding_volume_changes", "correspondence_fraction", "classification", "notes"]
    write_csv(args.output_dir / "volume_channel_qc.csv", volume_rows, volume_fields)


if __name__ == "__main__":
    main()
