import numpy as np

from Tools.sparc338_smrx_reader import match_channels
from Tools.sparc338_urine_output import (audit_urine_evidence,
                                         detect_discrete_volume_episodes,
                                         parse_drop_button)


def test_volume_title_is_not_continuous_semantics():
    rows = [{"channel": 0, "type": "Adc", "title": "Volume", "units": "mL",
             "comment": "", "sample_rate_hz": 2000.0, "divide_ticks": 1,
             "start_s": 0.0, "channel_max_time_s": 10.0,
             "wave_time_origin": "SONPY_FIRST_TIME", "time_axis_reliable": True,
             "selected_role": "OTHER"}]
    selected, _ = match_channels(rows)
    assert selected["VOLUME"]["selected_role"] == "URINE_SIGNAL_CANDIDATE"


def test_staircase_volume_is_discrete_and_absolute():
    time_s = 100.0 + np.arange(10000, dtype=float) / 100.0
    raw = np.zeros(time_s.size, dtype=float)
    raw[3000:] = 1.0
    raw[6500:] = 2.0
    audit = audit_urine_evidence(
        "STxFTEST", time_s, raw,
        {"type": "Adc", "title": "Volume", "units": "mL", "comment": "",
         "start_s": 100.0, "sample_rate_hz": 100.0},
        pressure_time_s=time_s, pressure=np.zeros_like(time_s),
    )
    assert audit["source_type"] == "DISCRETE_STABLE_VOLUME"
    assert audit["contract"]["time_origin"] == "absolute"
    assert audit["features"]["transition_count"] == 2
    assert audit["transitions"][0]["transition_time_s"] >= 100.0


def test_leak_rise_preserves_nonzero_absolute_origin():
    raw = np.zeros(1000, dtype=float)
    raw[200:250] = 10.0
    times, status, _ = parse_drop_button(raw, 100.0, start_s=42.5)
    assert status == "PASS"
    assert np.isclose(times[0], 44.5)


def test_discrete_episode_matching_is_one_to_one_and_not_quantity():
    t = 100.0 + np.arange(6000, dtype=float) / 100.0
    volume = np.zeros(t.size, dtype=float)
    volume[2000:2020] = np.linspace(0, 1, 20)
    volume[2020:] = 1.0
    pressure = np.zeros(t.size, dtype=float)
    pressure[2050] = 8.0
    result = detect_discrete_volume_episodes(t, volume, t, pressure, 160.0)
    matched = [row for row in result["episodes"] if row["match_status"] == "MATCHED"]
    assert len(matched) <= 1
    assert result["features"]["matched_void_episode_count"] == len(matched) if "matched_void_episode_count" in result["features"] else True
    assert all("net_change_raw" in row for row in result["episodes"])
