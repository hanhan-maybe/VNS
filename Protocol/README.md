# VNS Input Frame Protocol

## Overview

The PC-to-MCU data replay protocol transports dual-signal samples
(pressure and EUS envelope) from a host PC to the NUCLEO-N657X0-Q
target over a serial UART connection (default: 921600 baud, 8N1).

Frames are sent at a nominal rate of **100 Hz** (one sample per frame).

## Wire Format

36-byte binary frame, little-endian:

```
Offset  Size  Field             Type    Description
------  ----  ----------------- ------  -----------
     0     2  header            uint16  SOF = 0x55AA
     2     2  version           uint16  Protocol version = 1
     4     2  payload_length    uint16  26 (bytes after this field, before CRC)
     6     2  flags             uint16  Reserved (set to 0)
     8     4  sequence          uint32  Monotonically increasing counter
    12     8  timestamp_us      uint64  Microsecond timestamp from replay source
    20     4  pressure_q16      int32   Pressure sample in Q16.16 fixed-point
    24     4  eus_envelope_q16  int32   EUS envelope sample in Q16.16
    28     2  eus_flags         uint16  EUS status bits (reserved)
    30     2  reserved          uint16  Padding (set to 0)
    32     4  crc32             uint32  CRC-32 over bytes 0..31
```

Total frame size: **36 bytes**.

### Q16.16 Fixed-Point

The `pressure_q16` and `eus_envelope_q16` fields use signed Q16.16 format.

```
float_value = q16_value / 65536.0
q16_value   = (int32_t)(float_value * 65536.0)
```

Example: `pressure_q16 = 983040` → `983040 / 65536 = 15.0`

### CRC-32

Polynomial: `0xEDB88320` (reflected, standard Ethernet/zip CRC-32)

The CRC covers all 32 bytes preceding the `crc32` field
(bytes 0 through 31 of the frame).

Reference implementation (C):
```c
uint32_t FP_CRC32(const uint8_t *data, uint32_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (uint32_t i = 0; i < len; i++) {
        crc = s_crc32_table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFF;
}
```

## Frame Validation

On the MCU side, the `AcquisitionTask` validates each received frame:

1. **Header**: must equal `0x55AA`
2. **Version**: must equal `1`
3. **Payload length**: must equal `26`
4. **CRC-32**: computed over bytes 0..31, must match the stored CRC
5. **Sequence**: must be `previous_seq + 1` (wraparound allowed)
6. **Timestamp**: must be monotonically increasing

If any check fails, the corresponding error counter is incremented
and the frame is discarded.

## Data Timeout

If no valid frame is received for **500 ms**, the MCU asserts a
DATA_TIMEOUT condition, which:

- Sets a binary semaphore read by `StimTask`
- `StimTask` immediately aborts any ongoing stimulation
- Stimulation is inhibited until valid data resumes

## Serial Port Configuration

| Parameter  | Value    |
|------------|----------|
| Baud rate  | 921600   |
| Data bits  | 8        |
| Parity     | None     |
| Stop bits  | 1        |
| Flow ctrl  | None     |

## PC Replay Tool

`Tools/pc_replay.py` provides a reference host implementation:

```
python pc_replay.py --port COM3 --csv data.csv --speed 1 --loop
```

Features:
- Reads CSV with columns `timestamp_us`, `pressure`, `eus_envelope`
- Builds and sends 36-byte frames at the configured rate
- Speed multipliers: 0.5×, 1×, 2×, 5×
- Loop mode for continuous playback
- Fault injection: CRC errors, frame drops, communication pauses
- Real-time frame-count and drop statistics

## CSV Format

```csv
timestamp_us,pressure,eus_envelope
0,0.0,0.0
10000,12.5,0.8
20000,12.7,0.9
...
```

Timestamps are in microseconds, pressure and eus_envelope in physical
units (e.g., mmHg for pressure, volts for EUS envelope). The MCU
converts Q16.16 back to float for internal processing.
