# V5 STM32 HIL protocol

The host sends one 20-byte little-endian pressure frame per 100 Hz sample.
The board returns one 112-byte telemetry frame. Both frames end in IEEE CRC32.

Input fields: magic `V5PI`, sample index, pressure float32, validity, cycle-reset,
reserved zero, CRC32. Output fields: magic `V5TO`, sample index, pressure,
baseline, candidate event ID, status flags, 15 P-EARLY float32 values, score,
threshold, latched event ID, stimulation state, measured processing time in
microseconds, reserved zero, CRC32.

The board adapter supplies a monotonic microsecond callback to
`AppV5_ProcessHilFrame()`, which measures `AppV5_On100Hz()`. It must never map `stim_output_on` to a
physical output during HIL. Build-time `V5_ALLOW_PHYSICAL_STIMULATION` remains
zero by default.

Host command after a real serial port is available:

```powershell
python Test/v5_hil/replay_hil.py --port COMx --subject STxF37 --realtime
python Test/v5_hil/replay_hil.py --port COMx --subject STxF26 --realtime
```

HIL remains `NOT_EXECUTED_REQUIRES_HARDWARE` until both commands pass against a
NUCLEO-N657X0-Q build with no lost samples and a measured maximum processing
time below the 10 ms input deadline.
