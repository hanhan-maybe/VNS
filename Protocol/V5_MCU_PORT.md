# V5 MCU C port

The pressure-only Python-to-C algorithm port and generic SubjectConfig swap
have passed host validation. See `Protocol/V5_MCU_IMPLEMENTATION_STATUS.md`
for the authoritative Gate table and remaining STM32/HIL work.

Algorithm source of truth:

- `Tools/nvc_v5/config.py`
- `Tools/nvc_v5/parallel.py`
- `Tools/nvc_v5/modeling.py`
- `Tools/nvc_v5/final_validation.py`
- neutral causal detector definition in `Tools/dsd_feature_extraction/detectors.py`

Historic code outside `Tools` is not used as an algorithm reference.

## Phase-1 files

- `Modules/Inc|Src/v5_model.*`: exact StandardScaler + LogisticRegression inference.
- `Modules/Inc|Src/v5_runtime.*`: candidate-gated T0/T1/latch runtime.
- `Modules/Inc|Src/v5_stim_fsm.*`: fail-safe stimulation state machine; shadow by default.
- `Tools/nvc_v5/export_mcu_config.py`: frozen Python model -> C header.
- `Core/Src/v5_integration_example.c`: 100 Hz integration example.

## Implemented after phase 1

The following are now ported from current Python and verified against F26/F37:

1. `adaptive_local_pressure_events` -> `V5CandidateInput`.
2. `extract_p_early_features` -> 15-element feature vector.
3. 5 s pressure spectrum with the same linear detrend, Hann, FFT/PSD, band
   integration and spectral entropy definitions.

Do not reuse the old `Modules/feature_extraction.*` or `classifier.*` logic.

## Remaining development order

1. Freeze current Python V5 final validation.
2. Run `python -m Tools.nvc_v5.export_mcu_config STxF26` and STxF37.
3. Compile phase-1 C on PC first.
4. Restore/generate the STM32N657 CubeMX project skeleton from `VNS_N6.ioc`.
5. Replace portable DFT with a parity-checked CMSIS-DSP FFT if needed.
6. Run F37/F26 UART/USB HIL and verify the 10 ms deadline.
7. First in-vivo run: `shadow_mode=true`, `stimulation_enabled=false`.
8. Only after shadow safety audit may a controlled build explicitly set
   `V5_ALLOW_PHYSICAL_STIMULATION=1`.
