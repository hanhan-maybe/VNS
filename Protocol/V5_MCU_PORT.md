# V5 MCU C port — phase 1

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

## What is intentionally not guessed in phase 1

The following must be ported from current Python and verified against the existing
F26/F37 streaming vectors before STM32 deployment:

1. `adaptive_local_pressure_events` -> `V5CandidateInput`.
2. `extract_p_early_features` -> 15-element feature vector.
3. 5 s pressure spectrum with the same linear detrend, Hann, FFT/PSD, band
   integration and spectral entropy definitions.

Do not reuse the old `Modules/feature_extraction.*` or `classifier.*` logic.

## Correct development order

1. Freeze current Python V5 final validation.
2. Run `python -m Tools.nvc_v5.export_mcu_config STxF26` and STxF37.
3. Compile phase-1 C on PC first.
4. Port candidate detector from `Tools/dsd_feature_extraction/detectors.py`.
5. Port P-EARLY from `Tools/nvc_v5/parallel.py` + `feature_extraction.py`.
6. Replay `F37_F26_streaming_test_vectors.csv` through C.
7. Require same candidate event IDs, trigger counts and event attribution;
   trigger time must agree within one 0.25 s update.
8. Only then integrate STM32N657 HAL/CMSIS.
9. First in-vivo run: `shadow_mode=true`, `stimulation_enabled=false`.
10. Only after shadow safety audit enable physical VNS output.
