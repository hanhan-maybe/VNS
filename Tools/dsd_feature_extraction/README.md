# Shared DSD signal and frozen-label pipeline

This package now serves as the neutral signal-processing and frozen-label
foundation used by the independently packaged V3–V5 model versions. It reads
but never modifies raw cycles, detects teacher urine events, applies the unified
causal subject-adaptive pressure formula, and provides causal pressure/EUS
feature utilities.

Volume and EUS never enter the adaptive pressure threshold or pressure
teacher-label detector. Model definitions and evaluation logic now live in the
separate version packages.

Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m Tools.dsd_feature_extraction.pipeline `
  --input-root "D:\cubeIDE\project\VNS\data\DSD_cycles" `
  --output-root "D:\cubeIDE\project\VNS\data\DSD_nvc_results"

.\.venv\Scripts\python.exe -m pytest Tools\dsd_feature_extraction\tests\test_core.py -q
```

The SPARC338 records contain no VNS/sham condition. Outputs are pre-stimulation
NVC baselines and shadow replay only; they are not VNS efficacy estimates or
sham controls.

Active model-development entry points are listed in `Tools/NVC_VERSIONS.md`.
Legacy pre-V3 model code is not part of this active package.
