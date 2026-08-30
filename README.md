# VNS

STM32 VNS firmware and offline NVC model-development pipeline.

## Repository scope

- `Core/`, `Modules/`, `Drivers/`, `Middlewares/`, `USB_DEVICE/`: STM32 firmware.
- `Tools/`: data extraction, quality control, feature processing, and independent
  NVC model packages V3–V5.
- `Protocol/`: experiment and NVC feature-learning specifications.
- `Test/`: firmware/data-pipeline tests.

The local `data/` directory is intentionally excluded from GitHub because it
contains raw physiological recordings, derived cycles, frozen labels, and model
results. Recreate the expected local data layout from the project documentation
before running the offline pipelines.

## NVC versions

See [`Tools/NVC_VERSIONS.md`](Tools/NVC_VERSIONS.md) for version entry points.
The current V5 final validation entry point is:

```powershell
python -m Tools.nvc_v5.run_final_validation
```

The full active test suite can be run with:

```powershell
python -m pytest Tools/dsd_feature_extraction/tests Tools/nvc_v3/tests Tools/nvc_v3_1/tests Tools/nvc_v3_2/tests Tools/nvc_v4/tests Tools/nvc_v5/tests -q
```

Data and generated outputs remain local and are covered by `.gitignore`.
