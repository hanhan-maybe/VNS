# V5 individualized prospective NVC detection

- Subjects: STxF37 and STxF26, fitted independently.
- Validation: earlier cycles calibrate; later cycles are frozen prospective replay.
- Selected final model: M1 P-EARLY with causal pressure-candidate gating.
- Code dependencies: neutral `Tools.dsd_feature_extraction` utilities only.
- Five-model run: `python -m Tools.nvc_v5.run_parallel`.
- Final replay: `python -m Tools.nvc_v5.run_final_validation`.
- Results: `data/NVC_V5`.
