# NVC model code layout

The active model-development code starts at V3.  Every version is an independent
Python package and may import only neutral signal/data utilities from
`Tools.dsd_feature_extraction`; it must not import another `Tools.nvc_v*` package.

| Version | Package | Primary entry point | Result directory |
|---|---|---|---|
| V3 | `Tools.nvc_v3` | `python -m Tools.nvc_v3.run` | `data/NVC_V3` |
| V3.1 | `Tools.nvc_v3_1` | `python -m Tools.nvc_v3_1.run` | `data/NVC_V3_1` |
| V3.2 | `Tools.nvc_v3_2` | `python -m Tools.nvc_v3_2.run` | `data/NVC_V3_2` |
| V4 | `Tools.nvc_v4` | `python -m Tools.nvc_v4.run` | `data/NVC_V4` |
| V5 parallel | `Tools.nvc_v5` | `python -m Tools.nvc_v5.run_parallel` | `data/NVC_V5` |
| V5 final | `Tools.nvc_v5` | `python -m Tools.nvc_v5.run_final_validation` | `data/NVC_V5/v5_final_validation` |

Raw cycles, frozen teacher labels, and adaptive detector parameters remain in
their source directories.  They are scientific inputs, not version results.

V3 retains two immutable Dataset338 source-feature snapshots under
`data/NVC_V3/source_inputs`.  This removes the runtime dependency on the deleted
V2/V2.1 model packages while preserving exact V3 reproducibility.
