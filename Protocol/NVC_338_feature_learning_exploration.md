# SPARC338 NVC feature-learning exploration

Date: 2026-08-26

## Goal

Explore why the current Dataset338 NVC learner produces zero safe hits, and define a
development path that can show successful feature learning inside 338 before any
external validation or stimulation-enable decision.

## Literature takeaways

1. SCI rat cystometry papers usually define NVC as storage-phase bladder pressure
   contractions not associated with voiding. Reported analysis often uses event count,
   amplitude, duration, intercontraction interval, threshold/voiding pressure, voiding
   efficiency, and EUS pattern rather than a single decision-time pressure sample.
2. SCI rats can show NVCs that are absent in uninjured rats, while EUS activity separates
   into phasic bursting and tonic patterns during bladder contractions. This supports
   pressure-plus-EUS features, but EUS direction may vary across animals.
3. Several rat SCI studies use a high NVC amplitude definition, commonly around
   >=15 cmH2O for uninhibited contractions not associated with voiding. SPARC338 uses
   much smaller adaptive thresholds in mmHg, so subject-adaptive normalization is
   necessary, but amplitude alone is unlikely to separate early prevoid from NVC.
4. Real-time bladder-event work outside rats favors context-aware, multi-resolution
   pressure features: local/global trends, window size, sensitivity, wavelet or spectral
   structure, and efficient streaming implementation. This matches an embedded 338 path
   better than a large black-box model.

Useful source anchors:

- Comparison of complete and incomplete SCI in rats: NVCs and EUS phasic/tonic patterns.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2712947/
- SCI-induced bladder dysfunction urodynamic parameters: NVCs >15 cmH2O during filling.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6983271/
- Contusion vs transection SCI rat phenotypes: nonvoiding contractions and threshold pressure.
  https://escholarship.org/content/qt4gr9r1rm/qt4gr9r1rm.pdf
- Real-time bladder-event classification using Context-Aware Thresholding.
  https://pubmed.ncbi.nlm.nih.gov/26292331/
- Wireless conditional neuromodulation using CAT at 10 Hz.
  https://www.ics.org/2020/abstract/255

## Current 338 diagnosis

Authoritative local results:

- `data/DSD_nvc_results/final_model.json`: `HOLD_NO_CAUSAL_SEPARATION`.
- `data/DSD_nvc_v2_development/development_result_v2.json`:
  `HOLD_NO_SAFE_NONZERO_GAIN`, no `final_model_v2.json`.
- Frozen event labels: 45 events total, with 10 `NVC_CORE`, 30
  `PREVOID_PROGRESSIVE`, 1 `GREY_ZONE`, and 4 `INVALID`.
- Base decision eligibility in v2 is 10/10 NVC and 27/30 PREVOID at each 0, 0.5,
  and 1.0 s delay. This means the zero-hit result is not caused by lack of eligible
  NVC events.

Failure mode:

- C1/C2 logistic models do learn some analysis signal, but it is not safely separable.
  At analysis thresholds, C2 accepts 8-9 of 10 NVC events depending on delay, but also
  accepts 18-21 PREVOID events. The safety threshold therefore collapses to zero.
- C3 time-frequency features have promising univariate signal, but the strict
  preregistered 5 s current plus 25 s baseline requirement leaves only 5/10 NVC and
  16/27 PREVOID with all time-frequency features finite. Because v2 rejects any fold
  with missing preregistered features, C3 never actually tests a scored model.
- At the event confirm point, early `PREVOID_PROGRESSIVE` contractions can be tens of
  seconds before urine onset but are already on a long rising trajectory. A static
  classifier at confirm time is being asked to separate two physiologically similar
  rising pressure events before the recovery/progression evidence exists.

Best current feature signals from the v2 feature table:

- `pressure_low_band_fraction_0p2_0p6_ratio_5s`: AUC about 0.75 at 0-0.5 s.
- `pressure_bandpower_0p2_20_ratio_5s`: AUC about 0.76 at 1.0 s.
- `pressure_high_band_fraction_5_20_ratio_5s`: inverse AUC about 0.72-0.74 at
  0.5-1.0 s.
- `eus_tonic_occupancy`: AUC about 0.61-0.72, but prior feature separability marks
  EUS direction as cross-subject unstable.

## Development target

Separate two statuses:

1. Feature-learning success: 338-only nested animal-LOSO can produce nonzero NVC hits
   with an explicitly reported PREVOID/VOID false-trigger rate.
2. Stimulation-safe model freeze: zero dangerous triggers, at least two animals with
   NVC hits, and improvement over C0.

The current code treats status 2 as the only pass. For development, add an exploratory
status such as `PASS_338_FEATURE_LEARNING_NONZERO`, while keeping
`stimulation_enabled=false`.

## Recommended C4 exploratory model

Add a non-frozen, clearly named C4 exploratory path. Do not modify C0-C3 artifacts.

Feature groups:

1. Causal recovery morphology after confirm:
   - `drop_from_peak_to_now_norm`
   - `current_below_recent_peak_fraction`
   - `recovery_run_fraction_1s`
   - `slope_sign_changes_2s`
   - `post_confirm_min_slope_norm`
   - `time_since_last_new_peak_s`
2. Multi-resolution pressure context:
   - 1, 2, 5, and 10 s AUC growth
   - 1, 2, 5, and 10 s slope
   - low-band fraction around 0.2-0.6 Hz
   - total 0.2-20 Hz log bandpower vs a causal baseline
3. EUS features only as optional secondary evidence:
   - tonic occupancy
   - envelope slope
   - burst-band fraction
   - silence/burst duty-cycle metrics
4. Causal event-history features:
   - prior NVC-like candidate count in current filling cycle
   - time since previous recovered pressure event
   - current cycle time since previous void end

Decision design:

- Keep confirm-time scoring, but add delayed scoring at 3, 5, 8, and 10 seconds.
- For NVC, a delayed decision after recovery should be counted as a missed opportunity,
  not silently removed from the denominator.
- For PREVOID, keep a strict safety audit, but report the decision's remaining time to
  urine using consistent absolute times.
- Add a recovery-prediction target: `recovers_before_urine_within_H`, with H in
  5-15 seconds. This better matches the biology than asking a confirm-time snapshot to
  know the final event label.

Implementation sketch:

1. Create `Tools/dsd_feature_extraction/development_c4_exploratory.py`.
2. Reuse `load_locked_inputs`, `prior_and_cache`, `decision_feature_at_index`,
   `replay_final_triggers`, and nested animal-LOSO infrastructure.
3. Add a new feature helper, not a change to frozen `v2_time_frequency_features`, that
   computes shorter-window features with explicit missingness flags.
4. Write outputs under `data/DSD_nvc_c4_exploration/`:
   - `event_features_c4.csv`
   - `nested_loso_event_predictions_c4.csv`
   - `model_comparison_c4.csv`
   - `development_result_c4.json`
5. Gate `PASS_338_FEATURE_LEARNING_NONZERO` on nonzero LOSO NVC hits and complete
   accounting of dangerous triggers. Reserve `PASS_338_V2_SAFE_NONZERO_GAIN` for the
   stricter frozen/stimulation-safe path.

## Immediate fixes before C4

1. Fix v2 evaluation-time `remaining_to_urine_s` calculation for clarity. The code
   currently combines absolute `local_peak_time_s` and event-relative `time_to_urine_s`
   correctly for final urine onset, but the resulting large positive values at confirm
   time can be misread as a label bug. Write both `urine_onset_s` and
   `confirm_to_urine_s` explicitly.
2. Let exploratory time-frequency features report coverage instead of fold-holding on
   missing values. Missingness is informative when a 30 s causal history is unavailable.
3. Keep C1-C3 frozen outputs untouched, and make every exploratory artifact self-labeled
   as not externally validated and not stimulation enabled.
