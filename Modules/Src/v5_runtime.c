#include "v5_runtime.h"
#include <stddef.h>
#include <string.h>

void V5Runtime_Init(V5Runtime *r)
{
    if (r == NULL) return;
    memset(r, 0, sizeof(*r));
    r->shadow_mode = true;
    r->stimulation_enabled = false;
}

bool V5Runtime_LoadModel(V5Runtime *r, const V5ModelConfig *cfg)
{
    if (r == NULL || !V5Model_ValidateConfig(cfg)) return false;
    r->model = *cfg;
    r->model_loaded = true;
    return true;
}

void V5Runtime_SetShadowMode(V5Runtime *r, bool shadow_mode)
{
    if (r == NULL) return;
    r->shadow_mode = shadow_mode;
    if (shadow_mode) r->stimulation_enabled = false;
}

void V5Runtime_EnableStimulation(V5Runtime *r, bool enable)
{
    if (r == NULL) return;
    r->stimulation_enabled = enable && !r->shadow_mode;
}

V5RuntimeOutput V5Runtime_Step(V5Runtime *r,
                              V5CandidateInput candidate,
                              const float features[V5_MODEL_FEATURE_COUNT],
                              bool feature_available)
{
    V5RuntimeOutput out = {0};
    bool regular_update;

    if (r == NULL) return out;

    out.sample_index = r->sample_index;
    out.candidate_active = candidate.candidate_active;
    out.candidate_event_id = candidate.candidate_event_id;
    out.threshold = r->model.threshold;

    if (candidate.recovery_event || !candidate.candidate_active) {
        r->t1_positive_count = 0u;
        r->active_event_id = 0u;
        if (candidate.recovery_event && r->latched_event_id == candidate.candidate_event_id) {
            r->latched_event_id = 0u;
        }
    } else if (r->active_event_id != candidate.candidate_event_id) {
        r->active_event_id = candidate.candidate_event_id;
        r->t1_positive_count = 0u;
    }

    regular_update = (r->sample_index % V5_UPDATE_SAMPLES) == 0u;

    if (regular_update && candidate.candidate_active &&
        r->model_loaded && feature_available && features != NULL) {
        V5ModelOutput m = V5Model_Infer(&r->model, features);
        out.feature_available = m.valid;
        if (m.valid) {
            out.score = m.probability;
            out.score_positive = m.positive;
            if (m.positive) r->t1_positive_count++;
            else r->t1_positive_count = 0u;

            out.t0_trigger = m.positive &&
                             r->latched_event_id != candidate.candidate_event_id;
            out.t1_trigger = r->t1_positive_count >= 2u &&
                             r->latched_event_id != candidate.candidate_event_id;

            /* First MCU policy: T1 is the real trigger; T0 is telemetry only. */
            if (out.t1_trigger) r->latched_event_id = candidate.candidate_event_id;
        }
    }

    out.t1_positive_count = r->t1_positive_count;
    out.virtual_trigger = out.t1_trigger;
    out.stimulation_request = out.t1_trigger &&
                              r->stimulation_enabled &&
                              !r->shadow_mode;

    r->sample_index++;
    return out;
}
