#ifndef V5_RUNTIME_H
#define V5_RUNTIME_H

#include "v5_model.h"
#include <stdbool.h>
#include <stdint.h>

#define V5_FS_HZ 100u
#define V5_UPDATE_SAMPLES 25u

typedef struct {
    bool candidate_active;
    uint32_t candidate_event_id;
    bool recovery_event;
} V5CandidateInput;

typedef struct {
    V5ModelConfig model;
    bool model_loaded;
    bool shadow_mode;
    bool stimulation_enabled;

    uint32_t sample_index;
    uint32_t active_event_id;
    uint32_t latched_event_id;
    uint32_t t1_positive_count;
} V5Runtime;

typedef struct {
    uint32_t sample_index;
    uint32_t candidate_event_id;
    bool candidate_active;
    bool feature_available;
    float score;
    float threshold;
    bool score_positive;
    uint32_t t1_positive_count;
    bool t0_trigger;
    bool t1_trigger;
    bool virtual_trigger;
    bool stimulation_request;
} V5RuntimeOutput;

void V5Runtime_Init(V5Runtime *r);
bool V5Runtime_LoadModel(V5Runtime *r, const V5ModelConfig *cfg);
void V5Runtime_SetShadowMode(V5Runtime *r, bool shadow_mode);
void V5Runtime_EnableStimulation(V5Runtime *r, bool enable);

/* Call at 100 Hz. features may be NULL except on the regular 0.25 s update. */
V5RuntimeOutput V5Runtime_Step(V5Runtime *r,
                              V5CandidateInput candidate,
                              const float features[V5_MODEL_FEATURE_COUNT],
                              bool feature_available);

#endif
