#ifndef V5_CANDIDATE_H
#define V5_CANDIDATE_H

#include <stdbool.h>
#include <stdint.h>

#define V5_CANDIDATE_BASELINE_SAMPLES 2500u
#define V5_CANDIDATE_HISTORY_SAMPLES 30000u

typedef struct {
    float prior_sigma_p;
    float prior_sigma_dpdt;
} V5CandidateConfig;

typedef struct {
    float start;
    float confirm;
    float recovery;
    float sigma_p;
    float sigma_dpdt;
    bool warmup;
} V5CandidateThresholds;

typedef struct {
    bool candidate_active;
    uint32_t candidate_event_id;
    bool candidate_started;
    bool recovery_active;
    bool candidate_ended;
    bool data_valid;
    bool baseline_ready;
    float baseline;
    float residual;
    float peak_residual;
    V5CandidateThresholds thresholds;
} V5CandidateOutput;

typedef struct {
    V5CandidateConfig config;
    float baseline[V5_CANDIDATE_BASELINE_SAMPLES];
    uint32_t baseline_head, baseline_count;
    float residual_clear[V5_CANDIDATE_HISTORY_SAMPLES];
    float dpdt_clear[V5_CANDIDATE_HISTORY_SAMPLES];
    uint32_t history_head, history_count;
    float scratch[V5_CANDIDATE_HISTORY_SAMPLES];
    uint32_t sample_index, next_event_id, active_event_id;
    bool in_candidate, have_previous_pressure;
    float previous_pressure;
    bool have_previous_residual;
    float previous_residual;
    uint32_t trough_index, peak_index, start_index;
    float trough_residual, peak_residual;
    uint32_t main_count, possible_count, recovery_count;
    V5CandidateThresholds main_thresholds, explore_thresholds;
} V5CandidateState;

void V5Candidate_Init(V5CandidateState *state, const V5CandidateConfig *config);
void V5Candidate_BeginCycle(V5CandidateState *state);
V5CandidateOutput V5Candidate_Step(V5CandidateState *state,
                                   float pressure_mm_hg,
                                   bool signal_valid);

#endif
