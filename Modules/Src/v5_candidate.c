#include "v5_candidate.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define FS_HZ 100.0f
#define UPDATE_SAMPLES 25u
#define CONFIRM_SAMPLES 50u
#define RECOVERY_SAMPLES 100u
#define HISTORY_READY_SAMPLES 2500u
#define PRESSURE_MIN (-50.0f)
#define PRESSURE_MAX 100.0f
#define JUMP_LIMIT 50.0f
#define MAD_FACTOR 1.4826f
#define SIGMA_EPS 1e-6f

static int compare_float(const void *a, const void *b)
{
    const float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

static float quantile(V5CandidateState *state, const float *ring,
                      uint32_t head, uint32_t count, float q)
{
    uint32_t i;
    float position, fraction;
    uint32_t low, high;
    if (count == 0u) return NAN;
    for (i = 0u; i < count; ++i) {
        uint32_t index = count < V5_CANDIDATE_HISTORY_SAMPLES
            ? i : (head + i) % V5_CANDIDATE_HISTORY_SAMPLES;
        state->scratch[i] = ring[index];
    }
    qsort(state->scratch, count, sizeof(float), compare_float);
    position = (float)(count - 1u) * q;
    low = (uint32_t)floorf(position);
    high = (uint32_t)ceilf(position);
    fraction = position - (float)low;
    return state->scratch[low] + fraction * (state->scratch[high] - state->scratch[low]);
}

static float median_array(float *scratch, const float *values, uint32_t count)
{
    uint32_t i;
    if (count == 0u) return NAN;
    for (i = 0u; i < count; ++i) scratch[i] = values[i];
    qsort(scratch, count, sizeof(float), compare_float);
    if ((count & 1u) != 0u) return scratch[count / 2u];
    return 0.5f * (scratch[count / 2u - 1u] + scratch[count / 2u]);
}

static float ring_mad(V5CandidateState *state, const float *ring,
                      uint32_t head, uint32_t count)
{
    uint32_t i;
    float med;
    if (count == 0u) return NAN;
    for (i = 0u; i < count; ++i) {
        uint32_t index = count < V5_CANDIDATE_HISTORY_SAMPLES
            ? i : (head + i) % V5_CANDIDATE_HISTORY_SAMPLES;
        state->scratch[i] = ring[index];
    }
    qsort(state->scratch, count, sizeof(float), compare_float);
    med = (count & 1u) ? state->scratch[count / 2u]
                       : 0.5f * (state->scratch[count / 2u - 1u] + state->scratch[count / 2u]);
    for (i = 0u; i < count; ++i) state->scratch[i] = fabsf(state->scratch[i] - med);
    qsort(state->scratch, count, sizeof(float), compare_float);
    return MAD_FACTOR * ((count & 1u) ? state->scratch[count / 2u]
                       : 0.5f * (state->scratch[count / 2u - 1u] + state->scratch[count / 2u]));
}

static void update_thresholds(V5CandidateState *state, bool exploratory,
                              V5CandidateThresholds *out)
{
    const bool warmup = state->history_count < HISTORY_READY_SAMPLES;
    const float multiplier = exploratory ? 3.0f : 4.0f;
    float sigma, sigma_dpdt, q, confirm;
    if (warmup) {
        sigma = fmaxf(state->config.prior_sigma_p, SIGMA_EPS);
        sigma_dpdt = fmaxf(state->config.prior_sigma_dpdt, SIGMA_EPS);
        q = multiplier * sigma;
    } else {
        sigma = fmaxf(ring_mad(state, state->residual_clear, state->history_head,
                               state->history_count), SIGMA_EPS);
        q = quantile(state, state->residual_clear, state->history_head,
                     state->history_count, exploratory ? 0.99f : 0.995f);
        sigma_dpdt = fmaxf(ring_mad(state, state->dpdt_clear, state->history_head,
                                    state->history_count), SIGMA_EPS);
    }
    confirm = fmaxf(multiplier * sigma, q);
    confirm = fminf(3.68f, fmaxf(1.47f, confirm));
    out->sigma_p = sigma;
    out->sigma_dpdt = sigma_dpdt;
    out->confirm = confirm;
    out->start = 0.60f * confirm;
    out->recovery = 0.40f * confirm;
    out->warmup = warmup;
}

static void append_baseline(V5CandidateState *state, float value)
{
    state->baseline[state->baseline_head] = value;
    state->baseline_head = (state->baseline_head + 1u) % V5_CANDIDATE_BASELINE_SAMPLES;
    if (state->baseline_count < V5_CANDIDATE_BASELINE_SAMPLES) state->baseline_count++;
}

static float baseline_median(V5CandidateState *state)
{
    uint32_t i;
    for (i = 0u; i < state->baseline_count; ++i) state->scratch[i] = state->baseline[i];
    return median_array(state->scratch + state->baseline_count,
                        state->scratch, state->baseline_count);
}

static void append_history(V5CandidateState *state, float residual, float dpdt)
{
    state->residual_clear[state->history_head] = residual;
    state->dpdt_clear[state->history_head] = dpdt;
    state->history_head = (state->history_head + 1u) % V5_CANDIDATE_HISTORY_SAMPLES;
    if (state->history_count < V5_CANDIDATE_HISTORY_SAMPLES) state->history_count++;
}

void V5Candidate_Init(V5CandidateState *state, const V5CandidateConfig *config)
{
    if (state == NULL || config == NULL) return;
    memset(state, 0, sizeof(*state));
    state->config = *config;
    V5Candidate_BeginCycle(state);
}

void V5Candidate_BeginCycle(V5CandidateState *state)
{
    if (state == NULL) return;
    state->baseline_head = state->baseline_count = 0u;
    state->sample_index = state->next_event_id = state->active_event_id = 0u;
    state->in_candidate = state->have_previous_pressure = false;
    state->have_previous_residual = false;
    state->trough_index = state->peak_index = state->start_index = 0u;
    state->trough_residual = state->peak_residual = 0.0f;
    state->main_count = state->possible_count = state->recovery_count = 0u;
    update_thresholds(state, false, &state->main_thresholds);
    update_thresholds(state, true, &state->explore_thresholds);
}

V5CandidateOutput V5Candidate_Step(V5CandidateState *state,
                                   float pressure_mm_hg, bool signal_valid)
{
    V5CandidateOutput out = {0};
    bool valid;
    float base, residual, rise, prominence, fall, recovery_limit;
    if (state == NULL) return out;
    valid = signal_valid && isfinite(pressure_mm_hg) &&
            pressure_mm_hg >= PRESSURE_MIN && pressure_mm_hg <= PRESSURE_MAX;
    if (state->have_previous_pressure &&
        fabsf(pressure_mm_hg - state->previous_pressure) * FS_HZ > JUMP_LIMIT)
        valid = false;
    state->previous_pressure = pressure_mm_hg;
    state->have_previous_pressure = true;
    if ((state->sample_index % UPDATE_SAMPLES) == 0u) {
        update_thresholds(state, false, &state->main_thresholds);
        update_thresholds(state, true, &state->explore_thresholds);
    }
    out.thresholds = state->main_thresholds;
    out.data_valid = valid;
    if (!valid) {
        if (state->in_candidate) {
            out.candidate_ended = true;
            out.candidate_event_id = state->active_event_id;
        }
        state->in_candidate = false;
        state->active_event_id = 0u;
        if (state->have_previous_residual) {
            state->trough_index = state->sample_index > 0u ? state->sample_index - 1u : 0u;
            state->trough_residual = state->previous_residual;
        }
        state->main_count = state->possible_count = state->recovery_count = 0u;
        state->have_previous_residual = false;
        state->sample_index++;
        return out;
    }
    if (state->baseline_count < V5_CANDIDATE_BASELINE_SAMPLES) {
        append_baseline(state, pressure_mm_hg);
        state->sample_index++;
        return out;
    }
    base = baseline_median(state);
    residual = pressure_mm_hg - base;
    out.baseline_ready = true;
    out.baseline = base;
    out.residual = residual;
    if (!state->in_candidate) {
        if (state->trough_index == 0u || residual <= state->trough_residual) {
            state->trough_index = state->sample_index;
            state->trough_residual = residual;
        }
        rise = residual - state->trough_residual;
        if (rise > state->explore_thresholds.start) {
            state->in_candidate = true;
            state->start_index = state->sample_index;
            state->peak_index = state->sample_index;
            state->peak_residual = residual;
            state->active_event_id = ++state->next_event_id;
            out.candidate_started = true;
        } else {
            float previous = state->have_previous_residual
                ? state->previous_residual : residual;
            append_history(state, residual, (residual - previous) * FS_HZ);
            append_baseline(state, pressure_mm_hg);
        }
    } else {
        if (residual > state->peak_residual) {
            state->peak_residual = residual;
            state->peak_index = state->sample_index;
        }
        prominence = residual - state->trough_residual;
        if (state->possible_count < CONFIRM_SAMPLES) {
            state->possible_count = prominence > state->explore_thresholds.confirm
                ? state->possible_count + 1u : 0u;
        }
        if (state->main_count < CONFIRM_SAMPLES) {
            state->main_count = prominence > state->main_thresholds.confirm
                ? state->main_count + 1u : 0u;
        }
        prominence = state->peak_residual - state->trough_residual;
        fall = state->peak_residual - residual;
        recovery_limit = state->trough_residual +
            fmaxf(state->main_thresholds.recovery, 0.40f * prominence);
        if (fall >= 0.60f * prominence && residual <= recovery_limit)
            state->recovery_count++;
        else
            state->recovery_count = 0u;
        if (state->recovery_count >= RECOVERY_SAMPLES) {
            out.candidate_ended = true;
            out.candidate_event_id = state->active_event_id;
            state->in_candidate = false;
            state->active_event_id = 0u;
            state->trough_index = state->sample_index;
            state->trough_residual = residual;
            state->main_count = state->possible_count = state->recovery_count = 0u;
        }
    }
    if (state->in_candidate) {
        out.candidate_active = true;
        out.candidate_event_id = state->active_event_id;
        out.recovery_active = state->recovery_count > 0u;
        out.peak_residual = state->peak_residual;
    }
    state->previous_residual = residual;
    state->have_previous_residual = true;
    state->sample_index++;
    return out;
}
