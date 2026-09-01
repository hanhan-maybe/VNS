#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "v5_model_stxf26.h"
#include "v5_model_stxf37.h"
#include "v5_runtime.h"

#define MAX_LINE 16384
#define MAX_COLS 96
#define SCORE_TOLERANCE 1e-5f

static const char *const feature_names[V5_MODEL_FEATURE_COUNT] = {
    "p_current_delta", "p_peak_delta", "p_threshold_above_duration",
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt",
    "p_positive_dpdt_occupancy", "p_auc", "p_auc_growth",
    "pressure_curvature", "peak_to_current_drop", "p_trailing_variability_1s",
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
    "pressure_spectral_entropy"
};

static size_t split_csv(char *line, char **cols, size_t capacity)
{
    size_t count = 0;
    char *p = line;
    if (capacity == 0) return 0;
    cols[count++] = p;
    while (*p != '\0' && count < capacity) {
        if (*p == ',') {
            *p = '\0';
            cols[count++] = p + 1;
        } else if (*p == '\r' || *p == '\n') {
            *p = '\0';
            break;
        }
        ++p;
    }
    return count;
}

static int find_col(char **header, size_t count, const char *name)
{
    size_t i;
    for (i = 0; i < count; ++i) if (strcmp(header[i], name) == 0) return (int)i;
    fprintf(stderr, "missing CSV column: %s\n", name);
    return -1;
}

static int as_bool(const char *value)
{
    return strcmp(value, "True") == 0 || strcmp(value, "true") == 0 ||
           strcmp(value, "1") == 0;
}

static uint32_t event_id(const char *value)
{
    uint32_t hash = 2166136261u;
    if (value == NULL || *value == '\0') return 0u;
    while (*value != '\0') {
        hash ^= (uint8_t)*value++;
        hash *= 16777619u;
    }
    return hash == 0u ? 1u : hash;
}

int main(int argc, char **argv)
{
    FILE *handle;
    char line[MAX_LINE], *cols[MAX_COLS], *header[MAX_COLS];
    size_t ncols, i, rows = 0, score_mismatch = 0, positive_mismatch = 0;
    size_t state_mismatch = 0, trigger_mismatch = 0, event_mismatch = 0;
    size_t duplicate_triggers = 0;
    int animal_col, cycle_col, decision_col, active_col, event_col;
    int ended_col, available_col, score_col, positive_col, state_col, trigger_col;
    int fcols[V5_MODEL_FEATURE_COUNT];
    char prior_animal[32] = "", prior_cycle[32] = "";
    V5Runtime runtime;
    uint32_t last_trigger_event = 0u;
    float max_score_error = 0.0f;

    if (argc != 2) {
        fprintf(stderr, "usage: %s m1_full_cycle_replay.csv\n", argv[0]);
        return 2;
    }
    handle = fopen(argv[1], "rb");
    if (handle == NULL || fgets(line, sizeof(line), handle) == NULL) return 2;
    ncols = split_csv(line, header, MAX_COLS);
#define COL(name) find_col(header, ncols, name)
    animal_col = COL("animal"); cycle_col = COL("cycle_id");
    decision_col = COL("decision_index"); active_col = COL("candidate_active");
    event_col = COL("candidate_event_id"); ended_col = COL("candidate_event_end");
    available_col = COL("feature_available"); score_col = COL("score");
    positive_col = COL("score_positive"); state_col = COL("t0_state");
    trigger_col = COL("t0_trigger");
    for (i = 0; i < V5_MODEL_FEATURE_COUNT; ++i) fcols[i] = COL(feature_names[i]);
    if (trigger_col < 0) return 2;

    V5Runtime_Init(&runtime);
    while (fgets(line, sizeof(line), handle) != NULL) {
        V5CandidateInput candidate = {0};
        V5RuntimeOutput output;
        float features[V5_MODEL_FEATURE_COUNT] = {0};
        float expected_score = 0.0f;
        int available, expected_positive, expected_state, expected_trigger;
        uint32_t eid;
        ncols = split_csv(line, cols, MAX_COLS);
        if ((size_t)trigger_col >= ncols) continue;
        /* The CSV also contains exact event/latency audit probes.  Runtime
         * inference is registered only on the frozen 0.25 s grid. */
        if ((strtoul(cols[decision_col], NULL, 10) % V5_UPDATE_SAMPLES) != 0u)
            continue;
        if (strcmp(prior_animal, cols[animal_col]) != 0 ||
            strcmp(prior_cycle, cols[cycle_col]) != 0) {
            V5Runtime_Init(&runtime);
            if (strcmp(cols[animal_col], "STxF37") == 0)
                V5Runtime_LoadModel(&runtime, &g_v5_model_stxf37);
            else if (strcmp(cols[animal_col], "STxF26") == 0)
                V5Runtime_LoadModel(&runtime, &g_v5_model_stxf26);
            else continue;
            strncpy(prior_animal, cols[animal_col], sizeof(prior_animal) - 1u);
            strncpy(prior_cycle, cols[cycle_col], sizeof(prior_cycle) - 1u);
            last_trigger_event = 0u;
        }
        eid = event_id(cols[event_col]);
        candidate.candidate_active = as_bool(cols[active_col]);
        candidate.candidate_event_id = eid;
        candidate.candidate_ended = as_bool(cols[ended_col]);
        candidate.recovery_event = false;
        available = as_bool(cols[available_col]);
        if (available) {
            for (i = 0; i < V5_MODEL_FEATURE_COUNT; ++i)
                features[i] = strtof(cols[fcols[i]], NULL);
            expected_score = strtof(cols[score_col], NULL);
        }
        expected_positive = candidate.candidate_active && as_bool(cols[positive_col]);
        expected_state = as_bool(cols[state_col]);
        expected_trigger = as_bool(cols[trigger_col]);
        runtime.sample_index = (uint32_t)strtoul(cols[decision_col], NULL, 10);
        output = V5Runtime_Step(&runtime, candidate, features, available != 0);
        if (available && candidate.candidate_active) {
            float error = fabsf(output.score - expected_score);
            if (error > max_score_error) max_score_error = error;
            if (error > SCORE_TOLERANCE) score_mismatch++;
        }
        if ((int)output.score_positive != expected_positive) positive_mismatch++;
        if ((int)(candidate.candidate_active && output.score_positive) != expected_state)
            state_mismatch++;
        if ((int)output.t0_trigger != expected_trigger) trigger_mismatch++;
        if (output.t0_trigger && output.candidate_event_id != eid) event_mismatch++;
        if (output.t0_trigger && last_trigger_event == eid) duplicate_triggers++;
        if (output.t0_trigger) last_trigger_event = eid;
        if (candidate.candidate_ended || !candidate.candidate_active) last_trigger_event = 0u;
        rows++;
    }
    fclose(handle);
    printf("rows tested                 = %lu\n", (unsigned long)rows);
    printf("max score error             = %.9g\n", max_score_error);
    printf("score mismatch              = %lu\n", (unsigned long)score_mismatch);
    printf("score_positive mismatch     = %lu\n", (unsigned long)positive_mismatch);
    printf("T0 state mismatch           = %lu\n", (unsigned long)state_mismatch);
    printf("T0 trigger mismatch         = %lu\n", (unsigned long)trigger_mismatch);
    printf("event attribution mismatch  = %lu\n", (unsigned long)event_mismatch);
    printf("duplicate event triggers    = %lu\n", (unsigned long)duplicate_triggers);
    if (score_mismatch || positive_mismatch || state_mismatch || trigger_mismatch ||
        event_mismatch || duplicate_triggers) return 1;
    puts("PASS_F37_T0_RUNTIME_PARITY");
    puts("PASS_F26_T0_RUNTIME_PARITY");
    return 0;
}
