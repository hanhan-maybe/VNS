#ifndef V5_FEATURES_H
#define V5_FEATURES_H

#include <stdbool.h>
#include <stdint.h>

#include "v5_model.h"

#define V5_FEATURE_PRESSURE_RING 2701u
#define V5_FEATURE_BASELINE_SAMPLES 2500u
#define V5_FEATURE_SPECTRAL_SAMPLES 500u

typedef struct {
    float pressure[V5_FEATURE_PRESSURE_RING];
    uint8_t valid[V5_FEATURE_PRESSURE_RING];
    uint32_t head, count, next_sample_index;
    bool baseline_ready;
    double baseline_median, baseline_scale;
    double baseline_low_power, baseline_wide_power;
    double work[V5_FEATURE_BASELINE_SAMPLES];
    double psd[V5_FEATURE_BASELINE_SAMPLES / 2u + 1u];
} V5FeatureState;

typedef struct {
    bool available;
    float values[V5_MODEL_FEATURE_COUNT];
} V5FeatureOutput;

void V5Features_Init(V5FeatureState *state);
void V5Features_BeginCycle(V5FeatureState *state);
void V5Features_PushPressure(V5FeatureState *state, float pressure, bool valid);
V5FeatureOutput V5Features_Compute(V5FeatureState *state);

#endif
