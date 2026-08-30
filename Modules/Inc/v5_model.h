#ifndef V5_MODEL_H
#define V5_MODEL_H

#include <stdbool.h>
#include <stdint.h>

#define V5_MODEL_FEATURE_COUNT 15u

typedef struct {
    float mean[V5_MODEL_FEATURE_COUNT];
    float scale[V5_MODEL_FEATURE_COUNT];
    float coef[V5_MODEL_FEATURE_COUNT];
    float intercept;
    float threshold;
} V5ModelConfig;

typedef struct {
    float logit;
    float probability;
    bool positive;
    bool valid;
} V5ModelOutput;

bool V5Model_ValidateConfig(const V5ModelConfig *cfg);
V5ModelOutput V5Model_Infer(const V5ModelConfig *cfg,
                            const float features[V5_MODEL_FEATURE_COUNT]);

#endif
