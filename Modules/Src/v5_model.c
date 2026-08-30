#include "v5_model.h"
#include <math.h>
#include <stddef.h>

static bool v5_finite(float x) { return isfinite((double)x) != 0; }

bool V5Model_ValidateConfig(const V5ModelConfig *cfg)
{
    uint32_t i;
    if (cfg == NULL) return false;
    if (!v5_finite(cfg->intercept) || !v5_finite(cfg->threshold)) return false;
    if (!(cfg->threshold > 0.0f && cfg->threshold < 1.0f)) return false;
    for (i = 0; i < V5_MODEL_FEATURE_COUNT; ++i) {
        if (!v5_finite(cfg->mean[i]) || !v5_finite(cfg->scale[i]) || !v5_finite(cfg->coef[i])) return false;
        if (!(cfg->scale[i] > 0.0f)) return false;
    }
    return true;
}

V5ModelOutput V5Model_Infer(const V5ModelConfig *cfg,
                            const float features[V5_MODEL_FEATURE_COUNT])
{
    V5ModelOutput out = {0};
    float logit;
    uint32_t i;

    if (!V5Model_ValidateConfig(cfg) || features == NULL) return out;

    logit = cfg->intercept;
    for (i = 0; i < V5_MODEL_FEATURE_COUNT; ++i) {
        float z;
        if (!v5_finite(features[i])) return out;
        z = (features[i] - cfg->mean[i]) / cfg->scale[i];
        logit += cfg->coef[i] * z;
    }

    out.logit = logit;
    if (logit >= 0.0f) {
        float e = expf(-logit);
        out.probability = 1.0f / (1.0f + e);
    } else {
        float e = expf(logit);
        out.probability = e / (1.0f + e);
    }
    out.positive = out.probability >= cfg->threshold;
    out.valid = true;
    return out;
}
