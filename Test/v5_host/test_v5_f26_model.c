#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include "v5_model.h"
#include "v5_model_stxf26.h"
#include "f26_parity_vectors.h"

#define SCORE_TOLERANCE 1e-5f
#define THRESHOLD_TOLERANCE 1e-6f

int main(void)
{
    size_t i, score_failures = 0, threshold_failures = 0;
    size_t positive_mismatches = 0;
    float max_error = 0.0f, sum_error = 0.0f;
    if (!V5Model_ValidateConfig(&g_v5_model_stxf26)) return 2;
    printf("rows tested = %lu\n", (unsigned long)g_f26_parity_count);
    for (i = 0; i < g_f26_parity_count; ++i) {
        const F26ParityRow *row = &g_f26_parity_rows[i];
        V5ModelOutput out = V5Model_Infer(&g_v5_model_stxf26, row->x);
        float error;
        if (!out.valid) return 3;
        error = fabsf(out.probability - row->expected_score);
        sum_error += error;
        if (error > max_error) max_error = error;
        if (error > SCORE_TOLERANCE) score_failures++;
        if (fabsf(g_v5_model_stxf26.threshold - row->expected_threshold) >
            THRESHOLD_TOLERANCE) threshold_failures++;
        if ((out.positive ? 1u : 0u) != row->expected_positive)
            positive_mismatches++;
    }
    printf("max score error      = %.9g\n", max_error);
    printf("mean score error     = %.9g\n", sum_error / (float)g_f26_parity_count);
    printf("score failures       = %lu\n", (unsigned long)score_failures);
    printf("threshold failures   = %lu\n", (unsigned long)threshold_failures);
    printf("positive mismatches  = %lu\n", (unsigned long)positive_mismatches);
    if (score_failures || threshold_failures || positive_mismatches) return 1;
    puts("PASS_F26_MODEL_NUMERICAL_PARITY");
    return 0;
}
