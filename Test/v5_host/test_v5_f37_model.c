#include <stdio.h>
#include <math.h>
#include <stddef.h>

#include "v5_model.h"
#include "v5_model_stxf37.h"

#include "f37_parity_vectors.h"


#define SCORE_TOLERANCE 1e-5f
#define THRESHOLD_TOLERANCE 1e-6f


int main(void)
{
    size_t i;

    float max_error = 0.0f;
    float sum_error = 0.0f;

    size_t score_failures = 0;
    size_t threshold_failures = 0;
    size_t positive_mismatches = 0;

    size_t worst_index = 0;


    printf(
        "========================================\n"
    );

    printf(
        "F37 FULL Python -> C MODEL PARITY TEST\n"
    );

    printf(
        "========================================\n"
    );


    /*
     * Step 1:
     * validate frozen F37 model
     */

    if (!V5Model_ValidateConfig(
            &g_v5_model_stxf37))
    {
        printf(
            "FAIL: frozen F37 model invalid\n"
        );

        return 1;
    }


    printf(
        "rows tested = %zu\n",
        g_f37_parity_count
    );


    /*
     * Step 2:
     * run every valid Python P-EARLY row
     * through C inference
     */

    for (i = 0;
         i < g_f37_parity_count;
         ++i)
    {
        const F37ParityRow *r =
            &g_f37_parity_rows[i];


        V5ModelOutput out =
            V5Model_Infer(
                &g_v5_model_stxf37,
                r->x
            );


        if (!out.valid)
        {
            printf(
                "Invalid inference at row %zu\n",
                i
            );

            return 1;
        }


        /*
         * score parity
         */

        float error =
            fabsf(
                out.probability -
                r->expected_score
            );


        sum_error += error;


        if (error > max_error)
        {
            max_error = error;
            worst_index = i;
        }


        if (error > SCORE_TOLERANCE)
        {
            score_failures++;
        }


        /*
         * threshold parity
         */

        if (fabsf(
                g_v5_model_stxf37.threshold -
                r->expected_threshold
            ) > THRESHOLD_TOLERANCE)
        {
            threshold_failures++;
        }


        /*
         * classification parity
         */

        uint8_t c_positive =
            out.positive ? 1u : 0u;


        if (c_positive !=
            r->expected_positive)
        {
            positive_mismatches++;
        }
    }


    /*
     * Step 3:
     * statistics
     */

    float mean_error =
        g_f37_parity_count > 0
        ?
        sum_error /
        (float)g_f37_parity_count
        :
        0.0f;


    printf("\n");

    printf(
        "max score error      = %.9g\n",
        max_error
    );

    printf(
        "mean score error     = %.9g\n",
        mean_error
    );

    printf(
        "score failures       = %zu\n",
        score_failures
    );

    printf(
        "threshold failures   = %zu\n",
        threshold_failures
    );

    printf(
        "positive mismatches  = %zu\n",
        positive_mismatches
    );


    /*
     * Show worst row for debugging.
     */

    if (g_f37_parity_count > 0)
    {
        const F37ParityRow *worst =
            &g_f37_parity_rows[worst_index];


        V5ModelOutput out =
            V5Model_Infer(
                &g_v5_model_stxf37,
                worst->x
            );


        printf("\n");

        printf(
            "worst row:\n"
        );

        printf(
            "  cycle          = %s\n",
            worst->cycle_id
        );

        printf(
            "  decision_index = %d\n",
            worst->decision_index
        );

        printf(
            "  Python score   = %.9f\n",
            worst->expected_score
        );

        printf(
            "  C score        = %.9f\n",
            out.probability
        );
    }


    /*
     * Step 4:
     * deployment gate
     */

    if (score_failures != 0)
    {
        printf(
            "\nFAIL: score parity\n"
        );

        return 1;
    }


    if (threshold_failures != 0)
    {
        printf(
            "\nFAIL: threshold parity\n"
        );

        return 1;
    }


    if (positive_mismatches != 0)
    {
        printf(
            "\nFAIL: classification parity\n"
        );

        return 1;
    }


    printf(
        "\nPASS: F37 full Python/C model parity\n"
    );


    return 0;
}
