/* Integration example only. Do not use old Modules implementation. */
#include "v5_runtime.h"
#include "v5_stim_fsm.h"

static V5Runtime g_v5_runtime;
static V5StimFsm g_v5_stim;

void AppV5_Init(const V5ModelConfig *model)
{
    V5Runtime_Init(&g_v5_runtime);
    (void)V5Runtime_LoadModel(&g_v5_runtime, model);

    /* First live deployment must remain shadow mode. */
    V5Runtime_SetShadowMode(&g_v5_runtime, true);
    V5Runtime_EnableStimulation(&g_v5_runtime, false);

    V5StimFsm_Init(&g_v5_stim, 5.0f, 15.0f);
}

void AppV5_On100Hz(V5CandidateInput candidate,
                    const float p_early_features[V5_MODEL_FEATURE_COUNT],
                    bool feature_available,
                    bool pressure_signal_valid)
{
    V5RuntimeOutput out = V5Runtime_Step(
        &g_v5_runtime,
        candidate,
        p_early_features,
        feature_available
    );

    bool stim_on = V5StimFsm_Step(
        &g_v5_stim,
        out.stimulation_request,
        g_v5_runtime.stimulation_enabled && !g_v5_runtime.shadow_mode,
        pressure_signal_valid,
        candidate.candidate_active
    );

    (void)stim_on;
    /* Next: send out.* through UART for Python-vs-C parity audit. */
    /* Only after shadow audit: map stim_on to validated VNS hardware driver. */
}
