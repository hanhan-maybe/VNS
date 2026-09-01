/* Board-neutral V5 integration. HAL transport/output mapping stays outside. */
#include "v5_app.h"

#include <string.h>

static V5SubjectConfig g_subject;
static V5CandidateState g_candidate;
static V5FeatureState g_features;
static V5Runtime g_runtime;
static V5StimFsm g_stim;
static bool g_config_valid;
static float g_max_stim_s;
static float g_refractory_s;

bool AppV5_Init(const uint8_t *subject_config, size_t length,
                float max_stim_s, float refractory_s)
{
    V5CandidateConfig candidate_config;
    g_config_valid = false;
    V5Runtime_Init(&g_runtime);
    V5Features_Init(&g_features);
    V5StimFsm_Init(&g_stim, max_stim_s, refractory_s);
    if (max_stim_s <= 0.0f || refractory_s < 0.0f ||
        !V5SubjectConfig_Decode(&g_subject, subject_config, length) ||
        !V5SubjectConfig_ApplyCandidate(&g_subject, &candidate_config) ||
        !V5Runtime_LoadSubjectConfig(&g_runtime, &g_subject)) {
        V5StimFsm_Fault(&g_stim);
        return false;
    }
    g_max_stim_s = max_stim_s;
    g_refractory_s = refractory_s;
    V5Candidate_Init(&g_candidate, &candidate_config);
    V5Runtime_SetShadowMode(&g_runtime, true);
    V5Runtime_EnableStimulation(&g_runtime, false);
    g_config_valid = true;
    return true;
}

void AppV5_BeginCycle(void)
{
    if (!g_config_valid) return;
    V5Candidate_BeginCycle(&g_candidate);
    V5Features_BeginCycle(&g_features);
    V5Runtime_Init(&g_runtime);
    (void)V5Runtime_LoadSubjectConfig(&g_runtime, &g_subject);
    V5Runtime_SetShadowMode(&g_runtime, true);
    V5Runtime_EnableStimulation(&g_runtime, false);
    V5StimFsm_Init(&g_stim, g_max_stim_s, g_refractory_s);
}

V5AppOutput AppV5_On100Hz(float pressure_mm_hg, bool pressure_signal_valid)
{
    V5AppOutput output;
    V5CandidateInput input;
    memset(&output, 0, sizeof(output));
    output.config_valid = g_config_valid;
    if (!g_config_valid) return output;
    output.candidate = V5Candidate_Step(&g_candidate, pressure_mm_hg,
                                        pressure_signal_valid);
    V5Features_PushPressure(&g_features, pressure_mm_hg, pressure_signal_valid);
    output.features = V5Features_Compute(&g_features);
    input.candidate_active = output.candidate.candidate_active;
    input.candidate_event_id = output.candidate.candidate_event_id;
    input.recovery_event = false;
    input.candidate_ended = output.candidate.candidate_ended;
    output.runtime = V5Runtime_Step(&g_runtime, input, output.features.values,
                                    output.features.available);
    output.stim_output_on = V5StimFsm_Step(
        &g_stim, output.runtime.stimulation_request,
        g_runtime.stimulation_enabled && !g_runtime.shadow_mode,
        pressure_signal_valid && g_config_valid,
        output.candidate.candidate_active);
    output.stim_state = g_stim.state;
    return output;
}

void AppV5_SetShadowMode(bool shadow_mode)
{
    V5Runtime_SetShadowMode(&g_runtime, shadow_mode);
}

bool AppV5_EnableStimulation(bool enable)
{
#if !V5_ALLOW_PHYSICAL_STIMULATION
    (void)enable;
    V5Runtime_EnableStimulation(&g_runtime, false);
    return false;
#else
    if (!g_config_valid || g_stim.state == V5_STIM_FAULT) return false;
    V5Runtime_EnableStimulation(&g_runtime, enable);
    return g_runtime.stimulation_enabled;
#endif
}

void AppV5_EmergencyStop(void) { V5StimFsm_Fault(&g_stim); }
void AppV5_ReportTimingFault(void) { V5StimFsm_Fault(&g_stim); }
void AppV5_ReportWatchdogFault(void) { V5StimFsm_Fault(&g_stim); }
