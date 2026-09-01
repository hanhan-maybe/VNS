#ifndef V5_APP_H
#define V5_APP_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "v5_candidate.h"
#include "v5_features.h"
#include "v5_runtime.h"
#include "v5_stim_fsm.h"
#include "v5_subject_config.h"

#ifndef V5_ALLOW_PHYSICAL_STIMULATION
#define V5_ALLOW_PHYSICAL_STIMULATION 0
#endif

typedef struct {
    V5CandidateOutput candidate;
    V5FeatureOutput features;
    V5RuntimeOutput runtime;
    V5StimState stim_state;
    bool stim_output_on;
    bool config_valid;
} V5AppOutput;

bool AppV5_Init(const uint8_t *subject_config, size_t length,
                float max_stim_s, float refractory_s);
void AppV5_BeginCycle(void);
V5AppOutput AppV5_On100Hz(float pressure_mm_hg, bool pressure_signal_valid);
void AppV5_SetShadowMode(bool shadow_mode);
bool AppV5_EnableStimulation(bool enable);
void AppV5_EmergencyStop(void);
void AppV5_ReportTimingFault(void);
void AppV5_ReportWatchdogFault(void);

#endif
