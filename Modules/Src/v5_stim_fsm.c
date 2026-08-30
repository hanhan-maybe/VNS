#include "v5_stim_fsm.h"
#include "v5_runtime.h"
#include <stddef.h>
#include <string.h>

void V5StimFsm_Init(V5StimFsm *s, float max_stim_s, float refractory_s)
{
    if (s == NULL) return;
    memset(s, 0, sizeof(*s));
    s->max_active_ticks = (uint32_t)(max_stim_s * (float)V5_FS_HZ + 0.5f);
    s->refractory_limit_ticks = (uint32_t)(refractory_s * (float)V5_FS_HZ + 0.5f);
}

void V5StimFsm_Fault(V5StimFsm *s)
{
    if (s == NULL) return;
    s->state = V5_STIM_FAULT;
    s->output_on = false;
}

void V5StimFsm_ClearFault(V5StimFsm *s)
{
    if (s == NULL) return;
    s->state = V5_STIM_IDLE;
    s->output_on = false;
    s->active_ticks = 0u;
    s->refractory_ticks = 0u;
}

bool V5StimFsm_Step(V5StimFsm *s,
                    bool trigger,
                    bool allow_stim,
                    bool signal_valid,
                    bool candidate_active)
{
    if (s == NULL) return false;

    if (!signal_valid) {
        V5StimFsm_Fault(s);
        return false;
    }
    if (!allow_stim) {
        s->output_on = false;
        return false;
    }

    switch (s->state) {
    case V5_STIM_IDLE:
        s->output_on = false;
        if (trigger) {
            s->state = V5_STIM_ACTIVE;
            s->active_ticks = 0u;
            s->output_on = true;
        }
        break;

    case V5_STIM_ACTIVE:
        s->output_on = true;
        if (++s->active_ticks >= s->max_active_ticks || !candidate_active) {
            s->output_on = false;
            s->state = V5_STIM_REFRACTORY;
            s->refractory_ticks = 0u;
        }
        break;

    case V5_STIM_REFRACTORY:
        s->output_on = false;
        if (++s->refractory_ticks >= s->refractory_limit_ticks) {
            s->state = V5_STIM_IDLE;
        }
        break;

    default:
        s->output_on = false;
        break;
    }
    return s->output_on;
}
