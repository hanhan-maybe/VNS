#ifndef V5_STIM_FSM_H
#define V5_STIM_FSM_H

#include <stdbool.h>
#include <stdint.h>

typedef enum {
    V5_STIM_IDLE = 0,
    V5_STIM_ACTIVE,
    V5_STIM_REFRACTORY,
    V5_STIM_FAULT
} V5StimState;

typedef struct {
    V5StimState state;
    uint32_t active_ticks;
    uint32_t refractory_ticks;
    uint32_t max_active_ticks;
    uint32_t refractory_limit_ticks;
    bool output_on;
} V5StimFsm;

void V5StimFsm_Init(V5StimFsm *s, float max_stim_s, float refractory_s);
void V5StimFsm_Fault(V5StimFsm *s);
void V5StimFsm_ClearFault(V5StimFsm *s);
bool V5StimFsm_Step(V5StimFsm *s,
                    bool trigger,
                    bool allow_stim,
                    bool signal_valid,
                    bool candidate_active);

#endif
