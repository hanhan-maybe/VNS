/**
 * @file    stim_control.c
 * @brief   Stimulation control state machine implementation
 *
 * Independent of the classifier. Takes classification results as input.
 * State transitions use non-blocking time comparisons.
 */
#include "stim_control.h"
#include "classifier.h"
#include <string.h>

/* Internal reason codes */
#define REASON_ARM_CMD      1
#define REASON_DISARM_CMD   2
#define REASON_CLEAR_FAULT  3
#define REASON_CLASS1_CFM   4
#define REASON_CLASS2_BLOCK 5
#define REASON_PULSE_DONE   6
#define REASON_REFR_DONE    7
#define REASON_DATA_TIMEOUT 8
#define REASON_CLASS1_LOST  9
#define REASON_RATE_LIMIT   10

/* ------------------------------------------------------------------ */
/*  Default config                                                    */
/* ------------------------------------------------------------------ */
const StimConfig_t* SC_GetDefaultConfig(void)
{
    static const StimConfig_t s_cfg = {
        .class1_confirm_ms      = 500u,
        .trigger_pulse_ms       = 100u,
        .refractory_ms          = 10000u,
        .max_triggers_per_minute = 3u,
        .data_timeout_ms        = 500u,
        .require_manual_arm     = 1u,
    };
    return &s_cfg;
}

/* ------------------------------------------------------------------ */
/*  Internal helpers                                                  */
/* ------------------------------------------------------------------ */
static void enter_state(StimControl *sc, uint8_t to_state,
                        uint8_t ev_type, uint16_t reason)
{
    sc->previous_state = sc->state;
    sc->state          = to_state;

    sc->last_event.timestamp_us = (uint64_t)sc->get_tick_ms(sc->tick_ctx) * 1000ULL;
    sc->last_event.sequence++;
    sc->last_event.event_type  = ev_type;
    sc->last_event.from_state  = sc->previous_state;
    sc->last_event.to_state    = to_state;
    sc->last_event.reason_code = reason;
}

/* ------------------------------------------------------------------ */
/*  Init                                                              */
/* ------------------------------------------------------------------ */
void SC_Init(StimControl *sc, const StimConfig_t *cfg,
             SC_GPIO_WriteFn gpio_write, void *gpio_ctx,
             SC_GetTickMsFn get_tick, void *tick_ctx)
{
    if (!sc) return;
    memset(sc, 0, sizeof(*sc));
    sc->cfg       = cfg ? cfg : SC_GetDefaultConfig();
    sc->gpio_write = gpio_write;
    sc->gpio_ctx  = gpio_ctx;
    sc->get_tick_ms = get_tick;
    sc->tick_ctx  = tick_ctx;
    sc->state     = STIM_DISABLED;
    if (sc->gpio_write) sc->gpio_write(sc, 0);
}

/* ------------------------------------------------------------------ */
/*  Rate limiting                                                     */
/* ------------------------------------------------------------------ */
static bool check_rate_limit(StimControl *sc, uint32_t now_ms)
{
    uint32_t window = 60000u;
    uint32_t count = 0;
    for (uint8_t i = 0; i < sc->trigger_history_count; i++) {
        uint8_t idx = (sc->trigger_history_head - 1 - i) & 0x0F; /* mask */
        if ((now_ms - sc->trigger_timestamps[idx]) < window) count++;
    }
    return count < sc->cfg->max_triggers_per_minute;
}

static void record_trigger(StimControl *sc, uint32_t now_ms)
{
    sc->trigger_timestamps[sc->trigger_history_head] = now_ms;
    sc->trigger_history_head = (sc->trigger_history_head + 1) & 0x0F;
    if (sc->trigger_history_count < 10) sc->trigger_history_count++;
}

/* ------------------------------------------------------------------ */
/*  Main process ??call at 10??00 Hz                                 */
/* ------------------------------------------------------------------ */
void SC_Process(StimControl *sc, uint8_t class_id, uint8_t data_valid,
                uint32_t tick_ms)
{
    if (!sc) return;

    uint32_t now = sc->get_tick_ms ? sc->get_tick_ms(sc->tick_ctx) : tick_ms;

    /* ---- 1. Fault detection ---- */
    bool fresh_fault = (!data_valid || class_id == CLASS_INVALID);
    if (fresh_fault) {
        sc->fault_flags.data_timeout = true;
    }

    bool any_fault = sc->fault_flags.data_timeout || sc->fault_flags.estop ||
                     sc->fault_flags.comm_error  || sc->fault_flags.param_error ||
                     sc->fault_flags.system_fault;

    if (any_fault) {
        if (sc->state != STIM_FAULT) {
            if (sc->gpio_write) sc->gpio_write(sc, 0);
            enter_state(sc, STIM_FAULT, STIM_EVENT_FAULT, REASON_DATA_TIMEOUT);
        }
        return;
    }

    /* ---- 2. Class2 inhibition ---- */
    if (class_id == CLASS2_VOIDING) {
        if (sc->state == STIM_PENDING || sc->state == STIM_ACTIVE ||
            sc->state == STIM_REFRACTORY) {
            if (sc->gpio_write) sc->gpio_write(sc, 0);
            enter_state(sc, STIM_ARMED, STIM_EVENT_STATE_CHANGE, REASON_CLASS2_BLOCK);
        }
        return;
    }

    /* ---- 3. State machine ---- */
    switch (sc->state) {

    case STIM_DISABLED:
    case STIM_FAULT:
        break;

    case STIM_ARMED:
        if (class_id == CLASS1_UNSTABLE) {
            sc->pending_start_ms = now;
            enter_state(sc, STIM_PENDING, STIM_EVENT_STATE_CHANGE, REASON_CLASS1_CFM);
        }
        break;

    case STIM_PENDING:
        if (class_id == CLASS1_UNSTABLE) {
            if ((now - sc->pending_start_ms) >= sc->cfg->class1_confirm_ms) {
                if (check_rate_limit(sc, now)) {
                    sc->trigger_start_ms = now;
                    if (sc->gpio_write) sc->gpio_write(sc, 1);
                    sc->total_triggers++;
                    sc->last_event.event_type = STIM_EVENT_TRIGGER;
                    sc->last_event.reason_code = REASON_CLASS1_CFM;
                    record_trigger(sc, now);
                    enter_state(sc, STIM_ACTIVE, STIM_EVENT_TRIGGER, REASON_CLASS1_CFM);
                } else {
                    enter_state(sc, STIM_ARMED, STIM_EVENT_STATE_CHANGE, REASON_RATE_LIMIT);
                }
            }
        } else {
            enter_state(sc, STIM_ARMED, STIM_EVENT_STATE_CHANGE, REASON_CLASS1_LOST);
        }
        break;

    case STIM_ACTIVE:
        if ((now - sc->trigger_start_ms) >= sc->cfg->trigger_pulse_ms) {
            if (sc->gpio_write) sc->gpio_write(sc, 0);
            sc->refractory_start_ms = now;
            enter_state(sc, STIM_REFRACTORY, STIM_EVENT_STATE_CHANGE, REASON_PULSE_DONE);
        }
        break;

    case STIM_REFRACTORY:
        if ((now - sc->refractory_start_ms) >= sc->cfg->refractory_ms) {
            enter_state(sc, STIM_ARMED, STIM_EVENT_STATE_CHANGE, REASON_REFR_DONE);
        }
        break;
    }
}

/* ------------------------------------------------------------------ */
/*  Commands                                                          */
/* ------------------------------------------------------------------ */
int SC_Command(StimControl *sc, StimCommand_t cmd,
               int32_t param_id, float param_value)
{
    if (!sc) return -1;
    (void)param_id;
    (void)param_value;

    switch (cmd) {
    case STIM_CMD_ARM:
        if (sc->state == STIM_DISABLED || sc->state == STIM_ARMED) {
            enter_state(sc, STIM_ARMED, STIM_EVENT_COMMAND, REASON_ARM_CMD);
        }
        return 0;

    case STIM_CMD_DISARM:
        if (sc->state != STIM_DISABLED && sc->state != STIM_FAULT) {
            if (sc->gpio_write) sc->gpio_write(sc, 0);
            enter_state(sc, STIM_DISABLED, STIM_EVENT_COMMAND, REASON_DISARM_CMD);
        }
        return 0;

    case STIM_CMD_CLEAR_FAULT:
        if (sc->state == STIM_FAULT) {
            memset(&sc->fault_flags, 0, sizeof(sc->fault_flags));
            if (sc->gpio_write) sc->gpio_write(sc, 0);
            enter_state(sc, STIM_DISABLED, STIM_EVENT_COMMAND, REASON_CLEAR_FAULT);
        }
        return 0;

    case STIM_CMD_GET_STATUS:
        return sc->state;

    case STIM_CMD_SET_PARAM:
        return 0;   /* RAM-only for now */

    case STIM_CMD_SAVE_PARAM:
        return -1;  /* not implemented */
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Queries                                                           */
/* ------------------------------------------------------------------ */
uint8_t   SC_GetState(const StimControl *sc)     { return sc ? sc->state : STIM_FAULT; }
uint32_t  SC_GetTriggerCount(const StimControl *sc) { return sc ? sc->total_triggers : 0; }

uint16_t SC_GetFaultFlags(const StimControl *sc)
{
    if (!sc) return 0;
    uint16_t f = 0;
    if (sc->fault_flags.data_timeout)  f |= 0x01;
    if (sc->fault_flags.estop)          f |= 0x02;
    if (sc->fault_flags.comm_error)    f |= 0x04;
    if (sc->fault_flags.param_error)   f |= 0x08;
    if (sc->fault_flags.system_fault)  f |= 0x10;
    return f;
}

void SC_GetLastEvent(const StimControl *sc, StimEvent_t *ev)
{
    if (sc && ev) *ev = sc->last_event;
}

