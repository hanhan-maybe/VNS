/**
 * @file    stim_control.h
 * @brief   Independent stimulation control state machine
 *
 * State machine: DISABLED → ARMED → PENDING → ACTIVE → REFRACTORY → ARMED
 * All states can enter FAULT on error conditions.
 * Class1 triggers after confirmation period; Class2 inhibits immediately.
 */
#ifndef STIM_CONTROL_H
#define STIM_CONTROL_H

#include "vns_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  State IDs                                                         */
/* ------------------------------------------------------------------ */
#define STIM_DISABLED     0
#define STIM_ARMED        1
#define STIM_PENDING      2
#define STIM_ACTIVE       3
#define STIM_REFRACTORY   4
#define STIM_FAULT        5

/* ------------------------------------------------------------------ */
/*  Command IDs                                                       */
/* ------------------------------------------------------------------ */
typedef enum {
    STIM_CMD_ARM,
    STIM_CMD_DISARM,
    STIM_CMD_CLEAR_FAULT,
    STIM_CMD_GET_STATUS,
    STIM_CMD_SET_PARAM,
    STIM_CMD_SAVE_PARAM,
} StimCommand_t;

/* ------------------------------------------------------------------ */
/*  Event types and reasons                                           */
/* ------------------------------------------------------------------ */
#define STIM_EVENT_STATE_CHANGE  0
#define STIM_EVENT_TRIGGER       1
#define STIM_EVENT_FAULT         2
#define STIM_EVENT_COMMAND       3

typedef struct {
    uint64_t timestamp_us;
    uint8_t  event_type;
    uint8_t  from_state;
    uint8_t  to_state;
    uint16_t reason_code;
    uint32_t sequence;
} StimEvent_t;

/* ------------------------------------------------------------------ */
/*  Fault flags                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    bool data_timeout;
    bool estop;
    bool comm_error;
    bool param_error;
    bool system_fault;
} StimFaultFlags_t;

/* ------------------------------------------------------------------ */
/*  Configuration                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t class1_confirm_ms;         /**< Class1 must persist this long */
    uint32_t trigger_pulse_ms;          /**< GPIO pulse width              */
    uint32_t refractory_ms;             /**< Rest period after trigger     */
    uint32_t max_triggers_per_minute;   /**< Rate limit                    */
    uint32_t data_timeout_ms;           /**< No-data timeout               */
    uint8_t  require_manual_arm;        /**< Require ARM command           */
} StimConfig_t;

/* ------------------------------------------------------------------ */
/*  Main control block                                                */
/* ------------------------------------------------------------------ */
typedef struct StimControl StimControl;

/* Callbacks */
typedef void     (*SC_GPIO_WriteFn)(StimControl *sc, uint8_t state);
typedef uint32_t (*SC_GetTickMsFn)(void *ctx);

struct StimControl {
    const StimConfig_t *cfg;

    /* GPIO + timing */
    SC_GPIO_WriteFn  gpio_write;
    void            *gpio_ctx;
    SC_GetTickMsFn   get_tick_ms;
    void            *tick_ctx;

    /* State */
    uint8_t  state;
    uint8_t  previous_state;

    /* Timing */
    uint32_t pending_start_ms;
    uint32_t trigger_start_ms;
    uint32_t refractory_start_ms;

    /* Fault */
    StimFaultFlags_t fault_flags;

    /* Rate limiting */
    uint32_t trigger_timestamps[10];
    uint8_t  trigger_history_count;
    uint8_t  trigger_history_head;
    uint32_t total_triggers;

    /* Last event log */
    StimEvent_t last_event;
};

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/** @brief  Default configuration */
const StimConfig_t* SC_GetDefaultConfig(void);

/** @brief  Initialise the stimulation control block */
void SC_Init(StimControl *sc, const StimConfig_t *cfg,
             SC_GPIO_WriteFn gpio_write, void *gpio_ctx,
             SC_GetTickMsFn get_tick, void *tick_ctx);

/** @brief  Process one evaluation cycle (call at 10–100 Hz) */
void SC_Process(StimControl *sc, uint8_t class_id, uint8_t data_valid,
                uint32_t tick_ms);

/** @brief  Issue a command (ARM, DISARM, CLEAR_FAULT, etc.) */
int  SC_Command(StimControl *sc, StimCommand_t cmd,
                int32_t param_id, float param_value);

/** @brief  Query current state */
uint8_t SC_GetState(const StimControl *sc);

/** @brief  Get last event log entry by copy */
void    SC_GetLastEvent(const StimControl *sc, StimEvent_t *ev);

/** @brief  Get total trigger count since init */
uint32_t SC_GetTriggerCount(const StimControl *sc);

/** @brief  Get current fault flags */
uint16_t SC_GetFaultFlags(const StimControl *sc);

#ifdef __cplusplus
}
#endif

#endif /* STIM_CONTROL_H */
