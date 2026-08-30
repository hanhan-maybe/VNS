/**
 * @file    system_manager.h
 * @brief   VNS system state machine and power-on self-test
 *
 * States: BOOT -> SELF_TEST -> WAIT_DATA -> RUNNING <-> DATA_TIMEOUT
 *                                          ↕
 *                                      -> FAULT (from anywhere)
 */
#ifndef SYSTEM_MANAGER_H
#define SYSTEM_MANAGER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  System states                                                     */
/* ------------------------------------------------------------------ */
#define SY_BOOT          0
#define SY_SELF_TEST     1
#define SY_WAIT_DATA     2
#define SY_RUNNING       3
#define SY_DATA_TIMEOUT  4
#define SY_FAULT         5

/* ------------------------------------------------------------------ */
/*  Self-test result bits                                             */
/* ------------------------------------------------------------------ */
#define SY_TEST_RAM     0x0001u
#define SY_TEST_CFG_CRC 0x0002u
#define SY_TEST_UART    0x0004u
#define SY_TEST_STIM_LO 0x0008u
#define SY_TEST_ESTOP   0x0010u
#define SY_TEST_TASKS   0x0020u
#define SY_TEST_WDT     0x0040u
#define SY_TEST_ALL     0x007Fu

/* ------------------------------------------------------------------ */
/*  Callbacks for hardware-dependent self-tests                       */
/* ------------------------------------------------------------------ */
typedef bool (*SY_CheckFn)(void *ctx);

/* ------------------------------------------------------------------ */
/*  System manager                                                    */
/* ------------------------------------------------------------------ */
typedef struct {
    uint8_t  state;
    uint8_t  previous_state;
    uint16_t test_results;
    uint32_t boot_tick_ms;
    uint32_t entry_tick_ms;
    uint32_t last_valid_data_tick;

    /* Callbacks (set in SY_Init, overridable for testing) */
    SY_CheckFn check_stim_low;
    void      *stim_low_ctx;
    SY_CheckFn check_estop;
    void      *estop_ctx;
} SystemManager;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/** @brief  Initialise system manager, set state to SY_BOOT */
void SY_Init(SystemManager *sm, uint32_t boot_tick_ms);

/** @brief  Run all self-tests; sets test_results bitmask */
void SY_RunSelfTest(SystemManager *sm);

/** @brief  True if all mandatory tests passed */
bool SY_AllTestsPassed(const SystemManager *sm);

/** @brief  Transition to new state (logs change in sm->previous_state) */
void SY_Transition(SystemManager *sm, uint8_t new_state);

/** @brief  Human-readable state name (for logging) */
const char* SY_StateName(uint8_t state);

/** @brief  Get bitmask of failed tests */
uint16_t SY_FailedTests(const SystemManager *sm);

#ifdef __cplusplus
}
#endif

#endif /* SYSTEM_MANAGER_H */
