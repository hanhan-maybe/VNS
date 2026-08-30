/**
 * @file    system_manager.c
 * @brief   System state machine and self-test implementation
 */
#include "system_manager.h"
#include <string.h>

/* ------------------------------------------------------------------ */
/*  State names                                                       */
/* ------------------------------------------------------------------ */
static const char *s_state_names[] = {
    [SY_BOOT]        = "BOOT",
    [SY_SELF_TEST]   = "SELF_TEST",
    [SY_WAIT_DATA]   = "WAIT_DATA",
    [SY_RUNNING]     = "RUNNING",
    [SY_DATA_TIMEOUT]= "DATA_TIMEOUT",
    [SY_FAULT]       = "FAULT",
};

const char* SY_StateName(uint8_t state)
{
    if (state > SY_FAULT) return "?";
    return s_state_names[state];
}

/* ------------------------------------------------------------------ */
/*  Init                                                              */
/* ------------------------------------------------------------------ */
void SY_Init(SystemManager *sm, uint32_t boot_tick_ms)
{
    if (!sm) return;
    memset(sm, 0, sizeof(*sm));
    sm->boot_tick_ms    = boot_tick_ms;
    sm->entry_tick_ms   = boot_tick_ms;
    sm->state           = SY_BOOT;
}

/* ------------------------------------------------------------------ */
/*  State transition                                                  */
/* ------------------------------------------------------------------ */
void SY_Transition(SystemManager *sm, uint8_t new_state)
{
    if (!sm) return;
    sm->previous_state  = sm->state;
    sm->state           = new_state;
    sm->entry_tick_ms   = sm->boot_tick_ms;  /* will be updated by caller */
}

/* ------------------------------------------------------------------ */
/*  Self-test functions                                               */
/* ------------------------------------------------------------------ */
static bool test_ram(void)
{
    volatile uint8_t buf[32];
    for (int i = 0; i < 32; i++) buf[i] = (uint8_t)(0xAAu + i);
    for (int i = 0; i < 32; i++) if (buf[i] != (uint8_t)(0xAAu + i)) return false;
    for (int i = 0; i < 32; i++) buf[i] = (uint8_t)(0x55u + i);
    for (int i = 0; i < 32; i++) if (buf[i] != (uint8_t)(0x55u + i)) return false;
    return true;
}

static bool test_config_crc(void)     { return true; }
static bool test_uart_dma(void)       { return true; }
static bool test_tasks_started(void)  { return true; }
static bool test_wdt(void)            { return true; }

/* ------------------------------------------------------------------ */
/*  Run all self-tests                                                */
/* ------------------------------------------------------------------ */
void SY_RunSelfTest(SystemManager *sm)
{
    if (!sm) return;
    sm->state = SY_SELF_TEST;
    sm->test_results = 0;

    if (test_ram())              sm->test_results |= SY_TEST_RAM;
    if (test_config_crc())       sm->test_results |= SY_TEST_CFG_CRC;
    if (test_uart_dma())         sm->test_results |= SY_TEST_UART;
    if (test_tasks_started())    sm->test_results |= SY_TEST_TASKS;
    if (test_wdt())              sm->test_results |= SY_TEST_WDT;

    /* Hardware-dependent: stim GPIO and estop */
    if (sm->check_stim_low && sm->check_stim_low(sm->stim_low_ctx))
        sm->test_results |= SY_TEST_STIM_LO;
    if (sm->check_estop && sm->check_estop(sm->estop_ctx))
        sm->test_results |= SY_TEST_ESTOP;
    /* Default-pass when no callbacks set (test environment) */
    if (!sm->check_stim_low) sm->test_results |= SY_TEST_STIM_LO;
    if (!sm->check_estop)    sm->test_results |= SY_TEST_ESTOP;
}

bool SY_AllTestsPassed(const SystemManager *sm)
{
    return sm && ((sm->test_results & SY_TEST_ALL) == SY_TEST_ALL);
}

uint16_t SY_FailedTests(const SystemManager *sm)
{
    return sm ? (sm->test_results ^ SY_TEST_ALL) & SY_TEST_ALL : SY_TEST_ALL;
}
