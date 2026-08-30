
/**
 * @file    FreeRTOSConfig.h
 * @brief   FreeRTOS 配置 for STM32N657X0 (Cortex-M55)
 */
#ifndef FREERTOS_CONFIG_H
#define FREERTOS_CONFIG_H

#include "stm32n6xx_hal.h"

/* ------------------------------------------------------------------ */
/*  STM32CubeIDE 集成                                                 */
/* ------------------------------------------------------------------ */
#define configUSE_PREEMPTION                    1
#define configUSE_IDLE_HOOK                     0
#define configUSE_TICK_HOOK                     0
#define configCPU_CLOCK_HZ                       SystemCoreClock
#define configTICK_RATE_HZ                      ((TickType_t)1000)
#define configMAX_PRIORITIES                    ( 7 )
#define configMINIMAL_STACK_SIZE                ((uint16_t)128)
#define configTOTAL_HEAP_SIZE                   ((size_t)(64 * 1024))
#define configMAX_TASK_NAME_LEN                 ( 16 )
#define configUSE_TRACE_FACILITY                1
#define configUSE_16_BIT_TICKS                  0
#define configIDLE_SHOULD_YIELD                 1
#define configUSE_MUTEXES                       1
#define configQUEUE_REGISTRY_SIZE               8
#define configCHECK_FOR_STACK_OVERFLOW          2
#define configUSE_RECURSIVE_MUTEXES             1
#define configUSE_MALLOC_FAILED_HOOK            1
#define configUSE_APPLICATION_TASK_TAG          0
#define configUSE_COUNTING_SEMAPHORES           1
#define configUSE_QUEUE_SETS                    0

/* ------------------------------------------------------------------ */
/*  ARMv8-M (Cortex-M55) 相关                                         */
/* ------------------------------------------------------------------ */
#define configENABLE_TRUSTZONE                  0
#define configENABLE_FPU                        1
#define configENABLE_MVE                        1   /* Helium MVE 支持 */
#define configRUN_FREERTOS_SECURE_ONLY          1
#define configTASK_RETURN_ADDRESS               NULL
#define configENABLE_MPU                        0

/* ------------------------------------------------------------------ */
/*  IP 调用接口                                                      */
/* ------------------------------------------------------------------ */
#define configUSE_PORT_OPTIMISED_TASK_SELECTION 1

/* ------------------------------------------------------------------ */
/*  钩子函数                                                         */
/* ------------------------------------------------------------------ */
#define configUSE_STATS_FORMATTING_FUNCTIONS     0
#define configSUPPORT_DYNAMIC_ALLOCATION         1
#define configSUPPORT_STATIC_ALLOCATION          0

/* ------------------------------------------------------------------ */
/*  FreeRTOS API 映射                                                 */
/* ------------------------------------------------------------------ */
#define vPortSVCHandler     SVC_Handler
#define xPortPendSVHandler  PendSV_Handler
#define xPortSysTickHandler SysTick_Handler

#endif /* FREERTOS_CONFIG_H */
