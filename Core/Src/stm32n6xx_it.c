
/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
#include "stm32n6xx_it.h"
#include "main.h"

/* ================================================================== */
/*  Cortex-M55 ??????                                               */
/* ================================================================== */

void NMI_Handler(void)
{
    while (1) { __NOP(); }
}

void HardFault_Handler(void)
{
    while (1) { __NOP(); }
}

void MemManage_Handler(void)
{
    while (1) { __NOP(); }
}

void BusFault_Handler(void)
{
    while (1) { __NOP(); }
}

void UsageFault_Handler(void)
{
    while (1) { __NOP(); }
}

void SVC_Handler(void)
{
    /* FreeRTOS ??? */
}

void DebugMon_Handler(void)
{
    /* ?????*/
}

void PendSV_Handler(void)
{
    /* FreeRTOS ??? */
}

void SysTick_Handler(void)
{
    /* FreeRTOS tick */
    if (xTaskGetSchedulerState() != taskSCHEDULER_NOT_STARTED) {
        xPortSysTickHandler();
    }
    HAL_IncTick();
}

/* ================================================================== */
/*  ??????                                                          */
/* ================================================================== */

/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
void TIM6_IRQHandler(void)
{
    HAL_TIM_IRQHandler(&htim6);
}

/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
void USART1_IRQHandler(void)
{
    /* HAL ?????? HAL_UART_RxCpltCallback ?????*/
    HAL_UART_IRQHandler(&huart1);
}

/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
void OTG_FS_IRQHandler(void)
{
    HAL_PCD_IRQHandler(&hpcd_USB_OTG_FS);
}

/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
void DMA1_Channel1_IRQHandler(void)
{
    HAL_DMA_IRQHandler(&hdma_usart1_rx);
}

/* ================================================================== */
/*  HAL ????????(??main.c ?????                                  */
/* ================================================================== */

/* HAL_TIM_PeriodElapsedCallback ?????main.c ??*/
/* HAL_PCD_DataRxStageCallback ?????main.c ??*/

/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        /* ??? USART1 ?????????, ??? ring buffer */
        /* ????????? ??????????USB VCP */
    }
}

/** 
 * @file    stm32n6xx_it.c
 * @brief   NUCLEO-N657X0-Q interrupt service routines
 *
 * ISR policy: only the following operations are permitted:
 *   - DMA / USB peripheral status handling
 *   - Ring buffer write (RB_WriteByte)
 *   - Task notification (vTaskNotifyGiveFromISR)
 * Algorithm computation, printf, and HAL_Delay are strictly forbidden.
 */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    (void)huart;
    /* ?????? ????? Logger ??ring buffer ??? */
    /* ???: Logger_WriteRaw ???????????ISR ?????*/
}


/* ================================================================== */
/*  UART4 interrupt ? acquisition data from PC                        */
/* ================================================================== */
void UART4_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart_acq);
}

void DMA1_Channel3_IRQHandler(void)
{
    HAL_DMA_IRQHandler(huart_acq.hdmarx);
}

