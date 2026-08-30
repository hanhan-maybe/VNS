
/**
 * @file    stm32n6xx_it.h
 * @brief   NUCLEO-N657X0-Q 中断声明
 */
#ifndef STM32N6XX_IT_H
#define STM32N6XX_IT_H

#include "stm32n6xx_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

void NMI_Handler(void);
void HardFault_Handler(void);
void MemManage_Handler(void);
void BusFault_Handler(void);
void UsageFault_Handler(void);
void SVC_Handler(void);
void DebugMon_Handler(void);
void PendSV_Handler(void);
void SysTick_Handler(void);

/* 外设中断 (按实际使能添加) */
void DMA1_Channel1_IRQHandler(void);
void TIM6_IRQHandler(void);
void USART1_IRQHandler(void);
void OTG_FS_IRQHandler(void);

#ifdef __cplusplus
}
#endif

#endif /* STM32N6XX_IT_H */
