
/**
 * @file    stm32n6xx_hal_conf.h
 * @brief   HAL 配置模板 — NUCLEO-N657X0-Q
 *
 * 此文件由 CubeMX 生成; 这里提供最小配置。
 * 初次使用请在 CubeMX 中打开 VNS_N6.ioc 并重新生成。
 */
#ifndef STM32N6XX_HAL_CONF_H
#define STM32N6XX_HAL_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  模块使能                                                          */
/* ------------------------------------------------------------------ */
#define HAL_MODULE_ENABLED
#define HAL_ADC_MODULE_ENABLED
#define HAL_CORTEX_MODULE_ENABLED
#define HAL_DMA_MODULE_ENABLED
#define HAL_EXTI_MODULE_ENABLED
#define HAL_FLASH_MODULE_ENABLED
#define HAL_GPIO_MODULE_ENABLED
#define HAL_HSEM_MODULE_ENABLED
#define HAL_I2C_MODULE_ENABLED
#define HAL_IRDA_MODULE_ENABLED
#define HAL_IWDG_MODULE_ENABLED
#define HAL_PCD_MODULE_ENABLED
#define HAL_PWR_MODULE_ENABLED
#define HAL_RCC_MODULE_ENABLED
#define HAL_RTC_MODULE_ENABLED
#define HAL_SMARTCARD_MODULE_ENABLED
#define HAL_SPI_MODULE_ENABLED
#define HAL_TIM_MODULE_ENABLED
#define HAL_UART_MODULE_ENABLED
#define HAL_USART_MODULE_ENABLED
#define HAL_WWDG_MODULE_ENABLED

/* ------------------------------------------------------------------ */
/*  系统参数                                                          */
/* ------------------------------------------------------------------ */
#define HAL_NVIC_PRIO_BITS         5U
#define HAL_SYSTICK_FREQ           1000U
#define USE_HAL_ADC_REGISTER_CALLBACKS     0U
#define USE_HAL_PCD_REGISTER_CALLBACKS     0U
#define USE_HAL_TIM_REGISTER_CALLBACKS     0U
#define USE_HAL_UART_REGISTER_CALLBACKS    0U

/* ------------------------------------------------------------------ */
/*  HAL 时基                                                         */
/* ------------------------------------------------------------------ */
#define HAL_TICK_FREQ              1000U
#define HSE_VALUE                  24000000U   /* NUCLEO-N657X0-Q */
#define HSI_VALUE                  64000000U
#define LSE_VALUE                  32768U
#define LSI_VALUE                  32000U

/* ------------------------------------------------------------------ */
/*  外设断言宏                                                       */
/* ------------------------------------------------------------------ */
#ifndef assert_param
#define assert_param(expr) ((void)0U)
#endif

#ifdef __cplusplus
}
#endif

#endif /* STM32N6XX_HAL_CONF_H */
