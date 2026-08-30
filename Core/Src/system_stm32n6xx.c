
/**
 * @file    system_stm32n6xx.c
 * @brief   STM32N657X0 系统初始化
 *
 * 最小启动配置。
 * 完整版本由 CubeMX 生成, 此处仅提供 SystemCoreClockUpdate 和
 * SystemInit 的基本实现。
 */
#include "stm32n6xx.h"

/** 系统时钟频率 (Hz) — 由 SystemClock_Config 更新 */
uint32_t SystemCoreClock = 600000000UL;

/** 外部振荡器频率 */
const uint32_t HSIRhFrequency    = 64000000UL;
const uint32_t HSE_RTC_Frequency = 32768UL;

/**
 * @brief  初始化 FPU/MVE 设置, 向量表偏移
 */
void SystemInit(void)
{
    /* FPU 已在启动文件中使能 */

    /* 设置向量表偏移 */
#if defined(SCB_BASE) && defined(__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
    SCB->VTOR = FLASH_BASE;
#endif
}

/**
 * @brief  更新 SystemCoreClock 全局变量
 *
 * 实际实现需读取 RCC 寄存器计算时钟。
 * 简化版本返回默认值。
 */
void SystemCoreClockUpdate(void)
{
    /* 此处应实现时钟树解析。
     * 默认值已在 main.c 中通过 SystemClock_Config 设置。
     */
    SystemCoreClock = 600000000UL;
}
