
/**
 * @file    logger.h
 * @brief   串口日志系统
 *
 * 所有 I/O、错误、分类事件、刺激事件均通过此模块记录。
 * 日志级别可通过配置文件运行时调整。
 * 后端通过注入回调与具体 UART / USB 实现解耦。
 */
#ifndef LOGGER_H
#define LOGGER_H

#include "vns_types.h"
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  后端写回调                                                        */
/* ------------------------------------------------------------------ */
typedef void (*Logger_WriteFn)(const char *buf, uint32_t len, void *context);

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief  初始化日志系统
 * @param  write_fn  输出回调 (如 UART 发送函数)
 * @param  context   回调上下文
 */
void Logger_Init(Logger_WriteFn write_fn, void *context);

/** @brief  设置运行日志级别 (覆盖 config 中的设置) */
void Logger_SetLevel(VNS_LogLevel level);

/** @brief  获取当前日志级别 */
VNS_LogLevel Logger_GetLevel(void);

/**
 * @brief  写格式化日志
 * @param  level  本消息级别
 * @param  tag    模块标签 (如 "FEAT", "CLASS")
 * @param  fmt    格式化字符串
 */
void Logger_Log(VNS_LogLevel level, const char *tag,
                const char *fmt, ...) __attribute__((format(printf,3,4)));

/** @brief  vprintf 变体 */
void Logger_LogV(VNS_LogLevel level, const char *tag,
                 const char *fmt, va_list ap);

/** @brief  快速写原始字符串 (无格式, 无级别检查) */
void Logger_WriteRaw(const char *str);

/* ------------------------------------------------------------------ */
/*  便捷宏                                                            */
/* ------------------------------------------------------------------ */
#define LOGE(tag, fmt, ...) \
    Logger_Log(VNS_LOG_ERROR, tag, fmt, ##__VA_ARGS__)
#define LOGW(tag, fmt, ...) \
    Logger_Log(VNS_LOG_WARN,  tag, fmt, ##__VA_ARGS__)
#define LOGI(tag, fmt, ...) \
    Logger_Log(VNS_LOG_INFO,  tag, fmt, ##__VA_ARGS__)
#define LOGD(tag, fmt, ...) \
    Logger_Log(VNS_LOG_DEBUG, tag, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* LOGGER_H */
