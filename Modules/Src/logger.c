
/**
 * @file    logger.c
 * @brief   串口日志系统实现
 */
#include "logger.h"
#include "vns_config.h"
#include <stdio.h>
#include <string.h>

/* 日志前缀字符串 */
static const char *s_level_prefix[] = {
    [VNS_LOG_ERROR] = "[ERR]",
    [VNS_LOG_WARN]  = "[WRN]",
    [VNS_LOG_INFO]  = "[INF]",
    [VNS_LOG_DEBUG] = "[DBG]",
};

/* 内部状态 */
static Logger_WriteFn s_write_fn = 0;
static void          *s_context  = 0;
static VNS_LogLevel   s_level    = VNS_LOG_INFO;

/* ------------------------------------------------------------------ */

void Logger_Init(Logger_WriteFn write_fn, void *context)
{
    s_write_fn = write_fn;
    s_context  = context;

    /* 从配置读取初始级别 */
    const VNS_Config *cfg = VNS_ConfigGetDefault();
    s_level = cfg->log.level;
}

void Logger_SetLevel(VNS_LogLevel level)
{
    if (level >= VNS_LOG_OFF && level <= VNS_LOG_DEBUG)
        s_level = level;
}

VNS_LogLevel Logger_GetLevel(void)
{
    return s_level;
}

/* ------------------------------------------------------------------ */
/*  内部输出                                                          */
/* ------------------------------------------------------------------ */
static void output(const char *buf, uint32_t len)
{
    if (s_write_fn && buf && len)
        s_write_fn(buf, len, s_context);
}

/* ------------------------------------------------------------------ */

void Logger_LogV(VNS_LogLevel level, const char *tag,
                 const char *fmt, va_list ap)
{
    if (level == VNS_LOG_OFF || level > s_level)
        return;
    if (!fmt) return;

    char  buf[256];
    int   pos = 0;
    int   rem = (int)sizeof(buf) - 2;  /* 预留换行 + 结尾 */

    /* 前缀: [LVL][tag] */
    if (tag && level < sizeof(s_level_prefix)/sizeof(s_level_prefix[0])
        && s_level_prefix[level]) {
        pos += snprintf(buf + pos, rem - pos, "%s[%s] ",
                        s_level_prefix[level], tag);
    }

    /* 正文 */
    if (pos < rem) {
        pos += vsnprintf(buf + pos, rem - pos, fmt, ap);
    }

    /* 换行 */
    if (pos < (int)sizeof(buf) - 1) {
        buf[pos++] = '\n';
        buf[pos]   = '\0';
    }

    output(buf, pos);
}

void Logger_Log(VNS_LogLevel level, const char *tag,
                const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    Logger_LogV(level, tag, fmt, ap);
    va_end(ap);
}

void Logger_WriteRaw(const char *str)
{
    if (str) output(str, strlen(str));
}
