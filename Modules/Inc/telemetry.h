/**
 * @file    telemetry.h
 * @brief   Non-blocking telemetry output module
 *
 * Encodes SAMPLE/FEATURE/EVENT frames into priority ring buffers.
 * LoggerTask reads frames via TL_ReadFrame and sends via UART DMA.
 * High-priority frames (EVENT) use a dedicated buffer so they are
 * never dropped by lower-priority SAMPLE frames.
 */
#ifndef TELEMETRY_H
#define TELEMETRY_H

#include "vns_types.h"
#include "ring_buffer.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Frame type IDs                                                    */
/* ------------------------------------------------------------------ */
#define TL_TYPE_SAMPLE   0x01
#define TL_TYPE_FEATURE  0x02
#define TL_TYPE_EVENT    0x03
#define TL_SOF           0xA5u

/* ------------------------------------------------------------------ */
/*  Configuration                                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    bool     enable_sample_output;   /**< Set false to suppress high-freq samp */
    uint32_t sample_divider;         /**< Output 1 in every N TL_PushSample   */
    uint32_t feature_divider;        /**< Output 1 in every N TL_PushFeature   */
} TelemetryConfig_t;

/* ------------------------------------------------------------------ */
/*  Runtime statistics                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t sample_count;
    uint32_t feature_count;
    uint32_t event_count;
    uint32_t frame_drops;            /**< SAMPLE frames dropped (buf full)    */
    uint32_t queue_high_water;       /**< Max event-buffer occupancy (bytes)  */
    uint32_t crc_errors;             /**< CRC failures on read                */
} TelemetryStats_t;

/* ------------------------------------------------------------------ */
/*  Notification callback (optional, to wake LoggerTask)              */
/* ------------------------------------------------------------------ */
typedef void (*TL_NotifyFn)(void *ctx);

/* ------------------------------------------------------------------ */
/*  Telemetry context                                                 */
/* ------------------------------------------------------------------ */
typedef struct {
    TelemetryConfig_t cfg;

    /* Priority buffers: SAMPLE (low), FEATURE+EVENT (high) */
    uint8_t     sample_storage[1024];
    RingBuffer  rb_sample;
    uint8_t     event_storage[512];
    RingBuffer  rb_event;

    uint32_t    sample_counter;
    uint32_t    feature_counter;
    TelemetryStats_t stats;
    TL_NotifyFn notify_fn;
    void       *notify_ctx;
} TelemetryContext;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/** @brief  Default configuration */
const TelemetryConfig_t* TL_GetDefaultConfig(void);

/** @brief  Initialise telemetry */
void TL_Init(TelemetryContext *ctx, const TelemetryConfig_t *cfg);

/** @brief  Push a SAMPLE frame (low priority; dropped when buffer full) */
void TL_PushSample(TelemetryContext *ctx, uint64_t ts,
                   float pressure, float eus, uint16_t quality);

/** @brief  Push a FEATURE frame (high priority) */
void TL_PushFeature(TelemetryContext *ctx, uint64_t ts,
                    float delta_p, float max_dpdt, float energy,
                    float slope, float rise_dur, float rise_ratio,
                    uint8_t eus_bursting, uint8_t burst_count);

/** @brief  Push an EVENT frame (high priority) */
void TL_PushEvent(TelemetryContext *ctx, uint64_t ts,
                  uint8_t class_id, uint16_t reason_flags,
                  uint8_t stim_state, uint8_t stim_trigger,
                  uint16_t fault_flags, uint32_t seq_drops);

/**
 * @brief  Read next complete telemetry frame (called by LoggerTask)
 * @param  buf      Output buffer
 * @param  buf_size Buffer capacity
 * @return Number of bytes written to buf, or 0 if no frame available
 */
uint32_t TL_ReadFrame(TelemetryContext *ctx, uint8_t *buf, uint32_t buf_size);

/** @brief  Register wake-up callback */
void TL_SetNotify(TelemetryContext *ctx, TL_NotifyFn fn, void *fn_ctx);

/** @brief  Get statistics */
const TelemetryStats_t* TL_GetStats(const TelemetryContext *ctx);

#ifdef __cplusplus
}
#endif

#endif /* TELEMETRY_H */
