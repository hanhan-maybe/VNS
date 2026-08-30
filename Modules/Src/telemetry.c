/**
 * @file    telemetry.c
 * @brief   Non-blocking telemetry output implementation
 *
 * Encodes SAMPLE/FEATURE/EVENT binary frames into priority ring buffers.
 * LoggerTask calls TL_ReadFrame to dequeue and send via UART DMA.
 */
#include "telemetry.h"
#include <string.h>

/* ------------------------------------------------------------------ */

/* ------------------------------------------------------------------ */
/*  Default config                                                    */
/* ------------------------------------------------------------------ */
const TelemetryConfig_t* TL_GetDefaultConfig(void)
{
    static const TelemetryConfig_t s_cfg = {
        .enable_sample_output = false,    /* off by default (high-freq) */
        .sample_divider       = 1u,       /* every call                */
        .feature_divider      = 1u,       /* every call (= 10 Hz)      */
    };
    return &s_cfg;
}

/* ------------------------------------------------------------------ */
/*  Internal: push an encoded frame to a ring buffer                  */
/* ------------------------------------------------------------------ */
static void push_frame(TelemetryContext *ctx, RingBuffer *rb,
                       uint8_t type, const uint8_t *payload, uint32_t plen)
{
    uint32_t total = 1u + 1u + plen;    /* SOF + TYPE + payload + CRC */

        if (RB_FreeSpace(rb) < total) {
        ctx->stats.frame_drops++;
        return;
    }

    uint8_t hdr[2] = { TL_SOF, type };
    RB_WriteMulti(rb, hdr, 2);
    RB_WriteMulti(rb, payload, plen);

    /* CRC covers TYPE + PAYLOAD */
    
    uint32_t used = RB_Available(rb);
    if (used > ctx->stats.queue_high_water)
        ctx->stats.queue_high_water = used;

    /* Notify LoggerTask */
    if (ctx->notify_fn) ctx->notify_fn(ctx->notify_ctx);
}

/* ------------------------------------------------------------------ */
/*  Internal: read one frame from a ring buffer                       */
/* ------------------------------------------------------------------ */
static uint32_t read_frame(RingBuffer *rb, uint8_t *buf, uint32_t bs,
                           TelemetryStats_t *stats)
{
    /* Scan for SOF */
    while (RB_Available(rb) > 0) {
        uint8_t b;
        RB_Peek(rb, &b);
        if (b == TL_SOF) break;
        RB_ReadByte(rb, &b);   /* discard */
    }
    if (RB_Available(rb) < 2u) return 0;

    uint8_t type;
    RB_ReadByte(rb, &type);   /* SOF (save but we already know it) */
    buf[0] = TL_SOF;

    RB_ReadByte(rb, &type);   /* TYPE */
    buf[1] = type;

    uint32_t plen;
    switch (type) {
        case TL_TYPE_SAMPLE:  plen = 18u; break;
        case TL_TYPE_FEATURE: plen = 34u; break;
        case TL_TYPE_EVENT:   plen = 19u; break;
        default: return 0;
    }

    uint32_t total = plen;
    if (RB_Available(rb) < total) return 0;
    if (bs < 2u + total)        return 0;

    RB_ReadMulti(rb, buf + 2u, total);

    return 2u + total;

    
    return total;
}

/* ------------------------------------------------------------------ */
/*  Init                                                              */
/* ------------------------------------------------------------------ */
void TL_Init(TelemetryContext *ctx, const TelemetryConfig_t *cfg)
{
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->cfg = cfg ? *cfg : *TL_GetDefaultConfig();
    RB_Init(&ctx->rb_sample, ctx->sample_storage, sizeof(ctx->sample_storage));
    RB_Init(&ctx->rb_event,  ctx->event_storage,  sizeof(ctx->event_storage));
}

/* ------------------------------------------------------------------ */
/*  Push SAMPLE                                                        */
/* ------------------------------------------------------------------ */
void TL_PushSample(TelemetryContext *ctx, uint64_t ts,
                   float pressure, float eus, uint16_t quality)
{
    if (!ctx) return;
    ctx->stats.sample_count++;

    if (!ctx->cfg.enable_sample_output) return;
    ctx->sample_counter++;
    if (ctx->sample_counter % ctx->cfg.sample_divider != 0) return;

    uint8_t pl[18];
    memcpy(pl + 0, &ts, 8);
    memcpy(pl + 8, &pressure, 4);
    memcpy(pl + 12, &eus, 4);
    memcpy(pl + 16, &quality, 2);
    push_frame(ctx, &ctx->rb_sample, TL_TYPE_SAMPLE, pl, 18);
}

/* ------------------------------------------------------------------ */
/*  Push FEATURE                                                       */
/* ------------------------------------------------------------------ */
void TL_PushFeature(TelemetryContext *ctx, uint64_t ts,
                    float delta_p, float max_dpdt, float energy,
                    float slope, float rise_dur, float rise_ratio,
                    uint8_t eus_bursting, uint8_t burst_count)
{
    if (!ctx) return;
    ctx->stats.feature_count++;
    ctx->feature_counter++;
    if (ctx->feature_counter % ctx->cfg.feature_divider != 0) return;

    uint8_t pl[34];
    memcpy(pl + 0, &ts, 8);
    memcpy(pl + 8,  &delta_p, 4);
    memcpy(pl + 12, &max_dpdt, 4);
    memcpy(pl + 16, &energy, 4);
    memcpy(pl + 20, &slope, 4);
    memcpy(pl + 24, &rise_dur, 4);
    memcpy(pl + 28, &rise_ratio, 4);
    pl[32] = eus_bursting;
    pl[33] = burst_count;
    push_frame(ctx, &ctx->rb_event, TL_TYPE_FEATURE, pl, 34);
}

/* ------------------------------------------------------------------ */
/*  Push EVENT                                                         */
/* ------------------------------------------------------------------ */
void TL_PushEvent(TelemetryContext *ctx, uint64_t ts,
                  uint8_t class_id, uint16_t reason_flags,
                  uint8_t stim_state, uint8_t stim_trigger,
                  uint16_t fault_flags, uint32_t seq_drops)
{
    if (!ctx) return;
    ctx->stats.event_count++;

    uint8_t pl[19];
    memcpy(pl + 0, &ts, 8);
    pl[8]  = class_id;
    memcpy(pl + 9, &reason_flags, 2);
    pl[11] = stim_state;
    pl[12] = stim_trigger;
    memcpy(pl + 13, &fault_flags, 2);
    memcpy(pl + 15, &seq_drops, 4);
    push_frame(ctx, &ctx->rb_event, TL_TYPE_EVENT, pl, 19);
}

/* ------------------------------------------------------------------ */
/*  Read next frame (event buffer first, then sample)                 */
/* ------------------------------------------------------------------ */
uint32_t TL_ReadFrame(TelemetryContext *ctx, uint8_t *buf, uint32_t bs)
{
    if (!ctx || !buf || bs < 4u) return 0;

    uint32_t n = read_frame(&ctx->rb_event, buf, bs, &ctx->stats);
    if (n > 0) return n;

    return read_frame(&ctx->rb_sample, buf, bs, &ctx->stats);
}

/* ------------------------------------------------------------------ */
/*  Notify callback                                                   */
/* ------------------------------------------------------------------ */
void TL_SetNotify(TelemetryContext *ctx, TL_NotifyFn fn, void *fn_ctx)
{
    if (!ctx) return;
    ctx->notify_fn = fn;
    ctx->notify_ctx = fn_ctx;
}

/* ------------------------------------------------------------------ */
/*  Statistics                                                        */
/* ------------------------------------------------------------------ */
const TelemetryStats_t* TL_GetStats(const TelemetryContext *ctx)
{
    return ctx ? &ctx->stats : NULL;
}



