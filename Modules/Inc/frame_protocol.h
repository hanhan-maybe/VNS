/**
 * @file    frame_protocol.h
 * @brief   PC to MCU data replay protocol (VnsInputFrame_t)
 *
 * Wire format (binary, little-endian, 36 bytes total):
 *   Bytes 0-1:   header = 0x55AA
 *   Bytes 2-3:   version = 1
 *   Bytes 4-5:   payload_length = 30
 *   Bytes 6-7:   flags
 *   Bytes 8-11:  sequence (uint32 LE)
 *   Bytes 12-19: timestamp_us (uint64 LE)
 *   Bytes 20-23: pressure_q16 (int32 LE, Q16.16)
 *   Bytes 24-27: eus_envelope_q16 (int32 LE, Q16.16)
 *   Bytes 28-29: eus_flags (uint16 LE)
 *   Bytes 30-31: reserved (uint16 LE)
 *   Bytes 32-35: crc32 (uint32 LE)  — CRC-32 over bytes 0..31
 *
 * CRC-32 polynomial: 0xEDB88320 (reflected, standard Ethernet/zip)
 */
#ifndef FRAME_PROTOCOL_H
#define FRAME_PROTOCOL_H

#include "vns_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Protocol constants                                                */
/* ------------------------------------------------------------------ */
#define FRAME_HEADER            0x55AAu
#define FRAME_VERSION           1u
#define FRAME_CRC_LEN           4u
#define FRAME_HEADER_FIELDS_LEN 6u   /* header(2) + version(2) + payload_length(2) */
#define FRAME_PAYLOAD_LEN       26u  /* flags..reserved, excluding crc32 */
#define FRAME_TOTAL_LEN         (FRAME_HEADER_FIELDS_LEN + FRAME_PAYLOAD_LEN + FRAME_CRC_LEN)

/* Frame validation result codes */
#define FRAME_OK                0
#define FRAME_ERR_BAD_HEADER   -1
#define FRAME_ERR_BAD_VERSION  -2
#define FRAME_ERR_BAD_LENGTH   -3
#define FRAME_ERR_BAD_CRC      -4

/* ------------------------------------------------------------------ */
/*  Sequence / timestamp tracking context                             */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t last_seq;
    uint64_t last_timestamp_us;
    bool     has_valid;           /* true after first valid frame */
} FrameTracker;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/** CRC-32 over arbitrary data (standard reflected polynomial) */
uint32_t FP_CRC32(const uint8_t *data, uint32_t len);

/**
 * @brief  Validate a complete VnsInputFrame_t
 * @param  frame  Pointer to received frame
 * @param  size   Bytes received (must equal sizeof(VnsInputFrame_t))
 * @return FRAME_OK or FRAME_ERR_*
 */
int FP_ValidateFrame(const VnsInputFrame_t *frame, uint32_t size);

/** @brief  Check sequence for gaps; returns 0=ok, -1=drop detected */
int FP_CheckSequence(FrameTracker *tracker, uint32_t seq);

/** @brief  Check timestamp monotonicity; returns 0=ok, -1=non-monotonic */
int FP_CheckTimestamp(FrameTracker *tracker, uint64_t ts_us);

/** @brief  Initialize frame tracker */
void FP_TrackerInit(FrameTracker *tracker);

/** @brief  Convert Q16.16 fixed-point to float */
static inline float FP_Q16ToFloat(int32_t q16)
{
    return (float)q16 / 65536.0f;
}

/** @brief  Convert float to Q16.16 fixed-point */
static inline int32_t FP_FloatToQ16(float v)
{
    return (int32_t)(v * 65536.0f);
}

/** @brief  Fill VnsInputFrame_t fields and compute CRC32 */
void FP_FillFrame(VnsInputFrame_t *frame, uint32_t seq, uint64_t ts_us,
                  float pressure, float eus_envelope, uint16_t eus_flags);

/** @brief  Convert VnsInputFrame_t to internal SignalSample_t */
void FP_ToSample(const VnsInputFrame_t *frame, SignalSample_t *sample);

/** @brief  Search raw byte buffer for frame header (0x55AA) */
int FP_FindHeader(const uint8_t *buf, uint32_t buf_len, uint32_t start_offset);

#ifdef __cplusplus
}
#endif

#endif /* FRAME_PROTOCOL_H */
