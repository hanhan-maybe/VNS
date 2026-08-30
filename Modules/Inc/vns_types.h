/**
 * @file    vns_types.h
 * @brief   VNS project common type definitions
 */
#ifndef VNS_TYPES_H
#define VNS_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */
#define VNS_NUM_CHANNELS          2
#define VNS_SAMPLES_PER_FRAME     64
#define VNS_SAMPLE_RATE          1000
#define VNS_DATA_TIMEOUT_MS       500

/* ------------------------------------------------------------------ */
/*  System timing parameters (100 Hz base)                           */
/* ------------------------------------------------------------------ */
#define VNS_SAMPLE_PERIOD_US     10000
#define VNS_WINDOW_SIZE          200
#define VNS_WINDOW_DURATION_MS   2000
#define VNS_FEATURE_UPDATE_STEP  10
#define VNS_CLASSIFY_PERIOD_MS   100

/* ------------------------------------------------------------------ */
/*  Signal channels                                                   */
/* ------------------------------------------------------------------ */
typedef enum {
    VNS_CHAN_A = 0,
    VNS_CHAN_B = 1,
} VNS_Channel;

/* ------------------------------------------------------------------ */
/*  Classification result                                             */
/* ------------------------------------------------------------------ */
typedef enum {
    VNS_CLASS_0 = 0,
    VNS_CLASS_1 = 1,
    VNS_CLASS_2 = 2,
} VNS_Class;

/* ------------------------------------------------------------------ */
/*  PC to MCU binary input frame (packed, LE)                         */
/* ------------------------------------------------------------------ */
typedef struct __attribute__((packed)) {
    uint16_t header;              /* SOF = 0x55AA                 */
    uint16_t version;             /* Protocol version = 1         */
    uint16_t payload_length;      /* Bytes after this field       */
    uint16_t flags;
    uint32_t sequence;
    uint64_t timestamp_us;
    int32_t  pressure_q16;        /* Q16.16 */
    int32_t  eus_envelope_q16;    /* Q16.16 */
    uint16_t eus_flags;
    uint16_t reserved;
    uint32_t crc32;
} VnsInputFrame_t;

/* ------------------------------------------------------------------ */
/*  Signal quality flags                                              */
/* ------------------------------------------------------------------ */
#define QUALITY_NONE            0x00u
#define QUALITY_TIMING_WARNING  0x01u
#define QUALITY_SIGNAL_INVALID  0x02u
#define QUALITY_CLIPPED         0x04u
#define QUALITY_OVERRANGE       0x08u

/* ------------------------------------------------------------------ */
/*  Raw signal sample (from acquisition, before preprocessing)        */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t sequence;
    uint64_t timestamp_us;
    float    pressure_raw;
    float    eus_raw;
    uint16_t quality_flags;
} SignalSample_t;

/* ------------------------------------------------------------------ */
/*  Preprocessed signal sample (after PreprocessTask)                 */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t sequence;
    uint64_t timestamp_us;
    float    pressure_filtered;
    float    pressure_baseline_removed;
    float    pressure_derivative;
    float    eus_envelope;
    uint16_t quality_flags;
} ProcessedSample_t;

/* ------------------------------------------------------------------ */
/*  Error counters (AcquisitionTask status)                           */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t rx_bytes;
    uint32_t valid_frames;
    uint32_t crc_errors;
    uint32_t length_errors;
    uint32_t sequence_drops;
    uint32_t timestamp_errors;
    uint32_t queue_overflows;
} VNSErrorCounters;

/* ------------------------------------------------------------------ */
/*  Data timeout event                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    bool     timeout_active;
    uint32_t tick_last_valid;
} VNS_TimeoutStatus;

/* ------------------------------------------------------------------ */
/*  Legacy frame (for signal processing pipeline)                     */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t    seq;
    uint64_t    timestamp_us;
    uint8_t     chan_mask;
    uint16_t    samples[VNS_NUM_CHANNELS][VNS_SAMPLES_PER_FRAME];
    uint16_t    actual_count;
} VNS_RawFrame;

/* ------------------------------------------------------------------ */
/*  Event descriptor                                                  */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t    timestamp_us;
    VNS_Channel channel;
    uint32_t    sample_index;
    float       peak_amplitude;
    float       rms;
    float       zero_cross_rate;
    float       pulse_width_ms;
    float       energy;
} VNSEvent;

/* ------------------------------------------------------------------ */
/*  Feature vector for classifier                                     */
/* ------------------------------------------------------------------ */
typedef struct {
    VNS_Channel channel;
    float       features[8];
    uint32_t    num_features;
} VNS_FeatureVector;

/* ------------------------------------------------------------------ */
/*  Stimulation trigger command                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    VNS_Class   stim_class;
    uint32_t    pulse_width_us;
    uint32_t    burst_count;
    uint32_t    burst_period_ms;
} VNS_StimCommand;

/* ------------------------------------------------------------------ */
/*  Log level                                                         */
/* ------------------------------------------------------------------ */
typedef enum {
    VNS_LOG_OFF   = 0,
    VNS_LOG_ERROR = 1,
    VNS_LOG_WARN  = 2,
    VNS_LOG_INFO  = 3,
    VNS_LOG_DEBUG = 4,
} VNS_LogLevel;

/* ------------------------------------------------------------------ */
/*  Callback types for hardware abstraction                           */
/* ------------------------------------------------------------------ */
typedef void (*VNS_GPIO_WriteFn)(void *context, uint8_t pin_id, bool state);
typedef uint64_t (*VNS_GetTimeUsFn)(void *context);

#ifdef __cplusplus
}
#endif

#endif /* VNS_TYPES_H */

