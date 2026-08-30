/**
 * @file    feature_extraction.h
 * @brief   Pressure and EUS feature extraction from 2 s sliding window
 *
 * Window length = 200 samples at 100 Hz (2 s).
 * Features updated every 10 samples (100 ms).
 * Incremental statistics for most features; linear regression computed
 * on each window update.
 */
#ifndef FEATURE_EXTRACTION_H
#define FEATURE_EXTRACTION_H

#include "vns_types.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                         */
/* ------------------------------------------------------------------ */
#define FE_WINDOW_SIZE      200  /* 2 s at 100 Hz */
#define FE_UPDATE_STEP       10  /* 100 ms between compute calls */

/* ------------------------------------------------------------------ */
/*  Feature set output (computed every 100 ms)                       */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t timestamp_us;

    float    pressure_min;
    float    pressure_max;
    float    delta_p;
    float    max_dpdt;
    float    pressure_energy;
    float    pressure_slope;
    float    rise_duration_s;
    float    monotonic_rise_ratio;

    float    eus_mean;
    float    eus_rms;
    float    eus_max;
    uint16_t eus_transition_count;
    uint16_t eus_burst_count;
    float    eus_burst_duration_s;
    uint8_t  eus_bursting;

    uint8_t  window_ready;
    uint16_t quality_flags;
} FeatureSet_t;

/* ------------------------------------------------------------------ */
/*  Feature extractor (ring buffer + running state)                   */
/* ------------------------------------------------------------------ */
typedef struct {
    float    pressure_buf[FE_WINDOW_SIZE];
    float    eus_buf[FE_WINDOW_SIZE];
    uint32_t total_samples;
    uint64_t last_timestamp;
    float    last_pressure;
    float    fs;                     /* sample rate (Hz) */

    /* EUS burst configuration */
    float    eus_high_thresh;
    float    eus_low_thresh;
    float    min_burst_duration_s;
    float    min_burst_interval_s;

    /* Persistent burst state machine */
    bool     bursting;
    int      burst_start_idx;        /* sample index within window */
    int      refractory_remaining;   /* samples left in refractory */

    uint32_t samples_since_last_valid;
} FeatureExtractor;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/** @brief  Initialise extractor with default parameters */
void FE_Init(FeatureExtractor *fe, float fs);

/** @brief  Reset all state (clears buffer and counters) */
void FE_Reset(FeatureExtractor *fe);

/**
 * @brief  Feed one incoming sample
 * @param  fe         Extractor
 * @param  pressure   Raw pressure value
 * @param  eus        Raw EUS envelope value
 * @param  ts_us      Timestamp (microseconds)
 * @param  quality    QUALITY_* flags from upstream
 */
void FE_FeedSample(FeatureExtractor *fe, float pressure, float eus,
                   uint64_t ts_us, uint16_t quality);

/**
 * @brief  Compute features from the current window
 * @param  fe   Extractor
 * @param  out  Output feature set (only written when window_ready)
 */
void FE_Compute(FeatureExtractor *fe, FeatureSet_t *out);

#ifdef __cplusplus
}
#endif

#endif /* FEATURE_EXTRACTION_H */
