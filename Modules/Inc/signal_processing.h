
/**
 * @file    signal_processing.h
 * @brief   ????????????
 *
 * ??? IIR ????????? (Biquad) ???????? * ??????????????????????????? * ?????? II ???????? ???????????? *
 * ?????????????????????? */
#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include "vns_types.h"
#include "vns_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  ??????                                                          */
/* ------------------------------------------------------------------ */
#define BQ_MAX_SECTION 4   /* ????4 ??(2 ?? */

typedef struct {
    float b0, b1, b2;       /* ?????? */
    float a1, a2;           /* ?????? (a0 = 1) */
} BiquadCoeff;

typedef struct {
    BiquadCoeff coeff[BQ_MAX_SECTION];
    uint8_t     num_sections;
} IIRFilter;

/** ?????????????(????????????? */
typedef struct {
    float w1, w2;           /* ?????(???????? ???????????? */
} BiquadState;

typedef struct {
    BiquadState state[BQ_MAX_SECTION];
} IIRFilterState;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief  ??? Butterworth ???????? (????????
 * @param  flt      ??????????? * @param  order    ??? (2, 4)
 * @param  cutoff   ?????? (????? 0~0.5, ??f_c / fs)
 */
void SP_DesignLowPass(IIRFilter *flt, uint8_t order, float cutoff);

/**
 * @brief  ??? Butterworth ????????
 * @param  flt      ???
 * @param  order    ???
 * @param  cutoff   ??????????? */
void SP_DesignHighPass(IIRFilter *flt, uint8_t order, float cutoff);

/**
 * @brief  ??? Butterworth ????????
 * @param  flt      ???
 * @param  order    ???
 * @param  f_low    ???????? * @param  f_high   ???????? */
void SP_DesignBandPass(IIRFilter *flt, uint8_t order,
                       float f_low, float f_high);

/**
 * @brief  ?????? (Notch) ????? * @param  flt      ???
 * @param  freq     ??????????? * @param  q        ?????? (??? 0.5~10)
 */
void SP_DesignNotch(IIRFilter *flt, float freq, float q);

/** @brief  ?????????????(???) */
void SP_InitState(IIRFilterState *st);

/**
 * @brief  ?????????
 * @param  flt   ???????? * @param  st    ????(?????????)
 * @param  input ??????
 * @return ???????? */
float SP_ProcessSample(const IIRFilter *flt,
                       IIRFilterState *st, float input);

/**
 * @brief  ????????? (????????
 * @param  flt     ????? * @param  st      ???? * @param  input   ???????? * @param  output  ????????(??? input ???)
 * @param  n       ????? */
void SP_ProcessBlock(const IIRFilter *flt, IIRFilterState *st,
                     const float *input, float *output, uint32_t n);

/* ------------------------------------------------------------------ */
/*  ???????(?????+ ???????                                      */
/* ------------------------------------------------------------------ */

/**
 * @brief  ??????????(???????
 * @param  signal    ?????????
 * @param  len       ??????
 * @param  threshold ????(?????
 * @param  min_samples ???????????
 * @param  event_start ????????????
 * @param  event_end   ????????????
 * @return ????????? (0 ??1)
 */
int SP_DetectEvent(const float *signal, uint32_t len,
                   float threshold, uint16_t min_samples,
                   uint32_t *event_start, uint32_t *event_end);

#ifdef __cplusplus
}
#endif


/* ================================================================== */
/*  Preprocessing module ? real-time signal conditioning              */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/*  Pressure processor: IIR LPF + baseline removal + derivative       */
/* ------------------------------------------------------------------ */
typedef struct {
    float  b0, b1, a1;
    float  baseline;
    float  x_prev, y_prev;
    float  fs;
    float  lpf_cutoff_hz;
    float  baseline_tau_s;
    float  min_val, max_val;
    float  spike_reject_factor;
    bool   valid;
} PressureProcessor;

void PP_Init(PressureProcessor *p, const SignalConfig_t *cfg, float fs);
void PP_Reset(PressureProcessor *p);
int  PP_Process(PressureProcessor *p, float input, float *filtered_out, float *baseline_removed, float *derivative);

/* ------------------------------------------------------------------ */
/*  EUS processor: IIR LPF + baseline + adaptive norm + hysteresis   */
/* ------------------------------------------------------------------ */
typedef struct {
    float  b0, b1, a1;
    float  baseline;
    float  x_prev, y_prev;
    float  adaptive_max;
    float  fs;
    float  lpf_cutoff_hz;
    float  baseline_tau_s;
    float  adaptive_alpha;
    float  thresh_low, thresh_high;
    bool   high_state;
    bool   valid;
} EUSProcessor;

void EP_Init(EUSProcessor *p, const SignalConfig_t *cfg, float fs);
void EP_Reset(EUSProcessor *p);
int  EP_Process(EUSProcessor *p, float input, float *eus_out, bool *high_state);

/* ------------------------------------------------------------------ */
/*  Timestamp monitor: interval validation                            */
/* ------------------------------------------------------------------ */
typedef struct {
    uint64_t last_ts_us;
    uint32_t bad_interval_count;
    float    expected_interval_ms;
    float    interval_tolerance;
    uint32_t max_bad_intervals;
    bool     has_first;
} TimestampMonitor;

void       TM_Init(TimestampMonitor *tm, const SignalConfig_t *cfg);
uint16_t   TM_Check(TimestampMonitor *tm, uint64_t timestamp_us);

#endif /* SIGNAL_PROCESSING_H */



