/**
 * @file    signal_processing.c
 * @brief   IIR filter design + real-time signal preprocessing
 *
 * Contains:
 *   - Biquad IIR filter design (Butterworth LPF/HPF/BPF/Notch)
 *   - Pressure processor (LPF + baseline + dP/dt)
 *   - EUS processor (LPF + baseline + adaptive norm + hysteresis)
 *   - Timestamp monitor (interval validation)
 */
#include "signal_processing.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ================================================================== */
/*  IIR filter helpers                                                */
/* ================================================================== */
static void butter_poles(uint8_t n, float *real, float *imag)
{
    for (uint8_t k = 0; k < n; k++) {
        float angle = (float)M_PI * (2.0f * k + 1.0f) / (2.0f * n);
        real[k] = -sinf(angle);
        imag[k] = cosf(angle);
    }
}

static float prewarp(float omega_d, float fs)
{
    return 2.0f * fs * tanf(omega_d / 2.0f);
}

static void s_to_z(const float pole_re, const float pole_im,
                   float omega_a, float fs,
                   float *b0, float *b1, float *b2,
                   float *a1, float *a2)
{
    float sigma = pole_re * omega_a;
    float omega_p = pole_im * omega_a;
    float a_s1 = -2.0f * sigma;
    float a_s0 = sigma * sigma + omega_p * omega_p;
    float gain = omega_a * omega_a;
    float T = 1.0f / fs;
    float T2 = T * T;
    float ad = 4.0f + 2.0f * a_s1 * T + a_s0 * T2;
    *b0 = (gain * T2) / ad;
    *b1 = (2.0f * gain * T2) / ad;
    *b2 = *b0;
    *a1 = (2.0f * a_s0 * T2 - 8.0f) / ad;
    *a2 = (4.0f - 2.0f * a_s1 * T + a_s0 * T2) / ad;
}

/* ================================================================== */
/*  IIR filter design API                                             */
/* ================================================================== */

void SP_DesignLowPass(IIRFilter *flt, uint8_t order, float cutoff)
{
    if (!flt || order < 2 || order > 8 || cutoff <= 0.0f || cutoff >= 0.5f)
        { if (flt) flt->num_sections = 0; return; }
    uint8_t n = order / 2; flt->num_sections = n;
    float real[8], imag[8]; butter_poles(order, real, imag);
    float fs = 1.0f;
    float omega_a = prewarp(2.0f * (float)M_PI * cutoff, fs);
    for (uint8_t i = 0; i < n; i++)
        s_to_z(real[i], imag[i], omega_a, fs,
               &flt->coeff[i].b0, &flt->coeff[i].b1, &flt->coeff[i].b2,
               &flt->coeff[i].a1, &flt->coeff[i].a2);
}

void SP_DesignHighPass(IIRFilter *flt, uint8_t order, float cutoff)
{
    if (!flt || order < 2 || order > 8 || cutoff <= 0.0f || cutoff >= 0.5f) return;
    SP_DesignLowPass(flt, order, 0.5f - cutoff);
    for (uint8_t i = 0; i < flt->num_sections; i++) {
        flt->coeff[i].b1 = -flt->coeff[i].b1;
        flt->coeff[i].a1 = -flt->coeff[i].a1;
    }
}

void SP_DesignBandPass(IIRFilter *flt, uint8_t order,
                       float f_low, float f_high) { (void)flt; (void)order; (void)f_low; (void)f_high; }

void SP_DesignNotch(IIRFilter *flt, float freq, float q)
{
    if (!flt || freq <= 0.0f || freq >= 0.5f || q <= 0.0f) return;
    flt->num_sections = 1;
    float omega = 2.0f * (float)M_PI * freq;
    float alpha = sinf(omega) / (2.0f * q);
    flt->coeff[0].b0 = 1.0f; flt->coeff[0].b1 = -2.0f * cosf(omega); flt->coeff[0].b2 = 1.0f;
    flt->coeff[0].a1 = flt->coeff[0].b1; flt->coeff[0].a2 = 1.0f - alpha;
    float a0 = 1.0f + alpha;
    flt->coeff[0].b0 /= a0; flt->coeff[0].b1 /= a0; flt->coeff[0].b2 /= a0;
    flt->coeff[0].a1 /= -a0; flt->coeff[0].a2 /= -a0;
}

void SP_InitState(IIRFilterState *st) { if (st) memset(st, 0, sizeof(*st)); }

float SP_ProcessSample(const IIRFilter *flt, IIRFilterState *st, float input)
{
    if (!flt || !st) return input;
    float out = input;
    for (uint8_t i = 0; i < flt->num_sections; i++) {
        const BiquadCoeff *c = &flt->coeff[i];
        BiquadState       *s = &st->state[i];
        float y = c->b0 * out + s->w1;
        s->w1 = c->b1 * out - c->a1 * y + s->w2;
        s->w2 = c->b2 * out - c->a2 * y;
        out = y;
    }
    return out;
}

void SP_ProcessBlock(const IIRFilter *flt, IIRFilterState *st,
                     const float *input, float *output, uint32_t n)
{
    if (!flt || !st || !input || !output || n == 0) return;
    for (uint32_t i = 0; i < n; i++)
        output[i] = SP_ProcessSample(flt, st, input[i]);
}

int SP_DetectEvent(const float *signal, uint32_t len,
                   float threshold, uint16_t min_samples,
                   uint32_t *event_start, uint32_t *event_end)
{
    if (!signal || len == 0) return 0;
    uint32_t start = 0, count = 0;
    for (uint32_t i = 0; i < len; i++) {
        if (fabsf(signal[i]) > threshold) {
            if (count == 0) start = i; count++;
        } else {
            if (count >= min_samples) {
                if (event_start) *event_start = start;
                if (event_end)   *event_end   = i;
                return 1;
            } count = 0;
        }
    }
    if (count >= min_samples) {
        if (event_start) *event_start = start;
        if (event_end)   *event_end   = len;
        return 1;
    }
    return 0;
}

/* ================================================================== */
/*  Preprocessing: 1st-order LPF coefficient helper                   */
/* ================================================================== */
static void lpf_coeff_1st(float fs, float fc,
                           float *b0, float *b1, float *a1)
{
    if (fs <= 0.0f || fc <= 0.0f || fc >= fs * 0.5f) {
        *b0 = 1.0f; *b1 = 0.0f; *a1 = 0.0f; return;
    }
    float K = tanf((float)M_PI * fc / fs);
    float inv = 1.0f / (1.0f + K);
    *b0 = K * inv; *b1 = *b0; *a1 = (K - 1.0f) * inv;
}

/* ================================================================== */
/*  PressureProcessor                                                 */
/* ================================================================== */
void PP_Init(PressureProcessor *p, const SignalConfig_t *cfg, float fs)
{
    if (!p || !cfg) return;
    memset(p, 0, sizeof(*p));
    p->fs = fs;
    p->lpf_cutoff_hz    = cfg->pressure.cutoff_freq;
    p->baseline_tau_s   = cfg->pressure.baseline_tau_s;
    p->min_val          = cfg->pressure.min_value;
    p->max_val          = cfg->pressure.max_value;
    p->spike_reject_factor = cfg->pressure.spike_reject_factor;
    lpf_coeff_1st(fs, cfg->pressure.cutoff_freq, &p->b0, &p->b1, &p->a1);
    p->valid = true;
}

void PP_Reset(PressureProcessor *p)
{
    if (!p) return;
    p->x_prev = 0.0f; p->y_prev = 0.0f;
    p->baseline = 0.0f; p->valid = true;
}

int PP_Process(PressureProcessor *p, float input,
               float *filtered_out,
               float *baseline_removed, float *derivative)
{
    if (!p || !p->valid) return -1;
    if (isnan(input) || isinf(input)) {
        if (baseline_removed) *baseline_removed = 0.0f;
        if (derivative) *derivative = 0.0f;
        if (filtered_out) *filtered_out = p->y_prev;
        return 0;
    }
    if (input > p->max_val) input = p->max_val;
    if (input < p->min_val) input = p->min_val;

    /* 1st-order IIR LPF */
    float y = p->b0 * input + p->b1 * p->x_prev - p->a1 * p->y_prev;
    if (isnan(y) || isinf(y)) y = p->y_prev;

    /* Derivative before updating state */
    float d = (y - p->y_prev) * p->fs;
    if (derivative) *derivative = d;

    p->x_prev = input;
    p->y_prev = y;

    /* Baseline EMA */
    float alpha = 1.0f - expf(-1.0f / (p->baseline_tau_s * p->fs));
    p->baseline = alpha * y + (1.0f - alpha) * p->baseline;

    float br = y - p->baseline;
    if (baseline_removed) *baseline_removed = br;

    if (y > p->max_val) y = p->max_val;
    if (y < p->min_val) y = p->min_val;
    if (filtered_out) *filtered_out = y;
    return 0;
}

/* ================================================================== */
/*  EUSProcessor                                                      */
/* ================================================================== */
void EP_Init(EUSProcessor *p, const SignalConfig_t *cfg, float fs)
{
    if (!p || !cfg) return;
    memset(p, 0, sizeof(*p));
    p->fs = fs;
    p->lpf_cutoff_hz  = cfg->eus.cutoff_freq;
    p->baseline_tau_s = cfg->eus.baseline_tau_s;
    p->adaptive_alpha = cfg->eus.adaptive_alpha;
    p->thresh_low     = cfg->eus.threshold_low;
    p->thresh_high    = cfg->eus.threshold_high;
    p->adaptive_max   = 1e-6f;
    lpf_coeff_1st(fs, cfg->eus.cutoff_freq, &p->b0, &p->b1, &p->a1);
    p->valid = true;
}

void EP_Reset(EUSProcessor *p)
{
    if (!p) return;
    p->x_prev = 0.0f; p->y_prev = 0.0f;
    p->baseline = 0.0f;
    p->adaptive_max = 1e-6f;
    p->high_state = false;
    p->valid = true;
}

int EP_Process(EUSProcessor *p, float input,
               float *eus_out, bool *high_state)
{
    if (!p || !p->valid) return -1;
    if (isnan(input) || isinf(input)) {
        if (eus_out) *eus_out = p->y_prev;
        if (high_state) *high_state = p->high_state;
        return 0;
    }
    if (input < 0.0f) input = 0.0f;
    if (input > 10.0f) input = 10.0f;

    float y = p->b0 * input + p->b1 * p->x_prev - p->a1 * p->y_prev;
    if (isnan(y) || isinf(y)) y = p->y_prev;
    p->x_prev = input;
    p->y_prev = y;

    float alpha = 1.0f - expf(-1.0f / (p->baseline_tau_s * p->fs));
    p->baseline = alpha * y + (1.0f - alpha) * p->baseline;

    p->adaptive_max *= (1.0f - p->adaptive_alpha);
    float abs_y = fabsf(y);
    if (abs_y > p->adaptive_max) p->adaptive_max = abs_y;
    if (p->adaptive_max < 1e-6f) p->adaptive_max = 1e-6f;

    float normalized = (y - p->baseline) / p->adaptive_max;
    if (p->high_state) {
        if (normalized < p->thresh_low) p->high_state = false;
    } else {
        if (normalized > p->thresh_high) p->high_state = true;
    }
    if (eus_out) *eus_out = y;
    if (high_state) *high_state = p->high_state;
    return 0;
}

/* ================================================================== */
/*  TimestampMonitor                                                  */
/* ================================================================== */
void TM_Init(TimestampMonitor *tm, const SignalConfig_t *cfg)
{
    if (!tm || !cfg) return;
    memset(tm, 0, sizeof(*tm));
    tm->expected_interval_ms = cfg->timing.expected_interval_ms;
    tm->interval_tolerance   = cfg->timing.interval_tolerance;
    tm->max_bad_intervals    = cfg->timing.max_bad_intervals;
}

uint16_t TM_Check(TimestampMonitor *tm, uint64_t timestamp_us)
{
    if (!tm) return QUALITY_SIGNAL_INVALID;
    if (!tm->has_first) {
        tm->has_first = true;
        tm->last_ts_us = timestamp_us;
        return QUALITY_NONE;
    }
    uint64_t interval_us = timestamp_us - tm->last_ts_us;
    tm->last_ts_us = timestamp_us;
    float interval_ms = (float)interval_us / 1000.0f;
    float dev = fabsf(interval_ms - tm->expected_interval_ms) / tm->expected_interval_ms;
    if (dev > tm->interval_tolerance) {
        tm->bad_interval_count++;
        if (tm->bad_interval_count >= tm->max_bad_intervals)
            return QUALITY_SIGNAL_INVALID;
        return QUALITY_TIMING_WARNING;
    }
    tm->bad_interval_count = 0;
    return QUALITY_NONE;
}
