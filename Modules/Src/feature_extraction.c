/**
 * @file    feature_extraction.c
 * @brief   Pressure and EUS feature extraction implementation
 *
 * Maintains a 200-sample ring buffer (2 s at 100 Hz).
 * Features are computed every 100 ms (10-sample slide).
 * Single-pass window traversal for all features.
 */
#include "feature_extraction.h"
#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Init / Reset                                                      */
/* ------------------------------------------------------------------ */

void FE_Init(FeatureExtractor *fe, float fs)
{
    if (!fe) return;
    memset(fe, 0, sizeof(*fe));
    fe->fs = fs;
    /* Default EUS burst thresholds (normalised) */
    fe->eus_high_thresh     = 0.6f;
    fe->eus_low_thresh      = 0.2f;
    fe->min_burst_duration_s = 0.10f;   /* 100 ms */
    fe->min_burst_interval_s = 0.05f;   /*  50 ms */
}

void FE_Reset(FeatureExtractor *fe)
{
    if (!fe) return;
    float fs = fe->fs;
    memset(fe, 0, sizeof(*fe));
    fe->fs = fs;
    /* Restore defaults */
    fe->eus_high_thresh     = 0.6f;
    fe->eus_low_thresh      = 0.2f;
    fe->min_burst_duration_s = 0.10f;
    fe->min_burst_interval_s = 0.05f;
}

/* ------------------------------------------------------------------ */
/*  Feed sample                                                       */
/* ------------------------------------------------------------------ */

void FE_FeedSample(FeatureExtractor *fe, float pressure, float eus,
                   uint64_t ts_us, uint16_t quality)
{
    if (!fe) return;

    /* Track gaps */
    if (quality & QUALITY_SIGNAL_INVALID) {
        fe->samples_since_last_valid++;
        return;   /* Do not store invalid data */
    }
    fe->samples_since_last_valid = 0;

    fe->last_timestamp = ts_us;

    uint16_t pos = fe->total_samples % FE_WINDOW_SIZE;
    fe->pressure_buf[pos] = pressure;
    fe->eus_buf[pos]      = eus;
    fe->last_pressure     = pressure;
    fe->total_samples++;
}

/* ------------------------------------------------------------------ */
/*  Compute features ??single-pass window traversal                   */
/* ------------------------------------------------------------------ */


    /* ---- Populate output ---- */
void FE_Compute(FeatureExtractor *fe, FeatureSet_t *out)
{
    if (!fe || !out) return;
    memset(out, 0, sizeof(*out));
    out->timestamp_us = fe->last_timestamp;
    if (fe->total_samples < FE_WINDOW_SIZE) { out->window_ready = 0; return; }
    out->window_ready = 1;
    if (fe->samples_since_last_valid > 5) out->quality_flags |= QUALITY_SIGNAL_INVALID;

    uint16_t N = FE_WINDOW_SIZE;
    uint32_t start = (fe->total_samples - N) % N;
    const float sum_x = N * (N - 1.0f) * 0.5f;
    const float sum_xx = N * (N - 1.0f) * (2.0f * N - 1.0f) / 6.0f;
    float p_prev = fe->pressure_buf[start];
    float sum_y = 0, sum_xy = 0;
    float p_min = p_prev, p_max = p_prev;
    float max_dpdt_val = 0, energy_sum = 0;
    int rise_run = 0, max_rise_run = 0, mono_count = 0;
    float eus_sum = 0, eus_sum_sq = 0, eus_max_val = fe->eus_buf[start];
    int bstate = fe->bursting ? 1 : 0;
    int burst_start = fe->bursting ? 0 : -1;
    int burst_count = 0, transitions = 0, burst_total = 0;
    int refr_cnt = fe->refractory_remaining;
    const int MIN_BURST_SMP = (int)(fe->min_burst_duration_s * fe->fs + 0.5f);
    const int MIN_INTV_SMP  = (int)(fe->min_burst_interval_s * fe->fs + 0.5f);

    for (uint16_t i = 0; i < N; i++) {
        uint16_t idx = (start + i) % N;
        float p = fe->pressure_buf[idx];
        float e = fe->eus_buf[idx];
        if (i > 0) {
            float dp = p - p_prev;
            float dpdt = dp * fe->fs;
            if (dpdt > max_dpdt_val) max_dpdt_val = dpdt;
            energy_sum += dp * dp;
            if (dp > 0) mono_count++;
            if (dp > 0.1f) { rise_run++; if (rise_run > max_rise_run) max_rise_run = rise_run; }
            else rise_run = 0;
        }
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;
        sum_y += p; sum_xy += (float)i * p; p_prev = p;
        eus_sum += e; eus_sum_sq += e * e;
        if (e > eus_max_val) eus_max_val = e;

        if (bstate == 0) {
            if (e > fe->eus_high_thresh) { bstate = 1; burst_start = (int)i;  transitions++; }
        }
        if (bstate == 1) {
            burst_total++;
            if (e < fe->eus_low_thresh) {
                int dur = (int)i - burst_start;
                if (dur >= MIN_BURST_SMP) { burst_count++; bstate = 2; refr_cnt = 0; }
                else bstate = 0;
                transitions++;
            }
        } else if (bstate == 2) {
            refr_cnt++;
            if (refr_cnt >= MIN_INTV_SMP) {
                if (e > fe->eus_high_thresh) { bstate = 1; burst_start = (int)i; transitions++; }
                else bstate = 0;
            }
        }
    }

    float fN = (float)N;
    out->pressure_min = p_min; out->pressure_max = p_max;
    out->delta_p = p_max - p_min; out->max_dpdt = max_dpdt_val;
    out->pressure_energy = energy_sum / fN;
    out->rise_duration_s = (float)max_rise_run / fe->fs;
    out->monotonic_rise_ratio = (float)mono_count / (fN - 1.0f);
    float denom = fN * sum_xx - sum_x * sum_x;
    out->pressure_slope = (denom > 1e-12f) ? (fN * sum_xy - sum_x * sum_y) / denom : 0.0f;
    out->eus_mean = eus_sum / fN; out->eus_rms = sqrtf(eus_sum_sq / fN);
    out->eus_max = eus_max_val;
    out->eus_transition_count = (uint16_t)transitions;
    out->eus_burst_count = (uint16_t)burst_count;
    out->eus_burst_duration_s = (float)burst_total / fe->fs;
    out->eus_bursting = (uint8_t)(bstate == 1 ? 1 : 0);
    fe->bursting = (bstate == 1);
    fe->refractory_remaining = (bstate == 2) ? refr_cnt : 0;
    if (bstate != 1) fe->burst_start_idx = 0;
}

