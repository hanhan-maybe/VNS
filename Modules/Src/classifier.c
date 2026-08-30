/**
 * @file    classifier.c
 * @brief   Rule-based 3-class classifier with entry/exit hysteresis
 *
 * Priority: Class2 > Class1 > Class0.
 * Entry: Class1 needs 3 consecutive, Class2 needs 2.
 * Exit: Class2 needs 2 consecutive NOT Class2.
 */
#include "classifier.h"
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ */
/*  Default config                                                    */
/* ------------------------------------------------------------------ */
const ClassifierConfig_t* CL_GetDefaultConfig(void)
{
    static const ClassifierConfig_t s_cfg = {
        .thr_delta_p1              = 5.0f,
        .thr_delta_p2              = 20.0f,
        .thr_energy                = 5.0f,
        .thr_dpdt1                 = 200.0f,
        .thr_slope2                = 0.3f,
        .thr_rise_ratio            = 0.6f,
        .class2_min_rise_duration_s = 1.0f,
        .class_hysteresis_time_ms  = 500u,
        .invalid_timeout_ms        = 5000u,
    };
    return &s_cfg;
}

/* ------------------------------------------------------------------ */
/*  Init / Reset                                                      */
/* ------------------------------------------------------------------ */

void CL_Init(ClassifierState *state)
{
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->current_class = CLASS0_STABLE;
}

void CL_Reset(ClassifierState *state)
{
    CL_Init(state);
}

/* ------------------------------------------------------------------ */
/*  Core classification                                              */
/* ------------------------------------------------------------------ */

uint8_t CL_Classify(ClassifierState *state, const FeatureSet_t *fs,
                    const ClassifierConfig_t *cfg, uint32_t tick_ms,
                    ClassifierResult_t *result)
{
    if (!result) return CLASS_INVALID;
    memset(result, 0, sizeof(*result));
    result->timestamp_us = fs ? fs->timestamp_us : 0;
    result->previous_class = state ? state->current_class : CLASS_INVALID;

    if (!state || !fs || !cfg) {
        if (result) { result->class_id = CLASS_INVALID; result->reason_flags = REASON_INVALID; }
        return CLASS_INVALID;
    }

    /* ---- 1. Validity checks ---- */
    bool invalid = !fs->window_ready ||
                   (fs->quality_flags & QUALITY_SIGNAL_INVALID);

    if (state->last_valid_tick_ms > 0 &&
        (tick_ms - state->last_valid_tick_ms) > cfg->invalid_timeout_ms) {
        invalid = true;
    }

    if (invalid) {
        result->class_id = CLASS_INVALID;
        result->changed  = (state->current_class != CLASS_INVALID) ? 1 : 0;
        result->reason_flags = REASON_INVALID;
        state->current_class = CLASS_INVALID;
        return CLASS_INVALID;
    }
    state->last_valid_tick_ms = tick_ms;

    /* ---- 2. Tentative class (without hysteresis) ---- */
    uint8_t  tentative;
    uint16_t reason = 0;
    float    confidence = 0.0f;

    /* Class2 conditions (all must be met) */
    bool c2_dp   = fs->delta_p > cfg->thr_delta_p2;
    bool c2_rdur = fs->rise_duration_s >= cfg->class2_min_rise_duration_s;
    bool c2_slp  = fs->pressure_slope > cfg->thr_slope2;
    bool c2_rrat = fs->monotonic_rise_ratio > cfg->thr_rise_ratio;
    bool c2_brst = fs->eus_bursting == 1;

    if (c2_dp && c2_rdur && c2_slp && c2_rrat && c2_brst) {
        tentative  = CLASS2_VOIDING;
        confidence = 1.0f;
        if (c2_dp)   { reason |= REASON_DELTA_P2; }
        if (c2_rdur) { reason |= REASON_RISE_DURATION; }
        if (c2_slp)  { reason |= REASON_SLOPE; }
        if (c2_rrat) { reason |= REASON_RISE_DURATION; }
        if (c2_brst) { reason |= REASON_EUS_BURST; }
        confidence = 5.0f / 5.0f;   /* all 5 met */
    } else {
        /* Class1 conditions (any one of three, eus_bursting==0) */
        bool c1_en  = fs->pressure_energy > cfg->thr_energy;
        bool c1_dp  = fs->max_dpdt > cfg->thr_dpdt1;
        bool c1_dlt = fs->delta_p > cfg->thr_delta_p1;
        int  c1_met = (c1_en ? 1 : 0) + (c1_dp ? 1 : 0) + (c1_dlt ? 1 : 0);

        if (fs->eus_bursting == 0 && c1_met > 0) {
            tentative  = CLASS1_UNSTABLE;
            if (c1_dlt) { reason |= REASON_DELTA_P1; }
            if (c1_en)  { reason |= REASON_ENERGY; }
            if (c1_dp)  { reason |= REASON_DPDT; }
            confidence = (float)c1_met / 3.0f;
        } else {
            tentative  = CLASS0_STABLE;
            confidence = 1.0f;
        }
    }
    result->reason_flags = reason;

    /* ---- 3. Hysteresis state machine ---- */
    result->changed = 0;

    switch (state->current_class) {

    case CLASS_INVALID:
        state->class1_consecutive = 0;
        state->class2_consecutive = 0;
        state->not_class2_consecutive = 0;
        /* fall through */
    case CLASS0_STABLE:

        if (tentative == CLASS2_VOIDING) {
            state->class2_consecutive++;
            if (state->class2_consecutive >= 2) {
                state->previous_class = state->current_class;
                state->current_class = CLASS2_VOIDING;
                state->class2_consecutive = 0;
                state->not_class2_consecutive = 0;
                result->changed = 1;
            }
        } else if (tentative == CLASS1_UNSTABLE) {
            state->class1_consecutive++;
            if (state->class1_consecutive >= 3) {
                state->previous_class = state->current_class;
                state->current_class = CLASS1_UNSTABLE;
                state->class1_consecutive = 0;
                result->changed = 1;
            }
        }
        break;

    case CLASS1_UNSTABLE:
        /* Check promotion to Class2 */
        if (tentative == CLASS2_VOIDING) {
            state->class2_consecutive++;
            if (state->class2_consecutive >= 2) {
                state->previous_class = state->current_class;
                state->current_class = CLASS2_VOIDING;
                state->class2_consecutive = 0;
                state->not_class2_consecutive = 0;
                result->changed = 1;
                break;
            }
        } else {
            state->class2_consecutive = 0;
        }

        /* Stay or drop out of Class1 */
        if (tentative == CLASS1_UNSTABLE) {
            state->class1_consecutive = 0;   /* steady state */
        } else if (tentative != CLASS2_VOIDING) {
            /* Fell out ??immediate exit */
            state->previous_class = state->current_class;
            state->current_class = CLASS0_STABLE;
            result->changed = 1;
        }
        break;

    case CLASS2_VOIDING:
        if (tentative == CLASS2_VOIDING) {
            state->not_class2_consecutive = 0;
        } else {
            state->not_class2_consecutive++;
            if (state->not_class2_consecutive >= 2) {
                /* Drop out of Class2 */
                state->previous_class = state->current_class;
                result->changed = 1;

                /* Re-evaluate ??go to Class0 or Class1 */
                if (tentative == CLASS1_UNSTABLE) {
                    state->current_class = CLASS1_UNSTABLE;
                    state->class1_consecutive = 1;
                } else {
                    state->current_class = CLASS0_STABLE;
                }
            }
        }
        break;
    }

    result->class_id = state->current_class;
    result->confidence_rule_score = confidence;
    return state->current_class;
}

