/**
 * @file    classifier.h
 * @brief   Rule-based 3-class classifier with hysteresis
 *
 * Takes FeatureSet_t from feature extraction, applies configurable
 * thresholds with priority (Class2 > Class1 > Class0) and entry/exit
 * hysteresis counters.
 */
#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "vns_types.h"
#include "feature_extraction.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Class IDs                                                         */
/* ------------------------------------------------------------------ */
#define CLASS0_STABLE      0
#define CLASS1_UNSTABLE    1
#define CLASS2_VOIDING     2
#define CLASS_INVALID    255

/* ------------------------------------------------------------------ */
/*  Reason flags                                                      */
/* ------------------------------------------------------------------ */
#define REASON_DELTA_P1      0x0001u
#define REASON_DELTA_P2      0x0002u
#define REASON_ENERGY        0x0004u
#define REASON_DPDT          0x0008u
#define REASON_SLOPE         0x0010u
#define REASON_RISE_DURATION 0x0020u
#define REASON_EUS_BURST     0x0040u
#define REASON_INVALID       0x0080u

/* ------------------------------------------------------------------ */
/*  Classifier configuration                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    float    thr_delta_p1;                /**< Class1 delta_p threshold       */
    float    thr_delta_p2;                /**< Class2 delta_p threshold       */
    float    thr_energy;                  /**< Class1 energy threshold        */
    float    thr_dpdt1;                   /**< Class1 dP/dt threshold         */
    float    thr_slope2;                  /**< Class2 slope threshold         */
    float    thr_rise_ratio;              /**< Class2 monotonic rise ratio    */
    float    class2_min_rise_duration_s;  /**< Class2 min rise duration (s)   */
    uint32_t class_hysteresis_time_ms;    /**< Hysteresis window (ms)         */
    uint32_t invalid_timeout_ms;          /**< Timeout for invalid (ms)       */
} ClassifierConfig_t;

/* ------------------------------------------------------------------ */
/*  Classifier result (output per evaluation)                         */
/* ------------------------------------------------------------------ */
typedef struct {
    uint8_t  class_id;
    uint8_t  previous_class;
    uint8_t  changed;                    /**< 1 if class changed this call   */
    float    confidence_rule_score;      /**< 0.0..1.0 condition ratio       */
    uint16_t reason_flags;
    uint64_t timestamp_us;
} ClassifierResult_t;

/* ------------------------------------------------------------------ */
/*  Classifier state (persists across evaluations)                    */
/* ------------------------------------------------------------------ */
typedef struct {
    uint8_t  current_class;
    uint8_t  previous_class;
    uint32_t class1_consecutive;         /**< Counter for Class1 entry       */
    uint32_t class2_consecutive;         /**< Counter for Class2 entry       */
    uint32_t not_class2_consecutive;       /**< Counter for Class2 exit        */
    uint32_t last_valid_tick_ms;
} ClassifierState;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/** @brief  Get default classifier configuration */
const ClassifierConfig_t* CL_GetDefaultConfig(void);

/** @brief  Initialise state (class = CLASS0) */
void CL_Init(ClassifierState *state);

/** @brief  Reset all state counters */
void CL_Reset(ClassifierState *state);

/**
 * @brief  Classify one FeatureSet_t
 * @param  state    Persistent state (hysteresis counters)
 * @param  fs       Feature set from extraction
 * @param  cfg      Classifier thresholds
 * @param  tick_ms  HAL_GetTick() for timeout detection
 * @param  result   Output classification result
 * @return class_id (same as result->class_id)
 */
uint8_t CL_Classify(ClassifierState *state, const FeatureSet_t *fs,
                    const ClassifierConfig_t *cfg, uint32_t tick_ms,
                    ClassifierResult_t *result);

#ifdef __cplusplus
}
#endif

#endif /* CLASSIFIER_H */
