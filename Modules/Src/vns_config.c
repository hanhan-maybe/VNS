
/**
 * @file    vns_config.c
 * @brief   ????????? ???????????????? */
#include "vns_config.h"
#include <string.h>
#include <stdlib.h>

/* ????????? */
static const VNS_Config g_default_cfg = {
    .sample_rate        = 100U,
    .samples_per_frame  = 200U,
    .num_channels       = 2U,

    .filter = {
        .highpass_cutoff  = 0.5f,
        .lowpass_cutoff   = 100.0f,
        .notch_freq       = 50.0f,
        .filter_order     = 4,
    },

    .event = {
        .threshold_a      = 0.05f,
        .threshold_b      = 0.03f,
        .refractory_s     = 0.050f,
        .min_event_samples = 3,
    },

    .feature = {
        .pre_samples      = 10,
        .post_samples     = 30,
    },

    .classify = {
        .class1_thresh_amp   = 0.15f,
        .class1_thresh_rms   = 0.10f,
        .class1_thresh_zcr   = 0.20f,
        .class2_thresh_amp   = 0.40f,
        .class2_thresh_rms   = 0.30f,
        .class2_thresh_zcr   = 0.05f,
        .class2_ratio_ab     = 2.0f,
    },

    .stim = {
        .pulse_width_us   = 500U,
        .min_interval_ms  = 100U,
        .burst_duration_ms = 2000U,
        .stim_gpio_pin    = 0,
    },

    .log = {
        .baudrate         = 115200U,
        .level            = VNS_LOG_INFO,
    },

    .usb = {
        .rx_buf_size      = 2048U,
        .rx_timeout_ms    = 100U,
    },
};

/* ??????????????*/
static VNS_Config g_runtime_cfg;
static int         g_cfg_inited = 0;

/* ------------------------------------------------------------------ */

const VNS_Config* VNS_ConfigGetDefault(void)
{
    return &g_default_cfg;
}

VNS_Config* VNS_ConfigGet(void)
{
    if (!g_cfg_inited) {
        memcpy(&g_runtime_cfg, &g_default_cfg, sizeof(g_runtime_cfg));
        g_cfg_inited = 1;
    }
    return &g_runtime_cfg;
}

void VNS_ConfigApply(const VNS_Config *cfg)
{
    if (cfg) {
        memcpy(&g_runtime_cfg, cfg, sizeof(g_runtime_cfg));
        g_cfg_inited = 1;
    }
}

/* ------------------------------------------------------------------ */
/*  ?????????????(????????????)                                 */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *name;
    size_t      offset;
    size_t      size;
} FieldMap;

#define FIELD_MAP(type, struct_name, field) \
    { #field, offsetof(type, field), sizeof(((type*)0)->field) }

#define FIELD_MAP_NESTED(type, parent, field) \
    { #parent "_" #field, offsetof(type, parent) + offsetof(typeof(((type*)0)->parent), field), \
      sizeof(((type*)0)->parent.field) }

static const FieldMap s_field_map[] = {
    /* ??? */
    FIELD_MAP(VNS_Config, , sample_rate),
    FIELD_MAP(VNS_Config, , samples_per_frame),
    /* filter */
    FIELD_MAP_NESTED(VNS_Config, filter, highpass_cutoff),
    FIELD_MAP_NESTED(VNS_Config, filter, lowpass_cutoff),
    FIELD_MAP_NESTED(VNS_Config, filter, notch_freq),
    /* event */
    FIELD_MAP_NESTED(VNS_Config, event, threshold_a),
    FIELD_MAP_NESTED(VNS_Config, event, threshold_b),
    FIELD_MAP_NESTED(VNS_Config, event, refractory_s),
    FIELD_MAP_NESTED(VNS_Config, event, min_event_samples),
    /* classify */
    FIELD_MAP_NESTED(VNS_Config, classify, class1_thresh_amp),
    FIELD_MAP_NESTED(VNS_Config, classify, class1_thresh_rms),
    FIELD_MAP_NESTED(VNS_Config, classify, class1_thresh_zcr),
    FIELD_MAP_NESTED(VNS_Config, classify, class2_thresh_amp),
    FIELD_MAP_NESTED(VNS_Config, classify, class2_thresh_rms),
    FIELD_MAP_NESTED(VNS_Config, classify, class2_thresh_zcr),
    FIELD_MAP_NESTED(VNS_Config, classify, class2_ratio_ab),
    /* stim */
    FIELD_MAP_NESTED(VNS_Config, stim, pulse_width_us),
    FIELD_MAP_NESTED(VNS_Config, stim, min_interval_ms),
    FIELD_MAP_NESTED(VNS_Config, stim, burst_duration_ms),
};

static const size_t s_field_map_count =
    sizeof(s_field_map) / sizeof(s_field_map[0]);

int VNS_ConfigSetField(const char *name, float value)
{
    VNS_Config *cfg = VNS_ConfigGet();
    if (!cfg || !name) return -1;

    for (size_t i = 0; i < s_field_map_count; i++) {
        if (strcmp(s_field_map[i].name, name) == 0) {
            uint8_t *base = (uint8_t*)cfg;
            void    *addr = base + s_field_map[i].offset;

            if (s_field_map[i].size == sizeof(float)) {
                *(float*)addr = value;
            } else if (s_field_map[i].size == sizeof(uint32_t)) {
                *(uint32_t*)addr = (uint32_t)value;
            } else if (s_field_map[i].size == sizeof(uint16_t)) {
                *(uint16_t*)addr = (uint16_t)value;
            } else if (s_field_map[i].size == sizeof(uint8_t)) {
                *(uint8_t*)addr = (uint8_t)value;
            } else {
                return -1;
            }
            return 0;
        }
    }
    return -1; /* ?????*/
}

/* ------------------------------------------------------------------ */
/*  Signal processing default config                                  */
/* ------------------------------------------------------------------ */
const SignalConfig_t* VNS_SignalConfigGetDefault(void)
{
    static const SignalConfig_t s_sig_cfg = {
        .pressure = {
            .cutoff_freq       = 15.0f,
            .baseline_tau_s    = 10.0f,
            .min_value         = -50.0f,
            .max_value         = 300.0f,
            .spike_reject_factor = 5.0f,
        },
        .eus = {
            .cutoff_freq       = 10.0f,
            .baseline_tau_s    = 5.0f,
            .adaptive_alpha    = 0.01f,
            .threshold_low     = 0.2f,
            .threshold_high    = 0.8f,
        },
        .timing = {
            .expected_interval_ms = 10.0f,
            .interval_tolerance   = 0.2f,
            .max_bad_intervals    = 3,
        },
    };
    return &s_sig_cfg;
}

