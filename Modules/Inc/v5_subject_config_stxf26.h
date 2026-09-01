#ifndef V5_SUBJECT_CONFIG_STXF26_H
#define V5_SUBJECT_CONFIG_STXF26_H
#include "v5_subject_config.h"
static const V5SubjectConfig g_v5_subject_config_stxf26 = {
    .magic = V5_SUBJECT_CONFIG_MAGIC,
    .version = V5_SUBJECT_CONFIG_VERSION,
    .feature_count = V5_MODEL_FEATURE_COUNT,
    .feature_order_hash = V5_SUBJECT_CONFIG_FEATURE_HASH,
    .model_hash = { 0x3Eu, 0x59u, 0xE1u, 0xC3u, 0x79u, 0x08u, 0x48u, 0x91u, 0x52u, 0xD4u, 0xAEu, 0x3Cu, 0xEAu, 0x71u, 0xDAu, 0x84u, 0xA8u, 0xF2u, 0x95u, 0x2Fu, 0x29u, 0x1Bu, 0xF7u, 0xACu, 0x47u, 0xC9u, 0x82u, 0xABu, 0xC9u, 0x0Bu, 0x8Eu, 0xB2u },
    .model = {
        .mean = { 1.08596651f, 1.15973484f, 0.7975f, 0.113569584f, 0.103024733f, 2.06914927f, 0.526666667f, 1.59659587f, 1.0387859f, 0.0105448512f, 0.0737683248f, 0.0474541602f, 1.05117911f, 1.04988641f, 0.159469564f },
        .scale = { 0.837378357f, 0.830856237f, 0.82801092f, 0.21609192f, 0.219065267f, 0.792262879f, 0.0881602077f, 1.04168094f, 0.765545738f, 0.0397407956f, 0.084398222f, 0.0524910586f, 1.35986656f, 1.27095642f, 0.0754783674f },
        .coef = { 0.211031096f, 0.198074916f, 0.0325862072f, 0.237416903f, 0.247503967f, 0.158387979f, 0.219254294f, 0.0743527193f, 0.189200136f, -0.0733666263f, -0.143854844f, 0.233903944f, 0.144879438f, 0.143934719f, -0.0478852456f },
        .intercept = -1.05165352f,
        .threshold = 0.738942976f
    },
    .candidate_prior_sigma_p = 0.309076365f,
    .candidate_prior_sigma_dpdt = 3.17749689f,
    .crc32 = 0xB28C69EDu
};
#endif
