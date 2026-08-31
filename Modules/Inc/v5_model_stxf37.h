#ifndef V5_MODEL_STXF37_H
#define V5_MODEL_STXF37_H

#include "v5_model.h"

/*
 * AUTO-GENERATED FILE.
 *
 * Source:
 *   D:/cubeIDE/project/VNS/data/NVC_V5/v5_final_validation/STxF37_m1_frozen_config.json
 *
 * Model:
 *   V5 individualized M1 P-EARLY
 *
 * DO NOT manually tune these values on MCU.
 */

static const V5ModelConfig g_v5_model_stxf37 = {
    .mean = {
        31.782833f, 35.6261985f, 3.81230769f, 4.20993856f, 2.49283367f, 14.8775944f, 0.628461538f, 234.027168f, 25.1739034f, 1.71710489f, 3.84336545f, 0.991717446f, 3.62054067f, 2.56861755f, 0.216518434f
    },
    .scale = {
        33.150451f, 37.3067992f, 4.69518928f, 4.84141621f, 3.6568676f, 9.94234338f, 0.201603043f, 304.0758f, 27.226634f, 1.82946734f, 6.49981223f, 0.846542189f, 2.87453603f, 2.35624695f, 0.0914671649f
    },
    .coef = {
        -0.154510767f, -0.156919645f, -0.198202328f, 0.196647467f, 0.231466669f, 0.189913117f, 0.251998802f, -0.168661698f, -0.15111162f, 0.0577267888f, -0.11262911f, 0.208321241f, -0.0320413284f, -0.0614629246f, -0.0154519047f
    },
    .intercept = -0.315371123f,
    .threshold = 0.779175295f
};

#endif /* V5_MODEL_STXF37_H */
