#ifndef V5_MODEL_STXF26_H
#define V5_MODEL_STXF26_H

#include "v5_model.h"

/*
 * AUTO-GENERATED FILE.
 *
 * Source:
 *   D:/cubeIDE/project/VNS/data/NVC_V5/v5_final_validation/STxF26_m1_frozen_config.json
 *
 * Model:
 *   V5 individualized M1 P-EARLY
 *
 * DO NOT manually tune these values on MCU.
 */

static const V5ModelConfig g_v5_model_stxf26 = {
    .mean = {
        1.08596651f, 1.15973484f, 0.7975f, 0.113569584f, 0.103024733f, 2.06914927f, 0.526666667f, 1.59659587f, 1.0387859f, 0.0105448512f, 0.0737683248f, 0.0474541602f, 1.05117911f, 1.04988641f, 0.159469564f
    },
    .scale = {
        0.837378357f, 0.830856237f, 0.82801092f, 0.21609192f, 0.219065267f, 0.792262879f, 0.0881602077f, 1.04168094f, 0.765545738f, 0.0397407956f, 0.084398222f, 0.0524910586f, 1.35986656f, 1.27095642f, 0.0754783674f
    },
    .coef = {
        0.211031096f, 0.198074916f, 0.0325862072f, 0.237416903f, 0.247503967f, 0.158387979f, 0.219254294f, 0.0743527193f, 0.189200136f, -0.0733666263f, -0.143854844f, 0.233903944f, 0.144879438f, 0.143934719f, -0.0478852456f
    },
    .intercept = -1.05165352f,
    .threshold = 0.738942976f
};

#endif /* V5_MODEL_STXF26_H */
