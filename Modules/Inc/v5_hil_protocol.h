#ifndef V5_HIL_PROTOCOL_H
#define V5_HIL_PROTOCOL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "v5_app.h"

#define V5_HIL_PRESSURE_MAGIC 0x49503556u
#define V5_HIL_TELEMETRY_MAGIC 0x4F543556u
#define V5_HIL_PRESSURE_FRAME_SIZE 20u
#define V5_HIL_TELEMETRY_FRAME_SIZE 112u

typedef struct {
    uint32_t sample_index;
    float pressure;
    bool signal_valid;
    bool cycle_reset;
} V5HilPressureFrame;

bool V5Hil_DecodePressureFrame(V5HilPressureFrame *out,
                               const uint8_t *bytes, size_t length);
size_t V5Hil_EncodeTelemetry(uint8_t *bytes, size_t capacity,
                             uint32_t sample_index, float pressure,
                             const V5AppOutput *output,
                             uint32_t processing_time_us);
typedef uint32_t (*V5HilMicrosFn)(void);
size_t AppV5_ProcessHilFrame(const uint8_t *input, size_t input_length,
                             uint8_t *output, size_t output_capacity,
                             V5HilMicrosFn micros);

#endif
