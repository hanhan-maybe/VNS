/* Call this from a board UART/USB receive task; it performs no HAL I/O. */
#include "v5_hil_protocol.h"

size_t AppV5_ProcessHilFrame(const uint8_t *input, size_t input_length,
                             uint8_t *output, size_t output_capacity,
                             V5HilMicrosFn micros)
{
    V5HilPressureFrame frame;
    V5AppOutput result;
    uint32_t start_us = 0u, processing_time_us = 0u;
    if (!V5Hil_DecodePressureFrame(&frame, input, input_length)) return 0u;
    if (frame.cycle_reset) AppV5_BeginCycle();
    if (micros != 0) start_us = micros();
    result = AppV5_On100Hz(frame.pressure, frame.signal_valid);
    if (micros != 0) processing_time_us = micros() - start_us;
    return V5Hil_EncodeTelemetry(output, output_capacity, frame.sample_index,
                                 frame.pressure, &result, processing_time_us);
}
