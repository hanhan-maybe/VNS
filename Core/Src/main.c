/**
 * @file    main.c
 * @brief   VNS on NUCLEO-N657X0-Q -- main program
 *
 * FreeRTOS tasks:
 *   AcquisitionTask  - UART DMA reception + frame validation
 *   SignalProcTask   - filtering + event detection
 *   FeatureExtractTask - feature extraction
 *   ClassifierTask   - rule-based 3-class classification
 *   StimTask         - stimulation trigger output + data timeout handling
 *   LoggerTask       - serial log output
 *
 * ISR policy: only DMA status, ring buffer write, and task notification.
 */
#include "main.h"
#include "stm32n6xx_hal.h"
#include "stm32n6xx_it.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

/* ================================================================== */
/*  Global instances                                                  */
/* ================================================================== */

/* --- Queues --- */
QueueHandle_t g_signalSampleQueue = NULL;
QueueHandle_t g_rawFrameQueue     = NULL;
QueueHandle_t g_eventQueue        = NULL;
QueueHandle_t g_featureQueue      = NULL;
QueueHandle_t g_stimQueue         = NULL;

/* --- Acquisition UART DMA --- */
DmaRxBuffer        g_acq_dma;
UART_HandleTypeDef huart_acq;             /* CubeMX configures this as UART4 etc. */
static uint8_t     s_acq_dma_buf[2048];
static uint8_t     s_acq_rb_storage[4096];
static RingBuffer  g_acq_rb;

/* --- Error counters and timeout --- */
StimControl       g_stim;
TelemetryContext  g_telemetry;
VNSErrorCounters  g_acq_errors;
SemaphoreHandle_t g_dataTimeoutSem = NULL;
uint32_t          g_acq_last_valid_tick = 0;

/* --- Logger ring buffer --- */
static uint8_t    s_log_rb_storage[1024];
static RingBuffer s_log_rb;

/* --- Signal processing chain --- */
IIRFilter         g_bp_filter;
IIRFilterState    g_filter_state[VNS_NUM_CHANNELS];
FeatureExtractor  g_feature_extractor;
Classifier        g_classifier;


/* --- HAL handles (CubeMX-generated or manual) --- */
UART_HandleTypeDef  huart1;
PCD_HandleTypeDef   hpcd_USB_OTG_FS;
TIM_HandleTypeDef   htim6;

/* ================================================================== */
/*  Logger write callback ? non-blocking via ring buffer               */
/* ================================================================== */
static void logger_write_fn(const char *buf, uint32_t len, void *ctx)
{
    (void)ctx;
    RB_WriteMulti(&s_log_rb, (const uint8_t*)buf, len);
}

/* ================================================================== */
/*  Acquisition UART IDLE / DMA event callback (ISR context)          */
/* ================================================================== */
void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t size)
{
    if (huart->Instance == huart_acq.Instance) {
        if (size > 0) {
            RB_WriteMulti(&g_acq_rb, g_acq_dma.buffer, size);
        }
        g_acq_errors.rx_bytes += size;

        TaskHandle_t acq_task = xTaskGetHandle("ACQ");
        if (acq_task) {
            BaseType_t yield = pdFALSE;
            vTaskNotifyGiveFromISR(acq_task, &yield);
            portYIELD_FROM_ISR(yield);
        }

        HAL_UARTEx_ReceiveToIdle_DMA(huart, g_acq_dma.buffer, g_acq_dma.size);
    }
}

/* ================================================================== */
/*  TIM6 period callback ? 1 kHz tick for StimTask                    */
/* ================================================================== */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        TaskHandle_t stim_task = xTaskGetHandle("STIM");
        if (stim_task) {
            BaseType_t yield = pdFALSE;
            vTaskNotifyGiveFromISR(stim_task, &yield);
            portYIELD_FROM_ISR(yield);
        }
    }
}

/* ================================================================== */
/*  main() ? entry point                                              */
/* ================================================================== */
int main(void)
{
    HAL_Init();
    SystemClock_Config();

    MX_GPIO_Init();
    MX_DMA_Init();
    MX_USART1_UART_Init();
    MX_ACQ_UART_Init();
    MX_USB_OTG_FS_PCD_Init();
    MX_TIM6_Init();

    HAL_TIM_Base_Start_IT(&htim6);

    /* ---- Module initialisation ---- */
    {
        const VNS_Config *cfg = VNS_ConfigGetDefault();

        RB_Init(&s_log_rb, s_log_rb_storage, sizeof(s_log_rb_storage));

        float fs = (float)cfg->sample_rate;
        float hp = cfg->filter.highpass_cutoff / fs;
        float lp = cfg->filter.lowpass_cutoff  / fs;
        SP_DesignBandPass(&g_bp_filter, 4, hp, lp);
        for (int ch = 0; ch < VNS_NUM_CHANNELS; ch++)
            SP_InitState(&g_filter_state[ch]);

        FE_Init(&g_feature_extractor,
                cfg->feature.pre_samples,
                cfg->feature.post_samples,
                cfg->sample_rate);

        CL_Init(&g_classifier, cfg);

        Logger_Init(logger_write_fn, NULL);
        Logger_SetLevel(cfg->log.level);
    }

    /* ---- Error counters ---- */
    memset(&g_acq_errors, 0, sizeof(g_acq_errors));

    /* ---- DMA acquisition buffer ---- */
    g_acq_dma.buffer    = s_acq_dma_buf;
    g_acq_dma.size      = sizeof(s_acq_dma_buf);
    g_acq_dma.read_pos  = 0;
    g_acq_dma.write_pos = 0;
    g_acq_dma.huart     = &huart_acq;

    /* ---- Create FreeRTOS queues ---- */
        g_signalSampleQueue = xQueueCreate(QUEUE_LEN_SIGNAL_SAMPLE, sizeof(SignalSample_t));
    g_processedSampleQueue = xQueueCreate(QUEUE_LEN_PROCESSED_SAMPLE, sizeof(ProcessedSample_t));
    g_featureQueue = xQueueCreate(QUEUE_LEN_FEATURE, sizeof(FeatureSet_t));
    g_classifierResultQueue = xQueueCreate(QUEUE_LEN_CLASSIFIER_RESULT, sizeof(ClassifierResult_t));
    g_stimEventQueue = xQueueCreate(QUEUE_LEN_STIM_EVENT, sizeof(StimEvent_t));
    

    /* ---- Create data timeout semaphore ---- */
        g_dataTimeoutSem = xSemaphoreCreateBinary();

    /* ---- Start UART DMA circular reception ---- */
    HAL_UARTEx_ReceiveToIdle_DMA(&huart_acq, s_acq_dma_buf,
                                 sizeof(s_acq_dma_buf));

    /* ---- Create FreeRTOS tasks ---- */
    xTaskCreate(Task_Acquisition,  "ACQ",    TASK_STACK_ACQ,         NULL, TASK_PRIO_ACQ,   NULL);
    xTaskCreate(Task_Preprocess,   "PREPROC",TASK_STACK_PREPROCESS,  NULL, TASK_PRIO_PREPROCESS,NULL);
    xTaskCreate(Task_Feature,     "FEATURE",TASK_STACK_FEATURE,       NULL, TASK_PRIO_FEATURE,NULL);
    xTaskCreate(Task_Classifier,  "CLASS",  TASK_STACK_CLASSIFIER,   NULL, TASK_PRIO_CLASSIFIER,NULL);
    xTaskCreate(Task_StimControl, "STIM",   TASK_STACK_STIM,         NULL, TASK_PRIO_STIM,  NULL);
    xTaskCreate(Task_Telemetry,   "TELEM",  TASK_STACK_TELEMETRY,    NULL, TASK_PRIO_TELEMETRY,NULL);
    xTaskCreate(Task_Logger,      "LOGGER", TASK_STACK_LOGGER,       NULL, TASK_PRIO_LOGGER,NULL);
                NULL, TASK_PRIO_ACQ, NULL);
                NULL, TASK_PRIO_SIGNAL_PROC, NULL);
                NULL, TASK_PRIO_FEATURE, NULL);
    xTaskCreate(Task_Classifier,   "CLASS",  TASK_STACK_CLASSIFIER,
                NULL, TASK_PRIO_STIM, NULL);
                NULL, TASK_PRIO_LOGGER, NULL);

    /* ---- Start scheduler ---- */
    vTaskStartScheduler();

    while (1) { __NOP(); }
}

/* ================================================================== */
/*  Task: Acquisition ? UART DMA frame reception + validation         */
/* ================================================================== */
void Task_Acquisition(void *param)
{
    (void)param;
    FrameTracker tracker;
    FP_TrackerInit(&tracker);

    /* Sample accumulation buffer for downstream VNS_RawFrame */
    static float s_pressure_buf[VNS_SAMPLES_PER_FRAME];
    static float s_eus_buf[VNS_SAMPLES_PER_FRAME];
    static uint16_t s_buf_count = 0;
    static uint32_t s_frame_seq = 0;
    static uint8_t  s_work_buf[4096];

    const TickType_t timeout_check_ms = pdMS_TO_TICKS(50);

    for (;;)
    {
        /* Wait for notification from UART IDLE ISR, or periodic wake for timeout */
        ulTaskNotifyTake(pdTRUE, timeout_check_ms);

        /* ---- Data timeout check (500 ms) ---- */
        uint32_t now = HAL_GetTick();
        if ((now - g_acq_last_valid_tick) >= VNS_DATA_TIMEOUT_MS) {
            if (g_acq_last_valid_tick > 0) {
                /* Only signal once per timeout event */
                if (xSemaphoreGetCount(g_dataTimeoutSem) == 0) {
                    xSemaphoreGive(g_dataTimeoutSem);
                    LOGE("ACQ", "DATA TIMEOUT after %lu ms", (unsigned long)VNS_DATA_TIMEOUT_MS);
                }
            }
        }

                /* ---- Read available data from ring buffer ---- */
        uint32_t avail = RB_Available(&g_acq_rb);
        if (avail == 0) continue;
        if (avail > sizeof(s_work_buf)) avail = sizeof(s_work_buf);

        avail = RB_ReadMulti(&g_acq_rb, s_work_buf, avail);

        /* ---- Process work buffer for frames ---- */
        uint32_t offset = 0;
        while (offset + sizeof(VnsInputFrame_t) <= avail)
        {
            /* Find header */
            int hdr = FP_FindHeader(s_work_buf, avail, offset);
            if (hdr < 0) break;  /* No more headers */

            if ((uint32_t)hdr + sizeof(VnsInputFrame_t) > avail)
                break;  /* Incomplete frame -- wait for more data */

            VnsInputFrame_t *frame = (VnsInputFrame_t *)&s_work_buf[hdr];
            int ret = FP_ValidateFrame(frame, sizeof(VnsInputFrame_t));

            if (ret == FRAME_OK) {
                /* Sequence check */
                if (FP_CheckSequence(&tracker, frame->sequence) != 0) {
                    g_acq_errors.sequence_drops++;
                    LOGW("ACQ", "Seq drop: exp %lu got %lu",
                         (unsigned long)tracker.last_seq,
                         (unsigned long)frame->sequence);
                }

                /* Timestamp check */
                if (FP_CheckTimestamp(&tracker, frame->timestamp_us) != 0) {
                    g_acq_errors.timestamp_errors++;
                    LOGW("ACQ", "TS non-monotonic: %llu",
                         (unsigned long long)frame->timestamp_us);
                }

                /* Convert to internal sample */
                SignalSample_t sample;
                FP_ToSample(frame, &sample);
                xQueueSend(g_signalSampleQueue, &sample, 0);

                /* Accumulate into frame buffer */
                s_pressure_buf[s_buf_count] = sample.pressure_raw;
                s_eus_buf[s_buf_count]      = sample.eus_raw;
                s_buf_count++;

                if (s_buf_count >= VNS_SAMPLES_PER_FRAME) {
                    /* Build VNS_RawFrame for downstream processing */
                    VNS_RawFrame rf;
                    rf.seq          = s_frame_seq++;
                    rf.timestamp_us = sample.timestamp_us;
                    rf.chan_mask    = 0x03;
                    rf.actual_count = VNS_SAMPLES_PER_FRAME;

                    for (uint16_t i = 0; i < VNS_SAMPLES_PER_FRAME; i++) {
                        int v = (int)(s_pressure_buf[i] * 32768.0f);
                        if (v < 0) v = 0; if (v > 65535) v = 65535;
                        rf.samples[VNS_CHAN_A][i] = (uint16_t)v;

                        v = (int)(s_eus_buf[i] * 32768.0f);
                        if (v < 0) v = 0; if (v > 65535) v = 65535;
                        rf.samples[VNS_CHAN_B][i] = (uint16_t)v;
                    }

                    if (xQueueSend(g_rawFrameQueue, &rf, 0) != pdTRUE) {
                        g_acq_errors.queue_overflows++;
                    }
                    s_buf_count = 0;
                }

                g_acq_errors.valid_frames++;
                g_acq_last_valid_tick = HAL_GetTick();
                offset = hdr + (int)sizeof(VnsInputFrame_t);

            } else {
                /* Validation failed */
                switch (ret) {
                case FRAME_ERR_BAD_HEADER: /* fallthrough */
                case FRAME_ERR_BAD_VERSION:
                case FRAME_ERR_BAD_LENGTH:
                    g_acq_errors.length_errors++;
                    break;
                case FRAME_ERR_BAD_CRC:
                    g_acq_errors.crc_errors++;
                    break;
                }
                offset = hdr + 2;  /* Skip 2 bytes past suspect header */
                LOGD("ACQ", "Frame validation err %d at offset %lu",
                     ret, (unsigned long)offset);
            }
        }
    }
}

/* ================================================================== */
/*  Task: SignalProc ? filter + event detection                       */
/*  Receives VNS_RawFrame from AcquisitionTask accumulation           */
/* ================================================================== */
void Task_SignalProc(void *param)
{
    (void)param;
    static float s_buf_a[VNS_SAMPLES_PER_FRAME];
    static float s_buf_b[VNS_SAMPLES_PER_FRAME];
    const VNS_Config *cfg = VNS_ConfigGet();

    for (;;) {
        VNS_RawFrame frame;
        if (xQueueReceive(g_rawFrameQueue, &frame, portMAX_DELAY) != pdTRUE)
            continue;

        LOGD("SIG", "frame seq=%lu, samples=%u",
             (unsigned long)frame.seq, (unsigned)frame.actual_count);

        /* Channel A (pressure) */
        if (frame.chan_mask & 0x01) {
            for (uint16_t i = 0; i < frame.actual_count; i++) {
                s_buf_a[i] = ((int)frame.samples[VNS_CHAN_A][i] - 32768) / 32768.0f;
            }
            SP_ProcessBlock(&g_bp_filter, &g_filter_state[VNS_CHAN_A],
                            s_buf_a, s_buf_a, frame.actual_count);

            uint32_t ev_start, ev_end;
            if (SP_DetectEvent(s_buf_a, frame.actual_count,
                               cfg->event.threshold_a,
                               cfg->event.min_event_samples,
                               &ev_start, &ev_end) > 0) {
                VNSEvent ev;
                ev.timestamp_us   = frame.timestamp_us;
                ev.channel         = VNS_CHAN_A;
                ev.sample_index    = ev_start;
                ev.peak_amplitude  = 0.0f;
                ev.rms             = 0.0f;
                ev.zero_cross_rate = 0.0f;
                ev.pulse_width_ms  = 0.0f;
                ev.energy          = 0.0f;

                float sum_sq = 0.0f, max_v = 0.0f;
                for (uint32_t j = ev_start; j < ev_end; j++) {
                    float v = fabsf(s_buf_a[j]);
                    if (v > max_v) max_v = v;
                    sum_sq += v * v;
                }
                ev.peak_amplitude = max_v;
                ev.rms = sqrtf(sum_sq / (float)(ev_end - ev_start));

                LOGI("SIG", "ChA event @ %lu amp=%.3f",
                     (unsigned long)ev_start, (double)ev.peak_amplitude);
                xQueueSend(g_eventQueue, &ev, 0);
            }
        }

        /* Channel B (EUS envelope) */
        if (frame.chan_mask & 0x02) {
            for (uint16_t i = 0; i < frame.actual_count; i++) {
                s_buf_b[i] = ((int)frame.samples[VNS_CHAN_B][i] - 32768) / 32768.0f;
            }
            SP_ProcessBlock(&g_bp_filter, &g_filter_state[VNS_CHAN_B],
                            s_buf_b, s_buf_b, frame.actual_count);

            uint32_t ev_start, ev_end;
            if (SP_DetectEvent(s_buf_b, frame.actual_count,
                               cfg->event.threshold_b,
                               cfg->event.min_event_samples,
                               &ev_start, &ev_end) > 0) {
                VNSEvent ev;
                ev.timestamp_us   = frame.timestamp_us;
                ev.channel         = VNS_CHAN_B;
                ev.sample_index    = ev_start;
                ev.peak_amplitude  = 0.0f;
                ev.rms             = 0.0f;
                ev.zero_cross_rate = 0.0f;
                ev.pulse_width_ms  = 0.0f;
                ev.energy          = 0.0f;

                float sum_sq = 0.0f, max_v = 0.0f;
                for (uint32_t j = ev_start; j < ev_end; j++) {
                    float v = fabsf(s_buf_b[j]);
                    if (v > max_v) max_v = v;
                    sum_sq += v * v;
                }
                ev.peak_amplitude = max_v;
                ev.rms = sqrtf(sum_sq / (float)(ev_end - ev_start));

                LOGI("SIG", "ChB event @ %lu amp=%.3f",
                     (unsigned long)ev_start, (double)ev.peak_amplitude);
                xQueueSend(g_eventQueue, &ev, 0);
            }
        }
    }
}

/* ================================================================== */
/*  Task: FeatureExtract                                              */
/* ================================================================== */


/* ================================================================== */
/*  Task: Classifier ? rule-based 3-class + stim command              */
/* ================================================================== */


/* ================================================================== */
/*  Task: Stim ? stimulation trigger + data timeout handling          */
/* ================================================================== */


/* ================================================================== */
/*  Task: Logger ? ring buffer ? UART DMA                             */
/* ================================================================== */
void Task_Logger(void *param)
{
    (void)param;
    uint8_t buf[128];

    for (;;) {
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(50));

        uint32_t n = RB_ReadMulti(&s_log_rb, buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            HAL_UART_Transmit_DMA(&huart1, buf, n);
        }
    }
}

/* ================================================================== */
/* ================================================================== */
/*  Task: Preprocess -- real-time signal conditioning                  */
/* ================================================================== */
void Task_Preprocess(void *param)
{
    (void)param;
    const SignalConfig_t *scfg = VNS_SignalConfigGetDefault();
    PressureProcessor  pp;
    EUSProcessor       ep;
    TimestampMonitor   tm;
    SignalSample_t     in;
    ProcessedSample_t  out;
    PP_Init(&pp, scfg, 100.0f);
    EP_Init(&ep, scfg, 100.0f);
    TM_Init(&tm, scfg);
    for (;;) {
        if (xQueueReceive(g_signalSampleQueue, &in, portMAX_DELAY) != pdTRUE)
            continue;
        out.sequence = in.sequence;
        out.timestamp_us = in.timestamp_us;
        out.quality_flags = TM_Check(&tm, in.timestamp_us)
                          | (in.quality_flags & (QUALITY_CLIPPED | QUALITY_OVERRANGE));
        float filtered;
        PP_Process(&pp, in.pressure_raw, &filtered,
                   &out.pressure_baseline_removed,
                   &out.pressure_derivative);
        out.pressure_filtered = filtered;
        float eus_out;
        EP_Process(&ep, in.eus_raw, &eus_out, NULL);
        out.eus_envelope = eus_out;
        if (xQueueSend(g_processedSampleQueue, &out, 0) != pdTRUE) {
            LOGW("PREP", "processedSampleQueue full");
        }
    }
}
/* ================================================================== */
/*  Task: Feature ? accumulate & compute 2 s window features          */
/* ================================================================== */
void Task_Feature(void *param)
{
    (void)param;
    FeatureExtractor fe;
    FE_Init(&fe, (float)VNS_SAMPLE_RATE);
    ProcessedSample_t in;
    FeatureSet_t out;
    uint32_t count = 0;

    for (;;) {
        if (xQueueReceive(g_processedSampleQueue, &in, portMAX_DELAY) != pdTRUE)
            continue;
        count++;
        FE_FeedSample(&fe, in.pressure_filtered, in.eus_envelope,
                      in.timestamp_us, in.quality_flags);
        if (in.quality_flags & QUALITY_SIGNAL_INVALID) {
            out.window_ready = 0;
            xQueueSend(g_featureQueue, &out, 0);
            FE_Reset(&fe);
            continue;
        }
        if (count % VNS_FEATURE_UPDATE_STEP == 0) {
            FE_Compute(&fe, &out);
            xQueueSend(g_featureQueue, &out, 0);
        }
    }
}

/* ================================================================== */
/*  Task: Classifier ? rule-based 3-class                             */
/* ================================================================== */
void Task_Classifier(void *param)
{
    (void)param;
    ClassifierState st;
    const ClassifierConfig_t *cfg = CL_GetDefaultConfig();
    CL_Init(&st);
    FeatureSet_t fs;
    ClassifierResult_t cr;

    for (;;) {
        if (xQueueReceive(g_featureQueue, &fs, portMAX_DELAY) != pdTRUE)
            continue;
        CL_Classify(&st, &fs, cfg, HAL_GetTick(), &cr);
        if (cr.changed) {
            xQueueSend(g_classifierResultQueue, &cr, 0);
        }
    }
}

/* ================================================================== */
/*  Task: StimControl ? state machine with safety interlocks          */
/* ================================================================== */
static void stim_gpio_cb(StimControl *s, uint8_t v) { (void)s; HAL_GPIO_WritePin(GPIOA, GPIO_PIN_0, v ? GPIO_PIN_SET : GPIO_PIN_RESET); }
void Task_StimControl(void *param)
{
    (void)param;
    SC_Init(&g_stim, SC_GetDefaultConfig(), stim_gpio_cb, NULL, NULL, NULL);
    ClassifierResult_t cr;

    for (;;) {
        if (xQueueReceive(g_classifierResultQueue, &cr, pdMS_TO_TICKS(20)) != pdTRUE)
            continue;
        uint32_t now = HAL_GetTick();
        SC_Process(&g_stim, cr.class_id, cr.class_id != CLASS_INVALID, now);

        if (cr.class_id == CLASS2_VOIDING && cr.changed) {
            SC_Command(&g_stim, STIM_CMD_DISARM, 0, 0);
            LOGW("STIM", "Class2 => disarmed");
        }
        if (cr.class_id == CLASS_INVALID) {
            SC_Command(&g_stim, STIM_CMD_DISARM, 0, 0);
        }
    }
}

/* ================================================================== */
/*  Task: Telemetry ? format and send via UART DMA                    */
/* ================================================================== */
void Task_Telemetry(void *param)
{
    (void)param;
    uint8_t buf[256];
    TelemetryConfig_t tcfg = *TL_GetDefaultConfig();
    tcfg.enable_sample_output = false;
    TL_Init(&g_telemetry, &tcfg);

    for (;;) {
        ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(50));
        uint32_t n = TL_ReadFrame(&g_telemetry, buf, sizeof(buf));
        if (n > 0) {
            HAL_UART_Transmit_DMA(&huart1, buf, n);
        }
    }
}

/*  Hardware initialisation (CubeMX-generated; manual stubs below)    */
/* ================================================================== */

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState       = RCC_HSE_ON;
    RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM       = 3;
    RCC_OscInitStruct.PLL.PLLN       = 75;
    RCC_OscInitStruct.PLL.PLLP       = 2;
    RCC_OscInitStruct.PLL.PLLQ       = 2;
    RCC_OscInitStruct.PLL.PLLR       = 2;
    HAL_RCC_OscConfig(&RCC_OscInitStruct);

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_10);
}

/* --- Debug UART: USART1 (PA9=TX, PA10=RX, 115200 baud) --- */
void MX_USART1_UART_Init(void)
{
    huart1.Instance          = USART1;
    huart1.Init.BaudRate     = 115200;
    huart1.Init.WordLength   = UART_WORDLENGTH_8B;
    huart1.Init.StopBits     = UART_STOPBITS_1;
    huart1.Init.Parity       = UART_PARITY_NONE;
    huart1.Init.Mode         = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    HAL_UART_Init(&huart1);
}

/* --- Acquisition UART (e.g. UART4, configured via CubeMX) --- */
void MX_ACQ_UART_Init(void)
{
    const VNS_Config *cfg = VNS_ConfigGet();
    huart_acq.Instance          = UART4;        /* Change per CubeMX */
    huart_acq.Init.BaudRate     = cfg->acq_uart.baudrate;
    huart_acq.Init.WordLength   = UART_WORDLENGTH_8B;
    huart_acq.Init.StopBits     = UART_STOPBITS_1;
    huart_acq.Init.Parity       = UART_PARITY_NONE;
    huart_acq.Init.Mode         = UART_MODE_TX_RX;
    huart_acq.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
    huart_acq.Init.OverSampling = UART_OVERSAMPLING_16;
    HAL_UART_Init(&huart_acq);
}

void MX_USB_OTG_FS_PCD_Init(void)
{
    hpcd_USB_OTG_FS.Instance             = USB_OTG_FS;
    hpcd_USB_OTG_FS.Init.dev_endpoints   = 6;
    hpcd_USB_OTG_FS.Init.speed           = PCD_SPEED_FULL;
    hpcd_USB_OTG_FS.Init.dma_enable      = DISABLE;
    hpcd_USB_OTG_FS.Init.phy_itface      = PCD_PHY_EMBEDDED;
    hpcd_USB_OTG_FS.Init.Sof_enable      = DISABLE;
    hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
    hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
    hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
    HAL_PCD_Init(&hpcd_USB_OTG_FS);
}

void MX_TIM6_Init(void)
{
    htim6.Instance               = TIM6;
    htim6.Init.Prescaler         = 100 - 1;
    htim6.Init.CounterMode       = TIM_COUNTERMODE_UP;
    htim6.Init.Period            = 1000 - 1;
    htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
    HAL_TIM_Base_Init(&htim6);
}

void MX_GPIO_Init(void)
{
    GPIO_InitTypeDef gpio = {0};

    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE();

    /* Stim output: PA0 */
    gpio.Pin   = GPIO_PIN_0;
    gpio.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio.Pull  = GPIO_NOPULL;
    gpio.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOA, &gpio);

    /* USB power: PC5 */
    gpio.Pin   = GPIO_PIN_5;
    gpio.Mode  = GPIO_MODE_OUTPUT_PP;
    HAL_GPIO_Init(GPIOC, &gpio);
    HAL_GPIO_WritePin(GPIOC, GPIO_PIN_5, GPIO_PIN_SET);

    /* User button: PC13 */
    gpio.Pin   = GPIO_PIN_13;
    gpio.Mode  = GPIO_MODE_INPUT;
    gpio.Pull  = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOC, &gpio);
}

void MX_DMA_Init(void)
{
    __HAL_RCC_DMAMUX1_CLK_ENABLE();
    __HAL_RCC_DMA1_CLK_ENABLE();
    HAL_NVIC_SetPriority(DMA1_Channel1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(DMA1_Channel1_IRQn);
}

/* ================================================================== */
/*  FreeRTOS hooks                                                    */
/* ================================================================== */

void vApplicationMallocFailedHook(void)
{
    LOGE("FREERTOS", "malloc failed!");
    while (1) { __NOP(); }
}

void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    (void)xTask;
    LOGE("FREERTOS", "stack overflow: %s", pcTaskName ? pcTaskName : "?");
    while (1) { __NOP(); }
}

void vApplicationIdleHook(void) {}

/* ================================================================== */
/*  HAL MSP init (CubeMX-generated template)                          */
/* ================================================================== */

void HAL_UART_MspInit(UART_HandleTypeDef *huart)
{
    GPIO_InitTypeDef gpio = {0};
    if (huart->Instance == USART1) {
        __HAL_RCC_USART1_CLK_ENABLE();
        __HAL_RCC_GPIOA_CLK_ENABLE();
        gpio.Pin       = GPIO_PIN_9 | GPIO_PIN_10;
        gpio.Mode      = GPIO_MODE_AF_PP;
        gpio.Pull      = GPIO_NOPULL;
        gpio.Speed     = GPIO_SPEED_FREQ_HIGH;
        gpio.Alternate = GPIO_AF7_USART1;
        HAL_GPIO_Init(GPIOA, &gpio);
        HAL_NVIC_SetPriority(USART1_IRQn, 5, 0);
        HAL_NVIC_EnableIRQ(USART1_IRQn);
    }
    if (huart->Instance == UART4) {
        __HAL_RCC_UART4_CLK_ENABLE();
        __HAL_RCC_GPIOB_CLK_ENABLE();
        /* UART4 TX=PB9, RX=PB8 (NUCLEO-N657X0-Q default) */
        gpio.Pin       = GPIO_PIN_8 | GPIO_PIN_9;
        gpio.Mode      = GPIO_MODE_AF_PP;
        gpio.Pull      = GPIO_NOPULL;
        gpio.Speed     = GPIO_SPEED_FREQ_HIGH;
        gpio.Alternate = GPIO_AF8_UART4;    /* Verify per datasheet */
        HAL_GPIO_Init(GPIOB, &gpio);

        /* DMA for RX */
        __HAL_RCC_DMA1_CLK_ENABLE();
        HAL_NVIC_SetPriority(DMA1_Channel3_IRQn, 5, 0);
        HAL_NVIC_EnableIRQ(DMA1_Channel3_IRQn);
        HAL_NVIC_SetPriority(UART4_IRQn, 5, 0);
        HAL_NVIC_EnableIRQ(UART4_IRQn);
    }
}

void HAL_UART_MspDeInit(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) {
        __HAL_RCC_USART1_CLK_DISABLE();
        HAL_GPIO_DeInit(GPIOA, GPIO_PIN_9 | GPIO_PIN_10);
        HAL_NVIC_DisableIRQ(USART1_IRQn);
    }
    if (huart->Instance == UART4) {
        __HAL_RCC_UART4_CLK_DISABLE();
        HAL_GPIO_DeInit(GPIOB, GPIO_PIN_8 | GPIO_PIN_9);
        HAL_NVIC_DisableIRQ(UART4_IRQn);
    }
}

void HAL_TIM_Base_MspInit(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM6) {
        __HAL_RCC_TIM6_CLK_ENABLE();
        HAL_NVIC_SetPriority(TIM6_IRQn, 5, 0);
        HAL_NVIC_EnableIRQ(TIM6_IRQn);
    }
}

void HAL_PCD_MspInit(PCD_HandleTypeDef *hpcd)
{
    GPIO_InitTypeDef gpio = {0};
    if (hpcd->Instance == USB_OTG_FS) {
        __HAL_RCC_USB_OTG_FS_CLK_ENABLE();
        __HAL_RCC_GPIOA_CLK_ENABLE();
        gpio.Pin       = GPIO_PIN_11 | GPIO_PIN_12;
        gpio.Mode      = GPIO_MODE_AF_PP;
        gpio.Pull      = GPIO_NOPULL;
        gpio.Speed     = GPIO_SPEED_FREQ_HIGH;
        gpio.Alternate = GPIO_AF10_OTG1_FS;
        HAL_GPIO_Init(GPIOA, &gpio);
        gpio.Pin       = GPIO_PIN_10;
        gpio.Mode      = GPIO_MODE_AF_OD;
        gpio.Pull      = GPIO_PULLUP;
        gpio.Alternate = GPIO_AF10_OTG1_FS;
        HAL_GPIO_Init(GPIOA, &gpio);
        HAL_NVIC_SetPriority(OTG_FS_IRQn, 5, 0);
        HAL_NVIC_EnableIRQ(OTG_FS_IRQn);
    }
}





