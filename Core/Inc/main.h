/**
 * @file    main.h
 * @brief   VNS on NUCLEO-N657X0-Q - FreeRTOS integration header
 */
#ifndef MAIN_H
#define MAIN_H

#include "vns_types.h"
#include "vns_config.h"
#include "ring_buffer.h"
#include "frame_protocol.h"
#include "signal_processing.h"
#include "feature_extraction.h"
#include "classifier.h"

#include "logger.h"

#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "timers.h"
#include "semphr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Task priorities (higher = more urgent)                            */
/* ------------------------------------------------------------------ */
#define TASK_PRIO_ACQ         6   /* UART data acquisition (highest) */
#define TASK_PRIO_FEATURE     4
#define TASK_STACK_TELEMETRY 256
#define TASK_PRIO_TELEMETRY   1
#define TASK_PRIO_FEATURE     3
#define TASK_STACK_TELEMETRY 256
#define TASK_PRIO_LOGGER      1
#define TASK_PRIO_PREPROCESS  4

/* ------------------------------------------------------------------ */
/*  Task stack sizes (words)                                          */
/* ------------------------------------------------------------------ */
#define TASK_STACK_ACQ         512
#define TASK_STACK_CLASSIFIER 256
#define TASK_STACK_STIM       256
#define TASK_STACK_PREPROCESS  512
#define TASK_STACK_LOGGER     256
#define TASK_STACK_TELEMETRY  256
#define TASK_PRIO_FEATURE     4
#define TASK_PRIO_TELEMETRY   1

/* ------------------------------------------------------------------ */
/*  Queue lengths                                                     */
/* ------------------------------------------------------------------ */
#define QUEUE_LEN_SIGNAL_SAMPLE 32   /* SignalSample_t from AcquisitionTask */
#define QUEUE_LEN_FEATURE         16   /* FeatureSet_t  */
#define QUEUE_LEN_CLASSIFIER_RESULT  8    /* ClassifierResult_t */
#define QUEUE_LEN_STIM_EVENT       8    /* StimEvent_t       */
#define QUEUE_LEN_PROCESSED_SAMPLE 32   /* ProcessedSample_t */
#define QUEUE_LEN_SIGNAL_SAMPLE    32
#define QUEUE_LEN_FEATURE      8    /* VNS_FeatureVector */
#define QUEUE_LEN_PROCESSED_SAMPLE 32   /* ProcessedSample_t */

/* ------------------------------------------------------------------ */
/*  Queue handle externs                                              */
/* ------------------------------------------------------------------ */
extern QueueHandle_t g_signalSampleQueue;
extern QueueHandle_t g_signalSampleQueue;
extern QueueHandle_t g_processedSampleQueue;
extern QueueHandle_t g_featureQueue;
extern QueueHandle_t g_classifierResultQueue;
extern QueueHandle_t g_stimEventQueue;

/* ------------------------------------------------------------------ */
/*  DMA circular buffer for UART acquisition                          */
/* ------------------------------------------------------------------ */
typedef struct {
    UART_HandleTypeDef *huart;
    uint8_t            *buffer;
    uint32_t            size;
    volatile uint32_t   write_pos;    /* Updated in UART IDLE ISR */
    uint32_t            read_pos;     /* Updated in AcquisitionTask */
} DmaRxBuffer;

extern DmaRxBuffer        g_acq_dma;
extern UART_HandleTypeDef huart_acq;  /* Acquisition UART handle */

/* ------------------------------------------------------------------ */
/*  Error counters and data timeout                                   */
/* ------------------------------------------------------------------ */
extern VNSErrorCounters   g_acq_errors;
extern SemaphoreHandle_t  g_dataTimeoutSem;
extern RingBuffer         g_acq_rb;         /* Acquisition UART ring buffer */
extern uint32_t           g_acq_last_valid_tick;

/* ------------------------------------------------------------------ */
/*  Module instances                                                  */
/* ------------------------------------------------------------------ */
extern IIRFilter         g_bp_filter;
extern IIRFilterState    g_filter_state[VNS_NUM_CHANNELS];
extern FeatureExtractor  g_feature_extractor;
extern Classifier        g_classifier;


/* ------------------------------------------------------------------ */
/*  Hardware init (CubeMX-generated or manual)                        */
/* ------------------------------------------------------------------ */
void SystemClock_Config(void);
void MX_GPIO_Init(void);
void MX_USART1_UART_Init(void);
void MX_ACQ_UART_Init(void);        /* Acquisition UART (e.g. UART4) */
void MX_USB_OTG_FS_PCD_Init(void);
void MX_DMA_Init(void);
void MX_TIM6_Init(void);

/* ------------------------------------------------------------------ */
/*  Task entry points                                                 */
/* ------------------------------------------------------------------ */
void Task_Acquisition(void *param);
void Task_Acquisition(void *param);
void Task_Preprocess(void *param);
void Task_Feature(void *param);
void Task_Classifier(void *param);
void Task_StimControl(void *param);
void Task_Telemetry(void *param);
void Task_Logger(void *param);

#ifdef __cplusplus
}
#endif

#endif /* MAIN_H */




