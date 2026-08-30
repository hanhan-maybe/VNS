
/**
 * @file    ring_buffer.h
 * @brief   无锁环形缓冲区 — 单生产者 (ISR) / 单消费者 (Task)
 *
 * 适用于 ISR → Task 的字节流传递。
 * head (写指针) 仅由生产者更新；tail (读指针) 仅由消费者更新。
 * 容量必须为 2 的幂。
 */
#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include "vns_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  类型                                                              */
/* ------------------------------------------------------------------ */
typedef struct {
    uint8_t         *buffer;       /**< 存储区                     */
    volatile uint32_t head;        /**< 写索引 (生产者更新)        */
    volatile uint32_t tail;        /**< 读索引 (消费者更新)        */
    uint32_t         size;         /**< 总容量 (2^n)              */
    uint32_t         mask;         /**< size - 1                  */
} RingBuffer;

/* ------------------------------------------------------------------ */
/*  API                                                               */
/* ------------------------------------------------------------------ */

/**
 * @brief  初始化环形缓冲区
 * @param  rb       RingBuffer 指针
 * @param  storage  预先分配的 uint8_t 存储区
 * @param  size     存储区大小 (必须为 2 的幂)
 * @retval 0        成功
 * @retval -1       size 不是 2 的幂
 */
int  RB_Init(RingBuffer *rb, uint8_t *storage, uint32_t size);

/** @brief  重置缓冲区 (丢弃所有数据) */
void RB_Reset(RingBuffer *rb);

/**
 * @brief  生产者写入一个字节 (ISR 安全)
 * @retval 0  成功
 * @retval -1 缓冲区满
 */
int  RB_WriteByte(RingBuffer *rb, uint8_t byte);

/**
 * @brief  生产者写入多个字节 (ISR 安全)
 * @param  rb   缓冲区
 * @param  data 数据指针
 * @param  len  字节数
 * @return 实际写入的字节数 (剩余空间不足时可能 < len)
 */
uint32_t RB_WriteMulti(RingBuffer *rb, const uint8_t *data, uint32_t len);

/**
 * @brief  消费者读取一个字节 (Task 使用)
 * @param  rb  缓冲区
 * @param  out 输出字节
 * @retval 0  成功
 * @retval -1 缓冲区空
 */
int  RB_ReadByte(RingBuffer *rb, uint8_t *out);

/**
 * @brief  消费者读取多个字节 (Task 使用)
 * @return 实际读取的字节数
 */
uint32_t RB_ReadMulti(RingBuffer *rb, uint8_t *out, uint32_t max_len);

/**
 * @brief  消费者查看下一个字节但不移除 (peek)
 * @retval 0  成功
 * @retval -1 缓冲区空
 */
int  RB_Peek(const RingBuffer *rb, uint8_t *out);

/** @brief 返回当前可用字节数 (消费者视角) */
uint32_t RB_Available(const RingBuffer *rb);

/** @brief 返回剩余可写字节数 (生产者视角) */
uint32_t RB_FreeSpace(const RingBuffer *rb);

/** @brief 返回容量 */
uint32_t RB_Capacity(const RingBuffer *rb);

#ifdef __cplusplus
}
#endif

#endif /* RING_BUFFER_H */
