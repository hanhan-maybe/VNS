
/**
 * @file    ring_buffer.c
 * @brief   无锁环形缓冲区实现 (单生产者 / 单消费者)
 */
#include "ring_buffer.h"

/* 验证 size 为 2 的幂 */
static inline int is_pow2(uint32_t v) { return v && !(v & (v - 1)); }

int RB_Init(RingBuffer *rb, uint8_t *storage, uint32_t size)
{
    if (!rb || !storage || !is_pow2(size)) return -1;
    rb->buffer = storage;
    rb->size   = size;
    rb->mask   = size - 1;
    rb->head   = 0;
    rb->tail   = 0;
    return 0;
}

void RB_Reset(RingBuffer *rb)
{
    if (!rb) return;
    rb->head = 0;
    rb->tail = 0;
}

int RB_WriteByte(RingBuffer *rb, uint8_t byte)
{
    if (!rb) return -1;
    volatile uint32_t next = (rb->head + 1) & rb->mask;
    if (next == rb->tail) return -1;  /* 满 */
    rb->buffer[rb->head] = byte;
    rb->head = next;
    return 0;
}

uint32_t RB_WriteMulti(RingBuffer *rb, const uint8_t *data, uint32_t len)
{
    if (!rb || !data || len == 0) return 0;
    uint32_t written = 0;
    for (uint32_t i = 0; i < len; i++) {
        if (RB_WriteByte(rb, data[i]) != 0) break;
        written++;
    }
    return written;
}

int RB_ReadByte(RingBuffer *rb, uint8_t *out)
{
    if (!rb || !out) return -1;
    if (rb->head == rb->tail) return -1;  /* 空 */
    *out = rb->buffer[rb->tail];
    rb->tail = (rb->tail + 1) & rb->mask;
    return 0;
}

uint32_t RB_ReadMulti(RingBuffer *rb, uint8_t *out, uint32_t max_len)
{
    if (!rb || !out || max_len == 0) return 0;
    uint32_t read_count = 0;
    for (uint32_t i = 0; i < max_len; i++) {
        if (RB_ReadByte(rb, &out[i]) != 0) break;
        read_count++;
    }
    return read_count;
}

int RB_Peek(const RingBuffer *rb, uint8_t *out)
{
    if (!rb || !out) return -1;
    if (rb->head == rb->tail) return -1;
    *out = rb->buffer[rb->tail];
    return 0;
}

uint32_t RB_Available(const RingBuffer *rb)
{
    if (!rb) return 0;
    return (rb->head - rb->tail) & rb->mask;
}

uint32_t RB_FreeSpace(const RingBuffer *rb)
{
    if (!rb) return 0;
    return (rb->size - 1) - ((rb->head - rb->tail) & rb->mask);
}

uint32_t RB_Capacity(const RingBuffer *rb)
{
    return rb ? (rb->size - 1) : 0;
}
