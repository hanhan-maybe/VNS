#ifndef V5_SUBJECT_CONFIG_H
#define V5_SUBJECT_CONFIG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "v5_candidate.h"
#include "v5_runtime.h"

#define V5_SUBJECT_CONFIG_MAGIC 0x43533556u
#define V5_SUBJECT_CONFIG_VERSION 1u
#define V5_SUBJECT_CONFIG_FEATURE_HASH 0x215EAC03u
#define V5_SUBJECT_CONFIG_BINARY_SIZE 244u

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t feature_count;
    uint32_t feature_order_hash;
    uint8_t model_hash[32];
    V5ModelConfig model;
    float candidate_prior_sigma_p;
    float candidate_prior_sigma_dpdt;
    uint32_t crc32;
} V5SubjectConfig;

uint32_t V5SubjectConfig_Crc32(const uint8_t *data, size_t length);
bool V5SubjectConfig_Decode(V5SubjectConfig *out, const uint8_t *data, size_t length);
bool V5SubjectConfig_Validate(const V5SubjectConfig *config);
bool V5Runtime_LoadSubjectConfig(V5Runtime *runtime, const V5SubjectConfig *config);
bool V5SubjectConfig_ApplyCandidate(const V5SubjectConfig *config,
                                    V5CandidateConfig *candidate);

#endif
