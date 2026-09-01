#include "v5_subject_config.h"

#include <math.h>
#include <string.h>

static uint16_t read_u16(const uint8_t **p)
{
    uint16_t value=(uint16_t)(*p)[0]|((uint16_t)(*p)[1]<<8);*p+=2;return value;
}
static uint32_t read_u32(const uint8_t **p)
{
    uint32_t value=(uint32_t)(*p)[0]|((uint32_t)(*p)[1]<<8)|
        ((uint32_t)(*p)[2]<<16)|((uint32_t)(*p)[3]<<24);*p+=4;return value;
}
static float read_f32(const uint8_t **p)
{
    uint32_t bits=read_u32(p);float value;memcpy(&value,&bits,sizeof(value));return value;
}

uint32_t V5SubjectConfig_Crc32(const uint8_t *data,size_t length)
{
    uint32_t crc=0xFFFFFFFFu;size_t i;unsigned bit;
    if(data==NULL)return 0u;
    for(i=0;i<length;++i){crc^=data[i];for(bit=0;bit<8u;++bit)
        crc=(crc>>1)^((crc&1u)?0xEDB88320u:0u);}return crc^0xFFFFFFFFu;
}

bool V5SubjectConfig_Decode(V5SubjectConfig *out,const uint8_t *data,size_t length)
{
    const uint8_t*p=data;size_t i;uint32_t stored,actual;
    if(out==NULL||data==NULL||length!=V5_SUBJECT_CONFIG_BINARY_SIZE)return false;
    actual=V5SubjectConfig_Crc32(data,length-4u);stored=(uint32_t)data[length-4u]|
        ((uint32_t)data[length-3u]<<8)|((uint32_t)data[length-2u]<<16)|
        ((uint32_t)data[length-1u]<<24);if(actual!=stored)return false;
    memset(out,0,sizeof(*out));out->magic=read_u32(&p);out->version=read_u16(&p);
    out->feature_count=read_u16(&p);out->feature_order_hash=read_u32(&p);
    memcpy(out->model_hash,p,32u);p+=32;
    for(i=0;i<V5_MODEL_FEATURE_COUNT;++i)out->model.mean[i]=read_f32(&p);
    for(i=0;i<V5_MODEL_FEATURE_COUNT;++i)out->model.scale[i]=read_f32(&p);
    for(i=0;i<V5_MODEL_FEATURE_COUNT;++i)out->model.coef[i]=read_f32(&p);
    out->model.intercept=read_f32(&p);out->model.threshold=read_f32(&p);
    out->candidate_prior_sigma_p=read_f32(&p);
    out->candidate_prior_sigma_dpdt=read_f32(&p);out->crc32=stored;
    return V5SubjectConfig_Validate(out);
}

bool V5SubjectConfig_Validate(const V5SubjectConfig *config)
{
    size_t i;bool any_hash=false;
    if(config==NULL||config->magic!=V5_SUBJECT_CONFIG_MAGIC||
       config->version!=V5_SUBJECT_CONFIG_VERSION||
       config->feature_count!=V5_MODEL_FEATURE_COUNT||
       config->feature_order_hash!=V5_SUBJECT_CONFIG_FEATURE_HASH||
       !V5Model_ValidateConfig(&config->model)||
       !isfinite(config->candidate_prior_sigma_p)||config->candidate_prior_sigma_p<=0.0f||
       !isfinite(config->candidate_prior_sigma_dpdt)||config->candidate_prior_sigma_dpdt<=0.0f)
        return false;
    for(i=0;i<sizeof(config->model_hash);++i)if(config->model_hash[i]!=0u)any_hash=true;
    return any_hash;
}

bool V5Runtime_LoadSubjectConfig(V5Runtime *runtime,const V5SubjectConfig *config)
{
    return V5SubjectConfig_Validate(config)&&V5Runtime_LoadModel(runtime,&config->model);
}

bool V5SubjectConfig_ApplyCandidate(const V5SubjectConfig *config,V5CandidateConfig *candidate)
{
    if(!V5SubjectConfig_Validate(config)||candidate==NULL)return false;
    candidate->prior_sigma_p=config->candidate_prior_sigma_p;
    candidate->prior_sigma_dpdt=config->candidate_prior_sigma_dpdt;return true;
}
