#include "v5_hil_protocol.h"

#include <string.h>

static uint16_t get_u16(const uint8_t*p){return(uint16_t)p[0]|((uint16_t)p[1]<<8);}
static uint32_t get_u32(const uint8_t*p){return(uint32_t)p[0]|((uint32_t)p[1]<<8)|
    ((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);}
static float get_f32(const uint8_t*p){uint32_t x=get_u32(p);float f;memcpy(&f,&x,4);return f;}
static void put_u32(uint8_t*p,uint32_t x){p[0]=(uint8_t)x;p[1]=(uint8_t)(x>>8);
    p[2]=(uint8_t)(x>>16);p[3]=(uint8_t)(x>>24);}
static void put_f32(uint8_t*p,float f){uint32_t x;memcpy(&x,&f,4);put_u32(p,x);}

bool V5Hil_DecodePressureFrame(V5HilPressureFrame*out,const uint8_t*bytes,size_t length)
{
    uint32_t crc;if(out==NULL||bytes==NULL||length!=V5_HIL_PRESSURE_FRAME_SIZE||
        get_u32(bytes)!=V5_HIL_PRESSURE_MAGIC)return false;
    crc=V5SubjectConfig_Crc32(bytes,length-4u);if(crc!=get_u32(bytes+length-4u))return false;
    out->sample_index=get_u32(bytes+4u);out->pressure=get_f32(bytes+8u);
    out->signal_valid=bytes[12u]!=0u;out->cycle_reset=bytes[13u]!=0u;
    return get_u16(bytes+14u)==0u;
}

size_t V5Hil_EncodeTelemetry(uint8_t*bytes,size_t capacity,uint32_t sample_index,
                             float pressure,const V5AppOutput*out,uint32_t processing_time_us)
{
    uint32_t flags=0u,i,offset=24u;
    if(bytes==NULL||out==NULL||capacity<V5_HIL_TELEMETRY_FRAME_SIZE)return 0u;
    memset(bytes,0,V5_HIL_TELEMETRY_FRAME_SIZE);
    if(out->candidate.data_valid)flags|=1u<<0;
    if(out->candidate.candidate_active)flags|=1u<<1;
    if(out->features.available)flags|=1u<<2;
    if(out->runtime.score_positive)flags|=1u<<3;
    if(out->runtime.t0_trigger)flags|=1u<<4;
    if(out->runtime.shadow_mode)flags|=1u<<5;
    if(out->runtime.stimulation_request)flags|=1u<<6;
    if(out->stim_output_on)flags|=1u<<7;
    if(out->config_valid)flags|=1u<<8;
    if(out->runtime.stimulation_enabled)flags|=1u<<9;
    put_u32(bytes,V5_HIL_TELEMETRY_MAGIC);put_u32(bytes+4u,sample_index);
    put_f32(bytes+8u,pressure);put_f32(bytes+12u,out->candidate.baseline);
    put_u32(bytes+16u,out->candidate.candidate_event_id);put_u32(bytes+20u,flags);
    for(i=0u;i<V5_MODEL_FEATURE_COUNT;++i){put_f32(bytes+offset,out->features.values[i]);offset+=4u;}
    put_f32(bytes+84u,out->runtime.score);put_f32(bytes+88u,out->runtime.threshold);
    put_u32(bytes+92u,out->runtime.latched_event_id);put_u32(bytes+96u,(uint32_t)out->stim_state);
    put_u32(bytes+100u,processing_time_us);put_u32(bytes+104u,0u);
    put_u32(bytes+108u,V5SubjectConfig_Crc32(bytes,108u));return V5_HIL_TELEMETRY_FRAME_SIZE;
}
