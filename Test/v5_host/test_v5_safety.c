#include <stdio.h>
#include <stdlib.h>

#include "v5_app.h"

static int read_binary(const char *path,uint8_t *bytes,size_t length)
{
    FILE *file=fopen(path,"rb");size_t n;if(file==NULL)return 0;
    n=fread(bytes,1u,length,file);fclose(file);return n==length;
}

int main(int argc,char **argv)
{
    uint8_t bytes[V5_SUBJECT_CONFIG_BINARY_SIZE];V5AppOutput output;
    if(argc!=2||!read_binary(argv[1],bytes,sizeof(bytes)))return 2;
    if(!AppV5_Init(bytes,sizeof(bytes),1.0f,1.0f))return 1;
    if(AppV5_EnableStimulation(true))return 1;
    output=AppV5_On100Hz(10.0f,true);
    if(!output.config_valid||output.stim_output_on||output.runtime.stimulation_request)return 1;
    AppV5_ReportWatchdogFault();output=AppV5_On100Hz(10.0f,true);
    if(output.stim_state!=V5_STIM_FAULT||output.stim_output_on)return 1;
    bytes[30]^=1u;
    if(AppV5_Init(bytes,sizeof(bytes),1.0f,1.0f))return 1;
    output=AppV5_On100Hz(10.0f,true);
    if(output.config_valid||output.stim_output_on)return 1;
    puts("PASS_CONFIG_VALIDATION_AND_CRC");
    puts("PASS_STIM_FAIL_SAFE");
    puts("SHADOW_MODE_DEFAULT_TRUE");
    puts("STIMULATION_ENABLED_FALSE");
    return 0;
}
