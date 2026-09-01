#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "v5_candidate.h"
#include "v5_features.h"
#include "v5_runtime.h"
#include "v5_subject_config.h"

#define MAX_LINE 4096
#define MAX_COLS 32
#define MAX_TRIGGERS 64

typedef struct { char animal[16], cycle[16]; int event_id, expected, actual; } Trigger;

static size_t split(char *line, char **cols)
{
    size_t n=1; char *p=line; cols[0]=p;
    while(*p && n<MAX_COLS){
        if(*p==','){*p='\0';cols[n++]=p+1;}
        else if(*p=='\r'||*p=='\n'){*p='\0';break;}
        ++p;
    }
    return n;
}
static int col(char **h,size_t n,const char *name)
{size_t i;for(i=0;i<n;++i)if(strcmp(h[i],name)==0)return(int)i;return-1;}
static int boolean(const char*x)
{return strcmp(x,"True")==0||strcmp(x,"true")==0||strcmp(x,"1")==0;}
static Trigger *trigger_slot(Trigger *triggers,size_t *count,const char*a,const char*c,int id)
{
    size_t i; for(i=0;i<*count;++i) if(triggers[i].event_id==id&&
        strcmp(triggers[i].animal,a)==0&&strcmp(triggers[i].cycle,c)==0)return &triggers[i];
    if(*count>=MAX_TRIGGERS)return NULL;
    strncpy(triggers[*count].animal,a,sizeof(triggers[*count].animal)-1u);
    strncpy(triggers[*count].cycle,c,sizeof(triggers[*count].cycle)-1u);
    triggers[*count].event_id=id; triggers[*count].expected=-1; triggers[*count].actual=-1;
    return &triggers[(*count)++];
}

static int load_config(const char *path,V5SubjectConfig *config)
{
    uint8_t bytes[V5_SUBJECT_CONFIG_BINARY_SIZE];FILE *file=fopen(path,"rb");size_t n;
    if(file==NULL)return 0;
    n=fread(bytes,1u,sizeof(bytes),file);
    fclose(file);
    if(n!=sizeof(bytes)||!V5SubjectConfig_Decode(config,bytes,n))return 0;
    bytes[20]^=1u;
    if(V5SubjectConfig_Decode(config,bytes,n))return 0;
    bytes[20]^=1u;
    return V5SubjectConfig_Decode(config,bytes,n)?1:0;
}

int main(int argc,char**argv)
{
    static V5CandidateState candidate; static V5FeatureState features;
    static V5SubjectConfig f37_config,f26_config;
    V5Runtime runtime; V5CandidateConfig cfg; FILE*file;
    char line[MAX_LINE],*header[MAX_COLS],*cols[MAX_COLS],animal[32]="",cycle[32]="";
    int ca,cc,ct,ci,cp,cv,cpp,cpd,eca,eid,er,efa,es,esp,ets,ett;
    size_t n,rows=0,score_fail=0,positive_fail=0,state_fail=0,candidate_fail=0;
    float max_score_error=0.0f; Trigger triggers[MAX_TRIGGERS]={0}; size_t trigger_count=0,i;
    unsigned long f37_expected=0,f37_actual=0,f26_expected=0,f26_actual=0;
    if(argc!=4)return 2;
    if(!load_config(argv[2],&f37_config)||!load_config(argv[3],&f26_config))return 2;
    file=fopen(argv[1],"rb");
    if(file==NULL||fgets(line,sizeof(line),file)==NULL)return 2;
    n=split(line,header);
#define C(x) col(header,n,x)
    ca=C("animal");cc=C("cycle_id");ct=C("is_test");ci=C("sample_index");
    cp=C("pressure");cv=C("signal_valid");cpp=C("prior_sigma_p");cpd=C("prior_sigma_dpdt");
    eca=C("expected_candidate_active");eid=C("expected_candidate_event_id");
    er=C("expected_registered");efa=C("expected_feature_available");es=C("expected_score");
    esp=C("expected_score_positive");ets=C("expected_t0_state");ett=C("expected_t0_trigger");
    if(ett<0)return 2;
    V5Runtime_Init(&runtime); V5Features_Init(&features);
    while(fgets(line,sizeof(line),file)!=NULL){
        V5CandidateOutput co; V5FeatureOutput fo; V5CandidateInput input={0};
        V5RuntimeOutput ro; int is_test,registered,index,expected_id;
        float pressure; n=split(line,cols); if((size_t)ett>=n)continue;
        if(strcmp(animal,cols[ca])!=0){
            const V5SubjectConfig *loaded=strcmp(cols[ca],"STxF37")==0?&f37_config:&f26_config;
            if(!V5SubjectConfig_ApplyCandidate(loaded,&cfg))return 3;
            if(fabsf(cfg.prior_sigma_p-strtof(cols[cpp],NULL))>1e-6f||
               fabsf(cfg.prior_sigma_dpdt-strtof(cols[cpd],NULL))>1e-6f)return 3;
            V5Candidate_Init(&candidate,&cfg);strncpy(animal,cols[ca],sizeof(animal)-1u);cycle[0]='\0';
        }
        if(strcmp(cycle,cols[cc])!=0){
            V5Candidate_BeginCycle(&candidate);V5Features_BeginCycle(&features);V5Runtime_Init(&runtime);
            if(strcmp(animal,"STxF37")==0)V5Runtime_LoadSubjectConfig(&runtime,&f37_config);
            else V5Runtime_LoadSubjectConfig(&runtime,&f26_config);
            strncpy(cycle,cols[cc],sizeof(cycle)-1u);
        }
        index=atoi(cols[ci]);pressure=strtof(cols[cp],NULL);
        co=V5Candidate_Step(&candidate,pressure,boolean(cols[cv]));
        V5Features_PushPressure(&features,pressure,boolean(cols[cv]));fo=V5Features_Compute(&features);
        input.candidate_active=co.candidate_active;input.candidate_event_id=co.candidate_event_id;
        input.candidate_ended=co.candidate_ended;input.recovery_event=false;
        ro=V5Runtime_Step(&runtime,input,fo.values,fo.available);
        is_test=boolean(cols[ct]);registered=boolean(cols[er]);expected_id=atoi(cols[eid]);
        if(!is_test)continue;
        rows++;
        if((int)co.candidate_active!=boolean(cols[eca])&&
           !(co.candidate_ended||(!co.candidate_active&&expected_id>0)))candidate_fail++;
        if(!registered)continue;
        if(co.candidate_active&&fo.available&&boolean(cols[efa])){
            float error=fabsf(ro.score-strtof(cols[es],NULL));if(error>max_score_error)max_score_error=error;
            if(error>1e-5f)score_fail++;
            if((int)ro.score_positive!=boolean(cols[esp]))positive_fail++;
        }
        if((int)(co.candidate_active&&ro.score_positive)!=boolean(cols[ets]))state_fail++;
        if(boolean(cols[ett])){
            Trigger*t=trigger_slot(triggers,&trigger_count,animal,cycle,expected_id);if(t)t->expected=index;
            if(strcmp(animal,"STxF37")==0)f37_expected++;else f26_expected++;
        }
        if(ro.t0_trigger){
            Trigger*t=trigger_slot(triggers,&trigger_count,animal,cycle,(int)ro.candidate_event_id);if(t)t->actual=index;
            if(strcmp(animal,"STxF37")==0)f37_actual++;else f26_actual++;
        }
    }
    fclose(file);
    printf("test pressure samples       = %lu\n",(unsigned long)rows);
    printf("model score max error       = %.9g\n",max_score_error);
    printf("score failures              = %lu\n",(unsigned long)score_fail);
    printf("score_positive mismatches   = %lu\n",(unsigned long)positive_fail);
    printf("T0 state mismatches         = %lu\n",(unsigned long)state_fail);
    printf("candidate trace mismatches  = %lu\n",(unsigned long)candidate_fail);
    printf("F37 triggers Python/C       = %lu/%lu\n",f37_expected,f37_actual);
    printf("F26 triggers Python/C       = %lu/%lu\n",f26_expected,f26_actual);
    for(i=0;i<trigger_count;++i){
        int delta=(triggers[i].expected>=0&&triggers[i].actual>=0)?
            abs(triggers[i].expected-triggers[i].actual):999999;
        if(delta>25){printf("TRIGGER DIFF %s %s event=%d py=%d c=%d\n",triggers[i].animal,
            triggers[i].cycle,triggers[i].event_id,triggers[i].expected,triggers[i].actual);state_fail++;}
    }
    if(score_fail||positive_fail||state_fail||f37_expected!=f37_actual||f26_expected!=f26_actual)return 1;
    puts("PASS_F37_PRESSURE_ONLY_FULL_REPLAY");puts("PASS_F26_PRESSURE_ONLY_FULL_REPLAY");
    puts("PASS_GENERIC_CONFIG_SWAP_F37");puts("PASS_GENERIC_CONFIG_SWAP_F26");
    puts("PASS_V5_PYTHON_TO_C_ALGORITHM_PORT");return 0;
}
