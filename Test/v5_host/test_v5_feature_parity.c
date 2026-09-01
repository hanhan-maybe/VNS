#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "v5_features.h"

#define MAX_LINE 8192
#define MAX_COLS 48

static const char *const names[V5_MODEL_FEATURE_COUNT] = {
    "p_current_delta", "p_peak_delta", "p_threshold_above_duration",
    "p_slope_0p5s", "p_slope_1s", "p_max_positive_dpdt",
    "p_positive_dpdt_occupancy", "p_auc", "p_auc_growth",
    "pressure_curvature", "peak_to_current_drop", "p_trailing_variability_1s",
    "pressure_power_0p2_0p6_rel", "pressure_auc_0p2_20_rel",
    "pressure_spectral_entropy"
};

static size_t split(char *line, char **cols)
{
    size_t n = 1; char *p = line; cols[0] = p;
    while (*p && n < MAX_COLS) {
        if (*p == ',') { *p = '\0'; cols[n++] = p + 1; }
        else if (*p == '\r' || *p == '\n') { *p = '\0'; break; }
        ++p;
    }
    return n;
}
static int col(char **h, size_t n, const char *name)
{
    size_t i; for (i=0;i<n;++i) if(strcmp(h[i],name)==0) return (int)i;
    return -1;
}
static int boolean(const char *x)
{
    return strcmp(x,"True")==0 || strcmp(x,"true")==0 || strcmp(x,"1")==0;
}

int main(int argc, char **argv)
{
    static V5FeatureState state;
    FILE *file; char line[MAX_LINE], *header[MAX_COLS], *cols[MAX_COLS];
    char animal[32]="", cycle[32]=""; size_t n, i, compared=0, available_mismatch=0;
    int ca, cc, cp, cv, cav, cf[V5_MODEL_FEATURE_COUNT];
    double max_abs[V5_MODEL_FEATURE_COUNT]={0}, sum_abs[V5_MODEL_FEATURE_COUNT]={0};
    double max_rel[V5_MODEL_FEATURE_COUNT]={0}; size_t failures[V5_MODEL_FEATURE_COUNT]={0};
    if(argc!=2) return 2;
    file=fopen(argv[1],"rb");
    if(file==NULL || fgets(line,sizeof(line),file)==NULL) return 2;
    n=split(line,header); ca=col(header,n,"animal"); cc=col(header,n,"cycle_id");
    cp=col(header,n,"pressure");
    cv=col(header,n,"signal_valid"); cav=col(header,n,"expected_available");
    for(i=0;i<V5_MODEL_FEATURE_COUNT;++i) cf[i]=col(header,n,names[i]);
    if(cav<0 || cf[14]<0) return 2;
    V5Features_Init(&state);
    while(fgets(line,sizeof(line),file)!=NULL) {
        V5FeatureOutput out; int expected_available;
        n=split(line,cols); if((size_t)cf[14]>=n) continue;
        if(strcmp(animal,cols[ca])!=0 || strcmp(cycle,cols[cc])!=0) {
            V5Features_BeginCycle(&state);
            strncpy(animal,cols[ca],sizeof(animal)-1u);
            strncpy(cycle,cols[cc],sizeof(cycle)-1u);
        }
        V5Features_PushPressure(&state,strtof(cols[cp],NULL),boolean(cols[cv]));
        out=V5Features_Compute(&state); expected_available=boolean(cols[cav]);
        if((int)out.available!=expected_available) available_mismatch++;
        if(!expected_available || !out.available) continue;
        compared++;
        for(i=0;i<V5_MODEL_FEATURE_COUNT;++i) {
            const double expected=strtod(cols[cf[i]],NULL);
            const double error=fabs((double)out.values[i]-expected);
            const double relative=error/fmax(fabs(expected),1e-9);
            const double absolute_tolerance=(i==7u || i==8u) ? 1e-3 : 1e-4;
            sum_abs[i]+=error; if(error>max_abs[i])max_abs[i]=error;
            if(relative>max_rel[i])max_rel[i]=relative;
            if(error>absolute_tolerance+1e-5*fabs(expected)) failures[i]++;
        }
    }
    fclose(file);
    printf("feature rows compared = %lu\n",(unsigned long)compared);
    printf("availability mismatch = %lu\n",(unsigned long)available_mismatch);
    for(i=0;i<V5_MODEL_FEATURE_COUNT;++i)
        printf("%-34s n=%lu max_abs=%.9g mean_abs=%.9g max_rel=%.9g failures=%lu\n",
               names[i],(unsigned long)compared,max_abs[i],compared?sum_abs[i]/compared:0.0,
               max_rel[i],(unsigned long)failures[i]);
    if(available_mismatch) return 1;
    for(i=0;i<V5_MODEL_FEATURE_COUNT;++i) if(failures[i]) return 1;
    puts("PASS_F37_P_EARLY_PARITY"); puts("PASS_F26_P_EARLY_PARITY");
    return 0;
}
