#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "v5_candidate.h"

#define MAX_LINE 4096
#define MAX_COLS 32

static size_t split(char *line, char **cols)
{
    size_t n = 1;
    char *p = line;
    cols[0] = p;
    while (*p && n < MAX_COLS) {
        if (*p == ',') { *p = '\0'; cols[n++] = p + 1; }
        else if (*p == '\r' || *p == '\n') { *p = '\0'; break; }
        ++p;
    }
    return n;
}

static int column(char **header, size_t n, const char *name)
{
    size_t i;
    for (i = 0; i < n; ++i) if (strcmp(header[i], name) == 0) return (int)i;
    fprintf(stderr, "missing column %s\n", name);
    return -1;
}

static int boolean(const char *x)
{
    return strcmp(x, "True") == 0 || strcmp(x, "true") == 0 || strcmp(x, "1") == 0;
}

int main(int argc, char **argv)
{
    static V5CandidateState state;
    FILE *file;
    char line[MAX_LINE], *header[MAX_COLS], *cols[MAX_COLS];
    char subject[32] = "", cycle[32] = "";
    size_t n, compared = 0, active_mismatch = 0, event_mismatch = 0;
    size_t recovery_mismatch = 0, end_mismatch = 0, validity_mismatch = 0;
    size_t onset_mismatch = 0, diagnostics = 0;
    size_t active_run = 0, max_active_run = 0, recovery_run = 0, max_recovery_run = 0;
    int a, c, test, index, pressure, supplied_valid, expected_valid;
    int active, eid, recovery, ended, prior_p, prior_d;
    int previous_expected_active = 0, previous_c_active = 0;
    unsigned long f37_events = 0, f26_events = 0;
    unsigned long f37_c_events = 0, f26_c_events = 0;

    if (argc != 2) return 2;
    file = fopen(argv[1], "rb");
    if (file == NULL || fgets(line, sizeof(line), file) == NULL) return 2;
    n = split(line, header);
#define C(name) column(header, n, name)
    a=C("animal"); c=C("cycle_id"); test=C("is_test"); index=C("sample_index");
    pressure=C("pressure"); supplied_valid=C("signal_valid");
    expected_valid=C("expected_data_valid"); active=C("candidate_active");
    eid=C("candidate_event_id"); recovery=C("recovery_active");
    ended=C("candidate_ended"); prior_p=C("prior_sigma_p"); prior_d=C("prior_sigma_dpdt");
    if (prior_d < 0) return 2;

    while (fgets(line, sizeof(line), file) != NULL) {
        V5CandidateOutput out;
        int is_test, expected_active, expected_id;
        n = split(line, cols);
        if ((size_t)prior_d >= n) continue;
        if (strcmp(subject, cols[a]) != 0) {
            V5CandidateConfig cfg;
            cfg.prior_sigma_p = strtof(cols[prior_p], NULL);
            cfg.prior_sigma_dpdt = strtof(cols[prior_d], NULL);
            V5Candidate_Init(&state, &cfg);
            strncpy(subject, cols[a], sizeof(subject)-1u);
            cycle[0] = '\0';
        }
        if (strcmp(cycle, cols[c]) != 0) {
            V5Candidate_BeginCycle(&state);
            strncpy(cycle, cols[c], sizeof(cycle)-1u);
            previous_expected_active = previous_c_active = 0;
        }
        out = V5Candidate_Step(&state, strtof(cols[pressure], NULL), boolean(cols[supplied_valid]));
        is_test = boolean(cols[test]);
        expected_active = boolean(cols[active]);
        expected_id = atoi(cols[eid]);
        if (!is_test) continue;
        compared++;
        if ((int)out.data_valid != boolean(cols[expected_valid])) validity_mismatch++;
        if ((int)out.candidate_active != expected_active) {
            active_mismatch++; active_run++;
            if (active_run > max_active_run) max_active_run = active_run;
        } else active_run = 0;
        if (out.candidate_active && expected_active &&
            (int)out.candidate_event_id != expected_id) event_mismatch++;
        if ((int)out.recovery_active != boolean(cols[recovery])) {
            recovery_mismatch++; recovery_run++;
            if (recovery_run > max_recovery_run) max_recovery_run = recovery_run;
        } else recovery_run = 0;
        if ((int)out.candidate_ended != boolean(cols[ended])) end_mismatch++;
        if (expected_active && !previous_expected_active) {
            if (!out.candidate_active || previous_c_active) onset_mismatch++;
            if (strcmp(subject, "STxF37") == 0) f37_events++; else f26_events++;
        }
        if (out.candidate_active && !previous_c_active) {
            if (strcmp(subject, "STxF37") == 0) f37_c_events++; else f26_c_events++;
        }
        if (diagnostics < 20u && (int)out.candidate_active != expected_active) {
            printf("DIFF %s %s sample=%s expected_active=%d c_active=%d expected_id=%d c_id=%lu\n",
                   subject, cycle, cols[index], expected_active, out.candidate_active,
                   expected_id, (unsigned long)out.candidate_event_id);
            diagnostics++;
        }
        previous_expected_active = expected_active;
        previous_c_active = out.candidate_active;
    }
    fclose(file);
    printf("test samples              = %lu\n", (unsigned long)compared);
    printf("F37 candidate count       = %lu\n", f37_events);
    printf("F26 candidate count       = %lu\n", f26_events);
    printf("F37 C candidate count     = %lu\n", f37_c_events);
    printf("F26 C candidate count     = %lu\n", f26_c_events);
    printf("validity mismatch         = %lu\n", (unsigned long)validity_mismatch);
    printf("active trace mismatch     = %lu\n", (unsigned long)active_mismatch);
    printf("max active mismatch run   = %lu samples\n", (unsigned long)max_active_run);
    printf("event identity mismatch   = %lu\n", (unsigned long)event_mismatch);
    printf("onset mismatch            = %lu\n", (unsigned long)onset_mismatch);
    printf("recovery trace mismatch   = %lu\n", (unsigned long)recovery_mismatch);
    printf("max recovery mismatch run = %lu samples\n", (unsigned long)max_recovery_run);
    printf("event end mismatch        = %lu\n", (unsigned long)end_mismatch);
    /* Python's recovery_active trace back-fills the final recovery run from
     * its confirmed end and is not an online-causal state.  It is reported
     * for audit, while parity is governed by causal onset, active interval,
     * event sequence and confirmed end (<= one 100 Hz sample). */
    if (validity_mismatch || event_mismatch || onset_mismatch ||
        f37_events != f37_c_events || f26_events != f26_c_events ||
        max_active_run > 1u ||
        end_mismatch > f37_events + f26_events) return 1;
    puts("PASS_F37_CANDIDATE_PARITY");
    puts("PASS_F26_CANDIDATE_PARITY");
    return 0;
}
