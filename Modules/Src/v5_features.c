#include "v5_features.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define PI_D 3.1415926535897932384626433832795
#define FS_D 100.0
#define EPS_D 1e-9

static int compare_double(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint32_t oldest_absolute(const V5FeatureState *state)
{
    return state->next_sample_index - state->count;
}

static bool sample_at(const V5FeatureState *state, uint32_t absolute,
                      float *pressure, bool *valid)
{
    uint32_t oldest = oldest_absolute(state), offset, slot;
    if (absolute < oldest || absolute >= state->next_sample_index) return false;
    offset = absolute - oldest;
    slot = (state->head + V5_FEATURE_PRESSURE_RING - state->count + offset) %
           V5_FEATURE_PRESSURE_RING;
    *pressure = state->pressure[slot];
    *valid = state->valid[slot] != 0u;
    return true;
}

static double median_sorted(double *x, uint32_t n)
{
    qsort(x, n, sizeof(double), compare_double);
    return (n & 1u) ? x[n / 2u] : 0.5 * (x[n / 2u - 1u] + x[n / 2u]);
}

static bool initialize_baseline(V5FeatureState *state)
{
    uint32_t i;
    double med, mad;
    float value = 0.0f;
    bool valid;
    for (i = 0u; i < V5_FEATURE_BASELINE_SAMPLES; ++i) {
        if (!sample_at(state, i, &value, &valid) || !valid || !isfinite(value))
            return false;
        state->work[i] = (double)value;
    }
    med = median_sorted(state->work, V5_FEATURE_BASELINE_SAMPLES);
    for (i = 0u; i < V5_FEATURE_BASELINE_SAMPLES; ++i) {
        sample_at(state, i, &value, &valid);
        state->work[i] = fabs((double)value - med);
    }
    mad = 1.4826 * median_sorted(state->work, V5_FEATURE_BASELINE_SAMPLES);
    if (!(mad > 0.0) || !isfinite(mad)) mad = 2.220446049250313e-16;
    state->baseline_median = med;
    state->baseline_scale = mad;
    state->baseline_ready = true;
    return true;
}

static void periodogram(V5FeatureState *state, uint32_t start, uint32_t n)
{
    uint32_t i, k, bins = n / 2u + 1u;
    double sum_i = 0.0, sum_x = 0.0, sum_ii = 0.0, sum_ix = 0.0;
    double denom, slope, intercept;
    float value = 0.0f;
    bool valid;
    for (i = 0u; i < n; ++i) {
        sample_at(state, start + i, &value, &valid);
        state->work[i] = (double)value - (double)state->baseline_median;
        sum_i += (double)i;
        sum_x += state->work[i];
        sum_ii += (double)i * (double)i;
        sum_ix += (double)i * state->work[i];
    }
    denom = (double)n * sum_ii - sum_i * sum_i;
    slope = denom != 0.0 ? ((double)n * sum_ix - sum_i * sum_x) / denom : 0.0;
    intercept = (sum_x - slope * sum_i) / (double)n;
    for (i = 0u; i < n; ++i) {
        const double hann = 0.5 - 0.5 * cos(2.0 * PI_D * (double)i / (double)(n - 1u));
        state->work[i] = (state->work[i] - intercept - slope * (double)i) * hann;
    }
    for (k = 0u; k < bins; ++k) {
        double re = 0.0, im = 0.0;
        for (i = 0u; i < n; ++i) {
            const double angle = -2.0 * PI_D * (double)k * (double)i / (double)n;
            re += state->work[i] * cos(angle);
            im += state->work[i] * sin(angle);
        }
        state->psd[k] = (re * re + im * im) / ((double)n * FS_D);
    }
}

static double band_power(const V5FeatureState *state, uint32_t n,
                         double low, double high)
{
    uint32_t k, first = 0u, last = 0u;
    const double df = FS_D / (double)n;
    double area = 0.0;
    bool found = false;
    for (k = 0u; k <= n / 2u; ++k) {
        const double frequency = (double)k * df;
        if (frequency >= low && frequency <= high) {
            if (!found) first = k;
            last = k;
            found = true;
        }
    }
    if (!found) return 0.0;
    if (last <= first) return 0.0;
    for (k = first; k < last; ++k)
        area += 0.5 * (state->psd[k] + state->psd[k + 1u]) * df;
    return area;
}

static bool all_valid(const V5FeatureState *state, uint32_t start, uint32_t n)
{
    uint32_t i;
    float value;
    bool valid;
    for (i = 0u; i < n; ++i)
        if (!sample_at(state, start + i, &value, &valid) || !valid || !isfinite(value))
            return false;
    return true;
}

void V5Features_Init(V5FeatureState *state)
{
    if (state != NULL) memset(state, 0, sizeof(*state));
}

void V5Features_BeginCycle(V5FeatureState *state)
{
    V5Features_Init(state);
}

void V5Features_PushPressure(V5FeatureState *state, float pressure, bool valid)
{
    if (state == NULL) return;
    state->pressure[state->head] = pressure;
    state->valid[state->head] = valid && isfinite(pressure) ? 1u : 0u;
    state->head = (state->head + 1u) % V5_FEATURE_PRESSURE_RING;
    if (state->count < V5_FEATURE_PRESSURE_RING) state->count++;
    state->next_sample_index++;
}

V5FeatureOutput V5Features_Compute(V5FeatureState *state)
{
    V5FeatureOutput out = {0};
    uint32_t idx, i;
    float p[200], current = 0.0f;
    bool valid;
    double scale, h[200], d[100], d_sum = 0.0, d05_sum = 0.0;
    double peak, auc = 0.0, max_d, mean, variance = 0.0;
    double current_low, current_wide, entropy_sum = 0.0, entropy = 0.0;
    uint32_t above = 0u, positive = 0u;
    if (state == NULL || state->next_sample_index == 0u) return out;
    idx = state->next_sample_index - 1u;
    if ((idx % 25u) != 0u || idx < 2700u) return out;
    if (!state->baseline_ready && !initialize_baseline(state)) return out;
    if (!all_valid(state, idx - 499u, 500u)) return out;
    scale = (double)state->baseline_scale;
    if (!(scale > 0.0)) return out;
    for (i = 0u; i < 200u; ++i) {
        sample_at(state, idx - 199u + i, &p[i], &valid);
        h[i] = ((double)p[i] - (double)state->baseline_median) / scale;
    }
    peak = h[0];
    for (i = 0u; i < 200u; ++i) {
        if (h[i] > peak) peak = h[i];
        if (h[i] > 3.68) above++;
        if (i > 0u) auc += 0.5 * (fmax(h[i-1u], 0.0) + fmax(h[i], 0.0)) / FS_D;
    }
    for (i = 0u; i < 100u; ++i) {
        d[i] = (((double)p[100u + i] - (double)p[99u + i]) / scale) * FS_D;
        d_sum += d[i];
        if (i >= 50u) d05_sum += d[i];
        if (d[i] > 0.0) positive++;
    }
    max_d = d[0];
    for (i = 1u; i < 100u; ++i) if (d[i] > max_d) max_d = d[i];
    mean = 0.0;
    for (i = 100u; i < 200u; ++i) mean += h[i];
    mean /= 100.0;
    for (i = 100u; i < 200u; ++i) variance += (h[i] - mean) * (h[i] - mean);
    variance /= 100.0;

    if (!(state->baseline_low_power > 0.0) || !(state->baseline_wide_power > 0.0)) {
        periodogram(state, 0u, V5_FEATURE_BASELINE_SAMPLES);
        state->baseline_low_power = band_power(state, V5_FEATURE_BASELINE_SAMPLES, 0.2, 0.6);
        state->baseline_wide_power = band_power(state, V5_FEATURE_BASELINE_SAMPLES, 0.2, 20.0);
    }
    periodogram(state, idx - 499u, V5_FEATURE_SPECTRAL_SAMPLES);
    current_low = band_power(state, V5_FEATURE_SPECTRAL_SAMPLES, 0.2, 0.6);
    current_wide = band_power(state, V5_FEATURE_SPECTRAL_SAMPLES, 0.2, 20.0);
    for (i = 1u; i <= 100u; ++i) entropy_sum += state->psd[i];
    if (!(entropy_sum > 0.0)) return out;
    for (i = 1u; i <= 100u; ++i) {
        const double q = state->psd[i] / entropy_sum;
        entropy -= q * log(q + EPS_D);
    }
    entropy /= log(100.0);
    sample_at(state, idx, &current, &valid);
    out.values[0] = (float)(((double)current - state->baseline_median) / scale);
    out.values[1] = (float)peak;
    out.values[2] = (float)above / 100.0f;
    out.values[3] = (float)(d05_sum / 50.0);
    out.values[4] = (float)(d_sum / 100.0);
    out.values[5] = (float)max_d;
    out.values[6] = (float)positive / 100.0f;
    out.values[7] = (float)auc;
    out.values[8] = (float)(auc / fmax(2.0 * scale, EPS_D));
    out.values[9] = (float)(d05_sum / 50.0 - d_sum / 100.0);
    out.values[10] = (float)(peak - h[199]);
    out.values[11] = (float)sqrt(variance);
    out.values[12] = (float)log((current_low + EPS_D) /
                                (state->baseline_low_power + EPS_D));
    out.values[13] = (float)log((current_wide + EPS_D) /
                                (state->baseline_wide_power + EPS_D));
    out.values[14] = (float)entropy;
    for (i = 0u; i < V5_MODEL_FEATURE_COUNT; ++i)
        if (!isfinite(out.values[i])) return (V5FeatureOutput){0};
    out.available = true;
    return out;
}
