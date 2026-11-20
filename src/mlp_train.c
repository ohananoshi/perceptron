/* mlp_train.c - float32 backpropagation with online SGD.

   Per sample, after a forward pass that keeps every layer's output:
     1. delta_L = dLoss/dz at the output layer.
     2. For each layer l from the last down to the first:
          delta_{l-1} = (W_l^T delta_l) * f'(a_{l-1}), using W_l before its update,
          then W_l -= lr * delta_l a_{l-1}^T and b_l -= lr * delta_l.
   Doing the delta and the update row by row in that order means no buffer
   the size of the weights is needed: only two delta vectors.
*/
#include "mlp_train.h"

#if MLP_ENABLE_TRAIN

#include <math.h>
#include "mlp_act.h"
#include "mlp_internal.h"

uint32_t mlp_xorshift32(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Uniform in [-1, 1) from the top 24 bits. */
static float mlpi_uniform(uint32_t *s)
{
    return (float)(mlp_xorshift32(s) >> 8) * (1.0f / 8388608.0f) - 1.0f;
}

/* A trainable model is a RAM float model laid out exactly as
   mlp_model_init_f32() lays out `params`. */
static mlp_status_t mlpi_check_trainable(const mlp_model_t *m, const float *params)
{
    const float *p = params;
    mlp_status_t st;
    uint_fast8_t i;

    if (m == NULL || params == NULL) return MLP_ERR_ARG;
    st = mlp_model_check(m);
    if (st != MLP_OK) return st;
    if (m->dtype != MLP_F32) return MLP_ERR_DTYPE;
    if (m->flags & MLP_MODEL_ROM) return MLP_ERR_ARG;

    for (i = 0; i < m->n_layers; ++i) {
        const mlp_layer_t *L = &m->layers[i];
        if (L->w != (const void *)p) return MLP_ERR_ARG;
        p += (size_t)L->n_in * L->n_out;
        if (L->b != (const void *)p) return MLP_ERR_ARG;
        p += L->n_out;
    }
    return MLP_OK;
}

static mlp_status_t mlpi_check_loss(uint8_t loss, uint8_t out_act)
{
    switch (loss) {
    case MLP_LOSS_MSE: return out_act == MLP_ACT_SOFTMAX ? MLP_ERR_ACT : MLP_OK;
    case MLP_LOSS_CE:  return (out_act == MLP_ACT_SIGMOID || out_act == MLP_ACT_SOFTMAX) ? MLP_OK : MLP_ERR_ACT;
    default:           return MLP_ERR_ARG;
    }
}

/* Loss of y against t; if d is not NULL, also dLoss/dz for each output.
   Cross-entropy with a sigmoid or softmax output gives d = y - t exactly. */
static float mlpi_loss_delta(uint8_t loss, uint8_t act, const float *y, const float *t,
                             uint_fast16_t n, float *d)
{
    float sum = 0.0f;
    uint_fast16_t i;

    for (i = 0; i < n; ++i) {
        const float e = y[i] - t[i];

        if (loss == MLP_LOSS_MSE) {
            sum += 0.5f * e * e;
            if (d != NULL) d[i] = e * mlp_act_deriv(act, y[i]);
        } else {
            const float yc = y[i] < 1e-7f ? 1e-7f : (y[i] > 1.0f - 1e-7f ? 1.0f - 1e-7f : y[i]);

            if (act == MLP_ACT_SOFTMAX) sum -= t[i] * logf(yc);
            else sum -= t[i] * logf(yc) + (1.0f - t[i]) * logf(1.0f - yc);
            if (d != NULL) d[i] = e;
        }
    }
    return sum;
}

/* One weight row: accumulate its share of the previous layer's delta with the
   old weights (when dprev is not NULL), then apply the SGD/momentum step. */
static void mlpi_update_row(float *MLP_RESTRICT row, float *MLP_RESTRICT vrow,
                            const float *MLP_RESTRICT a, float *MLP_RESTRICT dprev,
                            uint_fast16_t n, float g, const mlp_sgd_t *opt)
{
    const float step = opt->lr * g;
    uint_fast16_t j;

    if (dprev != NULL)
        for (j = 0; j < n; ++j) dprev[j] += row[j] * g;
    if (vrow == NULL) {
        for (j = 0; j < n; ++j) row[j] -= step * a[j];
    } else {
        const float mu = opt->momentum;
        for (j = 0; j < n; ++j) {
            vrow[j] = mu * vrow[j] - step * a[j];
            row[j] += vrow[j];
        }
    }
}

mlp_status_t mlp_init_params(const mlp_model_t *m, float *params, uint32_t seed)
{
    uint32_t s = seed != 0u ? seed : 0x9E3779B9u;
    float *p = params;
    mlp_status_t st = mlpi_check_trainable(m, params);
    uint_fast8_t l;

    if (st != MLP_OK) return st;
    for (l = 0; l < m->n_layers; ++l) {
        const mlp_layer_t *L = &m->layers[l];
        const uint32_t nw = (uint32_t)L->n_in * L->n_out;
        const float fan = (L->act == MLP_ACT_RELU || L->act == MLP_ACT_LEAKY_RELU)
                        ? 6.0f / (float)L->n_in
                        : 6.0f / ((float)L->n_in + (float)L->n_out);
        const float lim = sqrtf(fan);
        uint32_t k;

        for (k = 0; k < nw; ++k) *p++ = lim * mlpi_uniform(&s);
        for (k = 0; k < L->n_out; ++k) *p++ = 0.0f;
    }
    return MLP_OK;
}

size_t mlp_train_ws_bytes(const mlp_model_t *m)
{
    if (m == NULL || m->layers == NULL || m->n_layers == 0u) return 0;
    return ((size_t)mlpi_total_outputs(m) + 2u * (size_t)mlpi_max_width(m)
            + mlp_n_inputs(m) + mlp_n_outputs(m)) * sizeof(float);
}

float mlp_loss(const mlp_model_t *m, uint8_t loss, const float *y, const float *t)
{
    const mlp_layer_t *out;

    if (m == NULL || m->layers == NULL || m->n_layers == 0u || y == NULL || t == NULL) return -1.0f;
    out = &m->layers[m->n_layers - 1u];
    if (mlpi_check_loss(loss, out->act) != MLP_OK) return -1.0f;
    return mlpi_loss_delta(loss, out->act, y, t, out->n_out, NULL);
}

mlp_status_t mlp_train_sample(const mlp_model_t *m, float *params, const mlp_sgd_t *opt,
                              const float *x, const float *t, void *ws, float *loss)
{
    float *const acts = (float *)ws;
    float *d, *dp;
    const mlp_layer_t *out;
    uint32_t aoff, poff;
    uint_fast8_t l;
    float L;
    mlp_status_t st = mlpi_check_trainable(m, params);

    if (st != MLP_OK) return st;
    if (opt == NULL || x == NULL || t == NULL || ws == NULL) return MLP_ERR_ARG;
    if (opt->momentum != 0.0f && opt->velocity == NULL) return MLP_ERR_ARG;
    out = &m->layers[m->n_layers - 1u];
    st = mlpi_check_loss(opt->loss, out->act);
    if (st != MLP_OK) return st;

    aoff = mlpi_total_outputs(m);
    d = acts + aoff;
    dp = d + mlpi_max_width(m);
    poff = 0;
    for (l = 0; l < m->n_layers; ++l) poff += MLP_LAYER_PARAMS(m->layers[l].n_in, m->layers[l].n_out);

    mlpi_forward_f32_all(m, x, acts);
    aoff -= out->n_out;
    L = mlpi_loss_delta(opt->loss, out->act, acts + aoff, t, out->n_out, d);

    for (l = m->n_layers; l-- > 0u;) {
        const uint_fast16_t n_in = m->layers[l].n_in, n_out = m->layers[l].n_out;
        const float *a_prev = l > 0u ? acts + aoff - n_in : x;
        float *const dprev = l > 0u ? dp : NULL;
        float *w, *b, *vw = NULL, *vb = NULL, *tmp;
        uint_fast16_t i;

        poff -= MLP_LAYER_PARAMS(n_in, n_out);
        w = params + poff;
        b = w + n_in * n_out;
        if (opt->momentum != 0.0f) {
            vw = opt->velocity + poff;
            vb = vw + n_in * n_out;
        }

        if (dprev != NULL)
            for (i = 0; i < n_in; ++i) dprev[i] = 0.0f;
        for (i = 0; i < n_out; ++i, w += n_in) {
            mlpi_update_row(w, vw, a_prev, dprev, n_in, d[i], opt);
            if (vw != NULL) {
                vw += n_in;
                vb[i] = opt->momentum * vb[i] - opt->lr * d[i];
                b[i] += vb[i];
            } else {
                b[i] -= opt->lr * d[i];
            }
        }

        if (dprev != NULL) {
            const uint8_t act = m->layers[l - 1u].act;
            for (i = 0; i < n_in; ++i) dprev[i] *= mlp_act_deriv(act, a_prev[i]);
            tmp = d; d = dp; dp = tmp;
            aoff -= (uint32_t)n_in;
        }
    }

    if (loss != NULL) *loss = L;
    return MLP_OK;
}

static uint32_t mlpi_gcd(uint32_t a, uint32_t b)
{
    while (b != 0u) {
        const uint32_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

/* Shuffling visits idx = start, start + stride, ... (mod n) with
   gcd(stride, n) = 1: every sample exactly once, and no index array in RAM. */
mlp_status_t mlp_train_epoch(const mlp_model_t *m, float *params, const mlp_sgd_t *opt,
                             mlp_sample_fn get, void *ctx, uint16_t n, void *ws, float *mean_loss)
{
    float *xs, *ts, sum = 0.0f;
    uint32_t idx = 0, stride = 1;
    uint_fast16_t k;
    mlp_status_t st = mlpi_check_trainable(m, params);

    if (st != MLP_OK) return st;
    if (opt == NULL || get == NULL || ws == NULL || n == 0u) return MLP_ERR_ARG;

    xs = (float *)ws + mlpi_total_outputs(m) + 2u * (size_t)mlpi_max_width(m);
    ts = xs + mlp_n_inputs(m);

    if (opt->shuffle != NULL && n > 1u) {
        idx = mlp_xorshift32(opt->shuffle) % n;
        do {
            stride = 1u + mlp_xorshift32(opt->shuffle) % (n - 1u);
        } while (mlpi_gcd(stride, n) != 1u);
    }

    for (k = 0; k < n; ++k) {
        float l;

        if (get(ctx, (uint16_t)idx, xs, ts) != 0) return MLP_ERR_IO;
        st = mlp_train_sample(m, params, opt, xs, ts, ws, &l);
        if (st != MLP_OK) return st;
        sum += l;
        idx += stride;
        if (idx >= n) idx -= n;
    }

    if (mean_loss != NULL) *mean_loss = sum / (float)n;
    return MLP_OK;
}

#endif /* MLP_ENABLE_TRAIN */
