/* mlp.c - model setup, validation, sizing and dtype dispatch. */
#include "mlp.h"
#include "mlp_internal.h"

uint16_t mlpi_max_hidden(const mlp_model_t *m)
{
    uint16_t w = 0;
    uint_fast8_t i;

    for (i = 0; i + 1u < m->n_layers; ++i)
        if (m->layers[i].n_out > w) w = m->layers[i].n_out;
    return w;
}

uint16_t mlpi_max_width(const mlp_model_t *m)
{
    uint16_t w = 0;
    uint_fast8_t i;

    for (i = 0; i < m->n_layers; ++i)
        if (m->layers[i].n_out > w) w = m->layers[i].n_out;
    return w;
}

uint32_t mlpi_total_outputs(const mlp_model_t *m)
{
    uint32_t n = 0;
    uint_fast8_t i;

    for (i = 0; i < m->n_layers; ++i) n += m->layers[i].n_out;
    return n;
}

int mlpi_dtype_enabled(uint8_t dtype)
{
    switch (dtype) {
    case MLP_F32: return MLP_ENABLE_F32;
    case MLP_Q7:  return MLP_ENABLE_Q7;
    case MLP_Q15: return MLP_ENABLE_Q15;
    default:      return 0;
    }
}

size_t mlpi_layer_bytes(uint16_t n_in, uint16_t n_out, uint8_t dtype, size_t *b_off)
{
    const size_t esize = dtype == MLP_Q7 ? 1u : (dtype == MLP_Q15 ? 2u : 4u);
    const size_t off = ((size_t)n_in * n_out * esize + 3u) & ~(size_t)3u;

    if (b_off != NULL) *b_off = off;
    return off + (size_t)n_out * 4u;    /* float and int32 biases are both 4 bytes */
}

float mlpi_pow2f(int e)
{
    union { float f; uint32_t u; } v;

    if (e < -126) e = -126;
    if (e > 127) e = 127;
    v.u = (uint32_t)(e + 127) << 23;
    return v.f;
}

uint32_t mlp_param_count(const uint16_t *sizes, uint8_t n_layers)
{
    uint32_t n = 0;
    uint_fast8_t i;

    if (sizes == NULL) return 0;
    for (i = 0; i < n_layers; ++i) n += MLP_LAYER_PARAMS(sizes[i], sizes[i + 1u]);
    return n;
}

size_t mlp_buf_bytes(const mlp_model_t *m, uint8_t dtype)
{
    size_t n = 0;
    uint_fast8_t i;

    if (m == NULL || m->layers == NULL) return 0;
    for (i = 0; i < m->n_layers; ++i) n += mlpi_layer_bytes(m->layers[i].n_in, m->layers[i].n_out, dtype, NULL);
    return n;
}

size_t mlp_forward_ws_bytes(const mlp_model_t *m)
{
    size_t elems;

    if (m == NULL || m->layers == NULL || m->n_layers <= 1u) return 0;
    elems = mlpi_max_hidden(m);
    if (m->n_layers > 2u) elems *= 2u;
    return elems * (m->dtype == MLP_F32 ? sizeof(float) : sizeof(int16_t));
}

uint16_t mlp_n_inputs(const mlp_model_t *m)
{
    return (m != NULL && m->layers != NULL && m->n_layers > 0u) ? m->layers[0].n_in : 0u;
}

uint16_t mlp_n_outputs(const mlp_model_t *m)
{
    return (m != NULL && m->layers != NULL && m->n_layers > 0u) ? m->layers[m->n_layers - 1u].n_out : 0u;
}

mlp_status_t mlp_model_init_f32(mlp_model_t *m, mlp_layer_t *layers,
                                const uint16_t *sizes, const uint8_t *acts,
                                uint8_t n_layers, float *params)
{
    float *p = params;
    uint_fast8_t i;

    if (m == NULL || layers == NULL || sizes == NULL || acts == NULL || params == NULL || n_layers == 0u)
        return MLP_ERR_ARG;

    for (i = 0; i < n_layers; ++i) {
        mlp_layer_t *L = &layers[i];

        L->n_in = sizes[i];
        L->n_out = sizes[i + 1u];
        L->act = acts[i];
        L->w_frac = 0;
        L->out_frac = 0;
        L->w = p;
        p += (size_t)L->n_in * L->n_out;
        L->b = p;
        p += L->n_out;
    }

    m->layers = layers;
    m->n_layers = n_layers;
    m->dtype = MLP_F32;
    m->in_frac = 0;
    m->flags = 0;
    return mlp_model_check(m);
}

mlp_status_t mlp_model_check(const mlp_model_t *m)
{
    uint_fast8_t i;

    if (m == NULL || m->layers == NULL || m->n_layers == 0u) return MLP_ERR_ARG;
    if (!mlpi_dtype_enabled(m->dtype)) return MLP_ERR_DTYPE;

    for (i = 0; i < m->n_layers; ++i) {
        const mlp_layer_t *L = &m->layers[i];

        if (L->n_in == 0u || L->n_out == 0u) return MLP_ERR_SHAPE;
        if (i > 0u && L->n_in != m->layers[i - 1u].n_out) return MLP_ERR_SHAPE;
        if (L->w == NULL || L->b == NULL) return MLP_ERR_ARG;
        if (L->act > MLP_ACT_SOFTMAX) return MLP_ERR_ACT;
        if (L->act == MLP_ACT_SOFTMAX && (i + 1u != m->n_layers || m->dtype != MLP_F32))
            return MLP_ERR_ACT;
    }
    return MLP_OK;
}

mlp_status_t mlp_forward(const mlp_model_t *m, const void *in, void *out, void *ws)
{
    if (m == NULL || m->layers == NULL || m->n_layers == 0u || in == NULL || out == NULL || in == out)
        return MLP_ERR_ARG;
    if (m->n_layers > 1u && ws == NULL) return MLP_ERR_ARG;

    switch (m->dtype) {
#if MLP_ENABLE_F32
    case MLP_F32:
        mlpi_forward_f32(m, (const float *)in, (float *)out, (float *)ws);
        return MLP_OK;
#endif
#if MLP_ENABLE_Q7
    case MLP_Q7:
#endif
#if MLP_ENABLE_Q15
    case MLP_Q15:
#endif
#if MLP_ENABLE_Q7 || MLP_ENABLE_Q15
        return mlpi_forward_q(m, (const int16_t *)in, (int16_t *)out, (int16_t *)ws);
#endif
    default:
        return MLP_ERR_DTYPE;
    }
}

uint16_t mlp_argmax(const mlp_model_t *m, const void *out)
{
    const uint16_t n = mlp_n_outputs(m);
    uint16_t best = 0, i;

    if (out == NULL || n == 0u) return 0;
    if (m->dtype == MLP_F32) {
        const float *y = (const float *)out;
        for (i = 1; i < n; ++i) if (y[i] > y[best]) best = i;
    } else {
        const int16_t *y = (const int16_t *)out;
        for (i = 1; i < n; ++i) if (y[i] > y[best]) best = i;
    }
    return best;
}

mlp_status_t mlp_input_from_f32(const mlp_model_t *m, const float *x, int16_t *q)
{
    uint_fast16_t i, n;
    float scale;

    if (m == NULL || x == NULL || q == NULL || m->layers == NULL || m->n_layers == 0u) return MLP_ERR_ARG;
    if (m->dtype == MLP_F32) return MLP_ERR_DTYPE;

    n = m->layers[0].n_in;
    scale = mlpi_pow2f(m->in_frac);
    for (i = 0; i < n; ++i) {
        float v = x[i] * scale;
        v += v >= 0.0f ? 0.5f : -0.5f;
        q[i] = v >= 32767.0f ? (int16_t)32767 : (v <= -32768.0f ? (int16_t)-32768 : (int16_t)v);
    }
    return MLP_OK;
}

mlp_status_t mlp_output_to_f32(const mlp_model_t *m, const int16_t *q, float *y)
{
    uint_fast16_t i, n;
    float scale;

    if (m == NULL || q == NULL || y == NULL || m->layers == NULL || m->n_layers == 0u) return MLP_ERR_ARG;
    if (m->dtype == MLP_F32) return MLP_ERR_DTYPE;

    n = m->layers[m->n_layers - 1u].n_out;
    scale = mlpi_pow2f(-m->layers[m->n_layers - 1u].out_frac);
    for (i = 0; i < n; ++i) y[i] = (float)q[i] * scale;
    return MLP_OK;
}
