/* mlp_quant.c - float32 -> Q7 / Q15 conversion with calibration. */
#include "mlp_quant.h"

#if MLP_ENABLE_QUANT

#include <math.h>
#include "mlp_internal.h"

#define ACC_LIMIT 1073741824.0f     /* 2^30: worst-case accumulator magnitude */
#define HEADROOM 1.25f
#define FRAC_MIN (-8)

/* Most fractional bits f <= hi with maxabs * 2^f <= qmax. */
static int_fast8_t mlpi_frac_for(float maxabs, float qmax, int_fast8_t hi)
{
    int_fast8_t f = hi;

    while (f > FRAC_MIN && maxabs * mlpi_pow2f(f) > qmax) --f;
    return f;
}

/* Round half away from zero, clamped to +-lim (lim <= 2^30). */
static int32_t mlpi_qround(float v, float lim)
{
    if (v > lim) v = lim;
    if (v < -lim) v = -lim;
    return (int32_t)(v >= 0.0f ? v + 0.5f : v - 0.5f);
}

static float mlpi_rd_f32(const float *p, uint32_t i, uint8_t rom)
{
#if MLP_HARVARD
    if (rom) return MLP_RD_F32(p + i);
#else
    (void)rom;
#endif
    return p[i];
}

/* Does every neuron of L keep |bias| + sum|w| * 32768 below ACC_LIMIT at
   weight bits f and input bits in_f? */
static int mlpi_bound_ok(const mlp_layer_t *L, uint8_t rom, int_fast8_t f, int_fast8_t in_f, float qmax)
{
    const float *w = (const float *)L->w, *b = (const float *)L->b;
    const float ws = mlpi_pow2f(f), bs = mlpi_pow2f(f + in_f);
    uint32_t i, j, k = 0;

    for (i = 0; i < L->n_out; ++i) {
        float s = 0.0f;
        for (j = 0; j < L->n_in; ++j, ++k)
            s += fabsf((float)mlpi_qround(mlpi_rd_f32(w, k, rom) * ws, qmax));
        if (s * 32768.0f + fabsf(mlpi_rd_f32(b, i, rom) * bs) > ACC_LIMIT) return 0;
    }
    return 1;
}

size_t mlp_quant_ws_bytes(const mlp_model_t *src)
{
    if (src == NULL || src->layers == NULL || src->n_layers == 0u) return 0;
    return ((size_t)src->n_layers + 1u + mlpi_total_outputs(src)
            + mlp_n_inputs(src) + mlp_n_outputs(src)) * sizeof(float);
}

mlp_status_t mlp_quantize(const mlp_model_t *src, uint8_t dtype,
                          mlp_sample_fn calib, void *ctx, uint16_t n_calib, void *ws,
                          mlp_model_t *dst, mlp_layer_t *dst_layers, void *buf, size_t buf_bytes)
{
    const float qmax = dtype == MLP_Q7 ? 127.0f : 32767.0f;
    float *amax, *acts, *xs, *ts;
    uint8_t *p = (uint8_t *)buf;
    const uint8_t rom = (uint8_t)(src != NULL ? src->flags & MLP_MODEL_ROM : 0u);
    int_fast8_t in_f;
    uint_fast16_t s, l;
    uint32_t i, k;
    mlp_status_t st;

    if (src == NULL || calib == NULL || n_calib == 0u || ws == NULL || dst == NULL
        || dst_layers == NULL || buf == NULL)
        return MLP_ERR_ARG;
    st = mlp_model_check(src);
    if (st != MLP_OK) return st;
    if (src->dtype != MLP_F32 || (dtype != MLP_Q7 && dtype != MLP_Q15) || !mlpi_dtype_enabled(dtype))
        return MLP_ERR_DTYPE;
    if (buf_bytes < mlp_buf_bytes(src, dtype)) return MLP_ERR_ARG;

    amax = (float *)ws;                         /* [0] input, [l + 1] output of layer l */
    acts = amax + src->n_layers + 1u;
    xs = acts + mlpi_total_outputs(src);
    ts = xs + mlp_n_inputs(src);
    for (l = 0; l <= src->n_layers; ++l) amax[l] = 0.0f;

    /* Calibration: largest magnitude of the input and of every layer output.
       A softmax layer is measured before the softmax, since it becomes linear. */
    for (s = 0; s < n_calib; ++s) {
        const mlp_layer_t *last = &src->layers[src->n_layers - 1u];
        float *a = acts;

        if (calib(ctx, (uint16_t)s, xs, ts) != 0) return MLP_ERR_IO;
        for (i = 0; i < src->layers[0].n_in; ++i) if (fabsf(xs[i]) > amax[0]) amax[0] = fabsf(xs[i]);
        mlpi_forward_f32_all(src, xs, acts);
        if (last->act == MLP_ACT_SOFTMAX) {
            mlp_layer_t lin = *last;
            float *out = acts + mlpi_total_outputs(src) - last->n_out;
            lin.act = MLP_ACT_LINEAR;
            mlpi_layer_f32(&lin, src->n_layers > 1u ? out - last->n_in : xs, out, rom);
        }
        for (l = 0; l < src->n_layers; ++l) {
            for (i = 0; i < src->layers[l].n_out; ++i, ++a)
                if (fabsf(*a) > amax[l + 1u]) amax[l + 1u] = fabsf(*a);
        }
    }

    in_f = mlpi_frac_for(amax[0] * HEADROOM, 32767.0f, 15);
    dst->in_frac = (int8_t)in_f;

    for (l = 0; l < src->n_layers; ++l) {
        const mlp_layer_t *S = &src->layers[l];
        mlp_layer_t *D = &dst_layers[l];
        const float *w = (const float *)S->w, *b = (const float *)S->b;
        const uint8_t act = S->act == MLP_ACT_SOFTMAX ? (uint8_t)MLP_ACT_LINEAR : S->act;
        const uint32_t nw = (uint32_t)S->n_in * S->n_out;
        size_t b_off;
        int_fast8_t f, out_f;
        float wmax = 0.0f, wsc, bsc;

        if (act == MLP_ACT_SIGMOID || act == MLP_ACT_TANH) out_f = 15;
        else if (act == MLP_ACT_STEP) out_f = 14;
        else out_f = mlpi_frac_for(amax[l + 1u] * HEADROOM, 32767.0f, 15);

        for (k = 0; k < nw; ++k) if (fabsf(mlpi_rd_f32(w, k, rom)) > wmax) wmax = fabsf(mlpi_rd_f32(w, k, rom));
        f = mlpi_frac_for(wmax, qmax, (int_fast8_t)(30 - in_f));    /* acc_frac stays <= 30 */
        while (!mlpi_bound_ok(S, rom, f, in_f, qmax)) {
            if (f <= FRAC_MIN) return MLP_ERR_RANGE;
            --f;
        }

        D->n_in = S->n_in;
        D->n_out = S->n_out;
        D->act = act;
        D->w_frac = (int8_t)f;
        D->out_frac = (int8_t)out_f;
        (void)mlpi_layer_bytes(S->n_in, S->n_out, dtype, &b_off);

        wsc = mlpi_pow2f(f);
        bsc = mlpi_pow2f(f + in_f);
        if (dtype == MLP_Q7) {
            int8_t *qw = (int8_t *)(void *)p;
            for (k = 0; k < nw; ++k) qw[k] = (int8_t)mlpi_qround(mlpi_rd_f32(w, k, rom) * wsc, qmax);
        } else {
            int16_t *qw = (int16_t *)(void *)p;
            for (k = 0; k < nw; ++k) qw[k] = (int16_t)mlpi_qround(mlpi_rd_f32(w, k, rom) * wsc, qmax);
        }
        {
            int32_t *qb = (int32_t *)(void *)(p + b_off);
            for (i = 0; i < S->n_out; ++i) qb[i] = mlpi_qround(mlpi_rd_f32(b, i, rom) * bsc, ACC_LIMIT);
        }
        D->w = p;
        D->b = p + b_off;
        p += mlpi_layer_bytes(S->n_in, S->n_out, dtype, NULL);
        in_f = out_f;
    }

    dst->layers = dst_layers;
    dst->n_layers = src->n_layers;
    dst->dtype = dtype;
    dst->flags = 0;
    return mlp_model_check(dst);
}

#endif /* MLP_ENABLE_QUANT */
