/* mlp_fixed.c - Q7 / Q15 inference kernels.

   Each neuron accumulates acc = bias + sum(w * x) in int32 at scale
   2^-(w_frac + in_frac). The accumulator is then shifted (with rounding and
   saturation) either to the layer's out_frac, or, for sigmoid and tanh, to
   the Q11 input format of the lookup table. Nothing here divides or uses floats.

   Right shifts of negative values are assumed to be arithmetic. That is
   implementation-defined in C99, but true for GCC, Clang, IAR, Keil and XC.
*/
#include "mlp.h"

#if MLP_ENABLE_Q7 || MLP_ENABLE_Q15

#include "mlp_internal.h"

#define LUT_IN_FRAC 11      /* sigmoid/tanh table input: Q11 */
#define LUT_STEP_BITS 9     /* table step 0.25 = 2^9 in Q11 */

/* round(32768 * sigmoid(k / 4)) for k = 0..32, capped at 32767.
   Negative inputs use sigmoid(-x) = 1 - sigmoid(x). Linear interpolation
   error is below 7.5e-4 over the whole range. */
static const int16_t mlpi_sigmoid_lut[33] MLP_ROM = {
    16384, 18421, 20397, 22255, 23955, 25471, 26790, 27917, 28862, 29644, 30282,
    30799, 31214, 31545, 31807, 32015, 32179, 32307, 32408, 32487, 32549, 32597,
    32635, 32664, 32687, 32705, 32719, 32730, 32738, 32745, 32750, 32754, 32757
};

MLP_INLINE int16_t mlpi_sat16(int32_t v)
{
    return v > 32767 ? (int16_t)32767 : (v < -32768 ? (int16_t)-32768 : (int16_t)v);
}

/* v * 2^-s, rounded to nearest (half up) for s > 0, saturated for s < 0. */
static int32_t mlpi_shift_round(int32_t v, int_fast8_t s)
{
    if (s > 0) {
        int32_t half;

        if (s > 30) return 0;
        half = (int32_t)1 << (s - 1);
        if (v > INT32_MAX - half) v = INT32_MAX - half;
        return (v + half) >> s;
    }
    if (s < 0) {
        const int_fast8_t k = (int_fast8_t)-s;
        int32_t lim;

        if (k > 30) return v > 0 ? INT32_MAX : (v < 0 ? -INT32_MAX : 0);
        lim = INT32_MAX >> k;
        if (v > lim) return INT32_MAX;
        if (v < -lim) return -INT32_MAX;
        return v * ((int32_t)1 << k);
    }
    return v;
}

/* sigmoid(z) in Q15 for z in Q11. */
static int32_t mlpi_sigmoid_q15(int32_t z)
{
    const int32_t a = z < 0 ? -z : z;
    int32_t y;

    if (a >= ((int32_t)32 << LUT_STEP_BITS)) {
        y = MLP_RD_I16(&mlpi_sigmoid_lut[32]);
    } else {
        const uint_fast8_t k = (uint_fast8_t)(a >> LUT_STEP_BITS);
        const int32_t fr = a & (((int32_t)1 << LUT_STEP_BITS) - 1);
        const int32_t y0 = MLP_RD_I16(&mlpi_sigmoid_lut[k]);
        const int32_t y1 = MLP_RD_I16(&mlpi_sigmoid_lut[k + 1u]);
        y = y0 + (((y1 - y0) * fr) >> LUT_STEP_BITS);
    }
    return z < 0 ? 32768 - y : y;
}

/* Casting both operands to int32_t matters on 8/16-bit targets where int is
   16 bits; compilers still emit a 16x16->32 widening multiply. */
#if MLP_ENABLE_Q7
static int32_t mlpi_dot_q7(const int8_t *MLP_RESTRICT w, const int16_t *MLP_RESTRICT x, uint_fast16_t n)
{
    int32_t acc = 0;
    uint_fast16_t i = 0;
#if MLP_UNROLL
    for (; i + 4u <= n; i += 4u) {
        acc += (int32_t)w[i] * (int32_t)x[i] + (int32_t)w[i + 1u] * (int32_t)x[i + 1u]
             + (int32_t)w[i + 2u] * (int32_t)x[i + 2u] + (int32_t)w[i + 3u] * (int32_t)x[i + 3u];
    }
#endif
    for (; i < n; ++i) acc += (int32_t)w[i] * (int32_t)x[i];
    return acc;
}
#ifndef MLP_DOT_Q7
#define MLP_DOT_Q7(w, x, n) mlpi_dot_q7((w), (x), (n))
#endif
#endif

#if MLP_ENABLE_Q15
static int32_t mlpi_dot_q15(const int16_t *MLP_RESTRICT w, const int16_t *MLP_RESTRICT x, uint_fast16_t n)
{
    int32_t acc = 0;
    uint_fast16_t i = 0;
#if MLP_UNROLL
    for (; i + 4u <= n; i += 4u) {
        acc += (int32_t)w[i] * (int32_t)x[i] + (int32_t)w[i + 1u] * (int32_t)x[i + 1u]
             + (int32_t)w[i + 2u] * (int32_t)x[i + 2u] + (int32_t)w[i + 3u] * (int32_t)x[i + 3u];
    }
#endif
    for (; i < n; ++i) acc += (int32_t)w[i] * (int32_t)x[i];
    return acc;
}
/* Hook for e.g. CMSIS-DSP arm_dot_prod_q15 (wrap it to return int32_t). */
#ifndef MLP_DOT_Q15
#define MLP_DOT_Q15(w, x, n) mlpi_dot_q15((w), (x), (n))
#endif
#endif

#if MLP_HARVARD
static int32_t mlpi_dot_rom(uint8_t dtype, const void *w, const int16_t *x, uint_fast16_t n)
{
    int32_t acc = 0;
    uint_fast16_t i;

    if (dtype == MLP_Q7) {
        const int8_t *w8 = (const int8_t *)w;
        for (i = 0; i < n; ++i) acc += (int32_t)MLP_RD_I8(w8 + i) * (int32_t)x[i];
    } else {
        const int16_t *w16 = (const int16_t *)w;
        for (i = 0; i < n; ++i) acc += (int32_t)MLP_RD_I16(w16 + i) * (int32_t)x[i];
    }
    return acc;
}
#endif

/* Pre-activation pass: y[i] = acc_i shifted to out_frac, or to Q11 for
   sigmoid/tanh. */
static void mlpi_layer_q_pre(const mlp_layer_t *L, uint8_t dtype, int_fast8_t sh,
                             const int16_t *MLP_RESTRICT x, int16_t *MLP_RESTRICT y, uint8_t rom)
{
    const uint_fast16_t n_in = L->n_in, n_out = L->n_out;
    const int32_t *b = (const int32_t *)L->b;
    uint_fast16_t i;

#if MLP_HARVARD
    if (rom) {
        const uint8_t *w = (const uint8_t *)L->w;
        const size_t row = (size_t)n_in * (dtype == MLP_Q7 ? 1u : 2u);
        for (i = 0; i < n_out; ++i, w += row)
            y[i] = mlpi_sat16(mlpi_shift_round(MLP_RD_I32(b + i) + mlpi_dot_rom(dtype, w, x, n_in), sh));
        return;
    }
#else
    (void)rom;
#endif

#if MLP_ENABLE_Q7
    if (dtype == MLP_Q7) {
        const int8_t *w = (const int8_t *)L->w;
        for (i = 0; i < n_out; ++i, w += n_in)
            y[i] = mlpi_sat16(mlpi_shift_round(b[i] + MLP_DOT_Q7(w, x, n_in), sh));
        return;
    }
#endif
#if MLP_ENABLE_Q15
    {
        const int16_t *w = (const int16_t *)L->w;
        for (i = 0; i < n_out; ++i, w += n_in)
            y[i] = mlpi_sat16(mlpi_shift_round(b[i] + MLP_DOT_Q15(w, x, n_in), sh));
    }
#endif
}

static void mlpi_layer_q(const mlp_layer_t *L, uint8_t dtype, int_fast8_t in_frac,
                         const int16_t *MLP_RESTRICT x, int16_t *MLP_RESTRICT y, uint8_t rom)
{
    const uint_fast16_t n = L->n_out;
    const int_fast8_t acc_frac = (int_fast8_t)(L->w_frac + in_frac);
    const int_fast8_t out_frac = L->out_frac;
    uint_fast16_t i;

    switch (L->act) {
    case MLP_ACT_SIGMOID:
        mlpi_layer_q_pre(L, dtype, (int_fast8_t)(acc_frac - LUT_IN_FRAC), x, y, rom);
        for (i = 0; i < n; ++i)
            y[i] = mlpi_sat16(mlpi_shift_round(mlpi_sigmoid_q15(y[i]), (int_fast8_t)(15 - out_frac)));
        break;
    case MLP_ACT_TANH:              /* tanh(z) = 2 * sigmoid(2z) - 1 */
        mlpi_layer_q_pre(L, dtype, (int_fast8_t)(acc_frac - LUT_IN_FRAC), x, y, rom);
        for (i = 0; i < n; ++i)
            y[i] = mlpi_sat16(mlpi_shift_round(2 * mlpi_sigmoid_q15(2 * (int32_t)y[i]) - 32768,
                                               (int_fast8_t)(15 - out_frac)));
        break;
    case MLP_ACT_RELU:
        mlpi_layer_q_pre(L, dtype, (int_fast8_t)(acc_frac - out_frac), x, y, rom);
        for (i = 0; i < n; ++i) if (y[i] < 0) y[i] = 0;
        break;
    case MLP_ACT_LEAKY_RELU: {
        const int32_t alpha = (int32_t)(MLP_LEAKY_ALPHA * 32768.0f + 0.5f);
        mlpi_layer_q_pre(L, dtype, (int_fast8_t)(acc_frac - out_frac), x, y, rom);
        for (i = 0; i < n; ++i) if (y[i] < 0) y[i] = (int16_t)(((int32_t)y[i] * alpha) >> 15);
        break;
    }
    case MLP_ACT_STEP: {
        const int16_t one = mlpi_sat16((int32_t)1 << (out_frac < 0 ? 0 : (out_frac > 15 ? 15 : out_frac)));
        mlpi_layer_q_pre(L, dtype, (int_fast8_t)(acc_frac - out_frac), x, y, rom);
        for (i = 0; i < n; ++i) y[i] = y[i] > 0 ? one : (int16_t)0;
        break;
    }
    default:                        /* MLP_ACT_LINEAR */
        mlpi_layer_q_pre(L, dtype, (int_fast8_t)(acc_frac - out_frac), x, y, rom);
        break;
    }
}

mlp_status_t mlpi_forward_q(const mlp_model_t *m, const int16_t *in, int16_t *out, int16_t *ws)
{
    const uint_fast8_t last = (uint_fast8_t)(m->n_layers - 1u);
    const size_t half = mlpi_max_hidden(m);
    const uint8_t rom = (uint8_t)(m->flags & MLP_MODEL_ROM);
    const int16_t *src = in;
    int_fast8_t in_frac = m->in_frac;
    uint_fast8_t i;

    if (!mlpi_dtype_enabled(m->dtype)) return MLP_ERR_DTYPE;

    for (i = 0; i <= last; ++i) {
        const mlp_layer_t *L = &m->layers[i];
        int16_t *dst = (i == last) ? out : ws + ((i & 1u) ? half : 0u);

        if (L->act == MLP_ACT_SOFTMAX) return MLP_ERR_ACT;
        mlpi_layer_q(L, m->dtype, in_frac, src, dst, rom);
        in_frac = L->out_frac;
        src = dst;
    }
    return MLP_OK;
}

#endif /* MLP_ENABLE_Q7 || MLP_ENABLE_Q15 */
