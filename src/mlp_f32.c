/* mlp_f32.c - float32 inference kernels. */
#include "mlp.h"

#if MLP_ENABLE_F32

#include "mlp_act.h"
#include "mlp_internal.h"

/* Two (or four, unrolled) independent accumulators let pipelined FPUs
   (Cortex-M4F/M7, ESP32) overlap multiply-adds instead of waiting on one sum.
   The RAM and ROM versions share this body so they sum in the same order and
   give bit-identical results. */
#define MLPI_RD_RAM(p) (*(p))
#if MLP_UNROLL
#define MLPI_DOT_F32_BODY(RD)                                               \
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;                       \
    uint_fast16_t i = 0;                                                    \
    for (; i + 4u <= n; i += 4u) {                                          \
        a0 += RD(w + i) * x[i];                                             \
        a1 += RD(w + i + 1u) * x[i + 1u];                                   \
        a2 += RD(w + i + 2u) * x[i + 2u];                                   \
        a3 += RD(w + i + 3u) * x[i + 3u];                                   \
    }                                                                       \
    a0 += a2;                                                               \
    a1 += a3;                                                               \
    MLPI_DOT_F32_TAIL(RD)
#else
#define MLPI_DOT_F32_BODY(RD)                                               \
    float a0 = 0.0f, a1 = 0.0f;                                             \
    uint_fast16_t i = 0;                                                    \
    MLPI_DOT_F32_TAIL(RD)
#endif
#define MLPI_DOT_F32_TAIL(RD)                                               \
    for (; i + 2u <= n; i += 2u) {                                          \
        a0 += RD(w + i) * x[i];                                             \
        a1 += RD(w + i + 1u) * x[i + 1u];                                   \
    }                                                                       \
    if (i < n) a0 += RD(w + i) * x[i];                                      \
    return a0 + a1;

static float mlpi_dot_f32(const float *MLP_RESTRICT w, const float *MLP_RESTRICT x, uint_fast16_t n)
{
    MLPI_DOT_F32_BODY(MLPI_RD_RAM)
}

/* Hook for a platform dot product, e.g. CMSIS-DSP arm_dot_prod_f32 or esp-dsp
   dsps_dotprod_f32 wrapped in a function returning float. */
#ifndef MLP_DOT_F32
#define MLP_DOT_F32(w, x, n) mlpi_dot_f32((w), (x), (n))
#endif

#if MLP_HARVARD
static float mlpi_dot_f32_rom(const float *w, const float *MLP_RESTRICT x, uint_fast16_t n)
{
    MLPI_DOT_F32_BODY(MLP_RD_F32)
}
#endif

void mlpi_layer_f32(const mlp_layer_t *L, const float *MLP_RESTRICT x, float *MLP_RESTRICT y, uint8_t rom)
{
    const float *w = (const float *)L->w;
    const float *b = (const float *)L->b;
    const uint_fast16_t n_in = L->n_in, n_out = L->n_out;
    uint_fast16_t i;

#if MLP_HARVARD
    if (rom) {
        for (i = 0; i < n_out; ++i, w += n_in) y[i] = MLP_RD_F32(b + i) + mlpi_dot_f32_rom(w, x, n_in);
    } else
#else
    (void)rom;
#endif
    {
        for (i = 0; i < n_out; ++i, w += n_in) y[i] = b[i] + MLP_DOT_F32(w, x, n_in);
    }
    mlp_act_apply(L->act, y, n_out);
}

/* Layer 0 reads the caller's input, the last layer writes the caller's output,
   and hidden layers alternate between the two halves of ws. */
void mlpi_forward_f32(const mlp_model_t *m, const float *in, float *out, float *ws)
{
    const uint_fast8_t last = (uint_fast8_t)(m->n_layers - 1u);
    const size_t half = mlpi_max_hidden(m);
    const uint8_t rom = (uint8_t)(m->flags & MLP_MODEL_ROM);
    const float *src = in;
    uint_fast8_t i;

    for (i = 0; i <= last; ++i) {
        float *dst = (i == last) ? out : ws + ((i & 1u) ? half : 0u);
        mlpi_layer_f32(&m->layers[i], src, dst, rom);
        src = dst;
    }
}

void mlpi_forward_f32_all(const mlp_model_t *m, const float *in, float *acts)
{
    const uint8_t rom = (uint8_t)(m->flags & MLP_MODEL_ROM);
    const float *src = in;
    uint_fast8_t i;

    for (i = 0; i < m->n_layers; ++i) {
        mlpi_layer_f32(&m->layers[i], src, acts, rom);
        src = acts;
        acts += m->layers[i].n_out;
    }
}

#endif /* MLP_ENABLE_F32 */
