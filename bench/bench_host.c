/* bench_host.c - time per inference and memory per number type on the host.
   Absolute times only describe this PC; the ratios between types and the
   byte counts carry over to MCUs. */
#include <stdio.h>
#include <time.h>
#include "mlp.h"
#include "mlp_quant.h"
#include "mlp_train.h"

#define MAX_PARAMS 4096
#define MAX_IN 64

typedef struct {
    const char *name;
    uint16_t sizes[3];
    uint8_t acts[2];
    long iters;
} bench_t;

static uint32_t g_rng = 2463534242u;

static int rand_input(void *ctx, uint16_t idx, float *x, float *t)
{
    uint16_t i, n = *(const uint16_t *)ctx;

    (void)idx;
    for (i = 0; i < n; ++i) x[i] = (float)(mlp_xorshift32(&g_rng) >> 8) * (2.0f / 16777216.0f) - 1.0f;
    t[0] = 0.0f;
    return 0;
}

static double now_ns(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
}

static double time_forward(const mlp_model_t *m, const void *in, void *out, void *ws, long iters)
{
    double t0, t1;
    long i;

    mlp_forward(m, in, out, ws);                /* warm up */
    t0 = now_ns();
    for (i = 0; i < iters; ++i) mlp_forward(m, in, out, ws);
    t1 = now_ns();
    return (t1 - t0) / (double)iters;
}

int main(void)
{
    static const bench_t benches[] = {
        { "8-16-4 tanh/sigmoid", { 8, 16, 4 }, { MLP_ACT_TANH, MLP_ACT_SIGMOID }, 2000000 },
        { "64-32-10 relu/linear", { 64, 32, 10 }, { MLP_ACT_RELU, MLP_ACT_LINEAR }, 200000 },
    };
    static float params[MAX_PARAMS], ws[MAX_PARAMS], fws[64], x[MAX_IN], y[16], t[1];
    static int32_t q7buf[MAX_PARAMS], q15buf[MAX_PARAMS];
    static int16_t xq7[MAX_IN], xq15[MAX_IN], yq[16], qws[64];
    unsigned b;

    printf("%-22s %-5s %10s %12s %10s\n", "model", "type", "ns/infer", "weights [B]", "ws [B]");
    for (b = 0; b < sizeof benches / sizeof benches[0]; ++b) {
        const bench_t *B = &benches[b];
        mlp_layer_t lf[2], l7[2], l15[2];
        mlp_model_t mf, m7, m15;
        uint16_t n_in = B->sizes[0];

        if (mlp_model_init_f32(&mf, lf, B->sizes, B->acts, 2, params) != MLP_OK
            || mlp_init_params(&mf, params, 1u + b) != MLP_OK
            || mlp_quantize(&mf, MLP_Q7, rand_input, &n_in, 256, ws, &m7, l7, q7buf, sizeof q7buf) != MLP_OK
            || mlp_quantize(&mf, MLP_Q15, rand_input, &n_in, 256, ws, &m15, l15, q15buf, sizeof q15buf) != MLP_OK)
            return 1;
        rand_input(&n_in, 0, x, t);
        mlp_input_from_f32(&m7, x, xq7);
        mlp_input_from_f32(&m15, x, xq15);

        printf("%-22s %-5s %10.1f %12lu %10lu\n", B->name, "F32", time_forward(&mf, x, y, fws, B->iters),
               (unsigned long)mlp_buf_bytes(&mf, MLP_F32), (unsigned long)mlp_forward_ws_bytes(&mf));
        printf("%-22s %-5s %10.1f %12lu %10lu\n", "", "Q15", time_forward(&m15, xq15, yq, qws, B->iters),
               (unsigned long)mlp_buf_bytes(&m15, MLP_Q15), (unsigned long)mlp_forward_ws_bytes(&m15));
        printf("%-22s %-5s %10.1f %12lu %10lu\n", "", "Q7", time_forward(&m7, xq7, yq, qws, B->iters),
               (unsigned long)mlp_buf_bytes(&m7, MLP_Q7), (unsigned long)mlp_forward_ws_bytes(&m7));
    }
    return 0;
}
