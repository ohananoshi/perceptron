/* test_quant.c - float -> Q7/Q15 accuracy, overflow safety and argument checks. */
#include <string.h>
#include "test.h"
#include "mlp_quant.h"

#if MLP_ENABLE_QUANT && MLP_ENABLE_Q7 && MLP_ENABLE_Q15

typedef struct {
    uint16_t n_in;
    uint32_t seed;
    float scale;
} rand_src_t;

/* Inputs uniform in [-scale, scale), reproducible per index. */
static int rand_get(void *ctx, uint16_t idx, float *x, float *t)
{
    const rand_src_t *r = (const rand_src_t *)ctx;
    uint32_t s = r->seed + 7919u * (uint32_t)idx + 1u;
    uint16_t i;

    for (i = 0; i < r->n_in; ++i) x[i] = r->scale * test_frand(&s);
    t[0] = 0.0f;
    return 0;
}

static int failing_get(void *ctx, uint16_t idx, float *x, float *t)
{
    (void)ctx; (void)idx; (void)x; (void)t;
    return 1;
}

/* Worst output error of the quantized model over 64 unseen inputs, relative
   to max(1, largest float output). */
static double quant_error(const uint16_t *sizes, const uint8_t *acts, uint8_t nl, uint8_t dtype, uint32_t seed)
{
    float params[512], ws[256], fws[2 * REF_MAX], x[REF_MAX], y[REF_MAX], yq_f[REF_MAX], t[REF_MAX];
    int32_t qbuf[256];
    int16_t xq[REF_MAX], yq[REF_MAX], qws[2 * REF_MAX];
    mlp_layer_t layers[4], qlayers[4];
    mlp_model_t m, q;
    rand_src_t calib = { 0, 0, 1.0f }, test = { 0, 0, 1.0f };
    double worst = 0.0, ymax = 1.0;
    uint32_t k, s = seed;
    uint16_t i, n;

    calib.n_in = test.n_in = sizes[0];
    calib.seed = seed * 3u;
    test.seed = seed * 5u + 1u;
    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, nl, params) == MLP_OK);
    for (k = 0; k < mlp_param_count(sizes, nl); ++k) params[k] = test_frand(&s);
    CHECK(mlp_quant_ws_bytes(&m) <= sizeof ws);
    CHECK(mlp_buf_bytes(&m, dtype) <= sizeof qbuf);
    CHECK(mlp_quantize(&m, dtype, rand_get, &calib, 64, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_OK);

    for (n = 0; n < 64; ++n) {
        rand_get(&test, n, x, t);
        CHECK(mlp_forward(&m, x, y, fws) == MLP_OK);
        CHECK(mlp_input_from_f32(&q, x, xq) == MLP_OK);
        CHECK(mlp_forward(&q, xq, yq, qws) == MLP_OK);
        CHECK(mlp_output_to_f32(&q, yq, yq_f) == MLP_OK);
        for (i = 0; i < sizes[nl]; ++i) {
            const double e = fabs((double)yq_f[i] - (double)y[i]);
            if (e > worst) worst = e;
            if (fabs((double)y[i]) > ymax) ymax = fabs((double)y[i]);
        }
    }
    return worst / ymax;
}

static void test_accuracy(void)
{
    static const uint16_t sizes[] = { 4, 8, 6, 3 };
    static const uint8_t configs[][3] = {
        { MLP_ACT_TANH, MLP_ACT_TANH, MLP_ACT_LINEAR },
        { MLP_ACT_SIGMOID, MLP_ACT_TANH, MLP_ACT_SIGMOID },
        { MLP_ACT_RELU, MLP_ACT_LEAKY_RELU, MLP_ACT_LINEAR },
        { MLP_ACT_TANH, MLP_ACT_RELU, MLP_ACT_TANH },
    };
    unsigned c;

    /* Q15 is limited by the sigmoid/tanh table (7.5e-4 per layer) and int16
       activations; Q7 by 8-bit weights (step 1/127 of the largest weight). */
    for (c = 0; c < sizeof configs / sizeof configs[0]; ++c) {
        const double e15 = quant_error(sizes, configs[c], 3, MLP_Q15, 11u + c);
        const double e7 = quant_error(sizes, configs[c], 3, MLP_Q7, 11u + c);
        CHECK(e15 < 5e-3);
        CHECK(e7 < 6e-2);
    }
}

/* 300 inputs with large weights: the quantizer must give up weight bits
   until even all-saturated inputs cannot overflow the int32 accumulator
   (UBSan would flag the signed overflow). */
static void test_overflow_bound(void)
{
    enum { NI = 300 };
    static const uint16_t sizes[] = { NI, 2 };
    static const uint8_t acts[] = { MLP_ACT_LINEAR };
    static float params[NI * 2 + 2], ws[NI + 8];
    static int32_t qbuf[(NI * 2 * 2 + 3) / 4 + 4];
    static int16_t xq[NI];
    rand_src_t calib = { NI, 77u, 1.0f };
    mlp_layer_t layers[1], qlayers[1];
    mlp_model_t m, q;
    int16_t y[2];
    uint8_t dt;
    int i;

    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 1, params) == MLP_OK);
    for (i = 0; i < NI; ++i) { params[i] = 0.9f; params[NI + i] = -0.9f; }
    params[2 * NI] = 0.5f;
    params[2 * NI + 1] = -0.5f;
    for (dt = MLP_Q7; dt <= MLP_Q15; ++dt) {
        CHECK(mlp_quantize(&m, dt, rand_get, &calib, 16, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_OK);
        for (i = 0; i < NI; ++i) xq[i] = 32767;
        CHECK(mlp_forward(&q, xq, y, NULL) == MLP_OK);
        CHECK(y[0] == 32767 && y[1] == -32768);
        for (i = 0; i < NI; ++i) xq[i] = -32768;
        CHECK(mlp_forward(&q, xq, y, NULL) == MLP_OK);
        CHECK(y[0] == -32768 && y[1] == 32767);
    }
}

static void test_softmax_classifier(void)
{
    static const uint16_t sizes[] = { 4, 6, 3 };
    static const uint8_t acts[] = { MLP_ACT_RELU, MLP_ACT_SOFTMAX };
    float params[64], ws[64], fws[8], x[4], y[3], t[1];
    int32_t qbuf[32];
    int16_t xq[4], yq[3], qws[8];
    rand_src_t src = { 4, 5u, 1.0f };
    mlp_layer_t layers[2], qlayers[2];
    mlp_model_t m, q;
    uint32_t s = 321u, k;
    int agree = 0;
    uint16_t n;

    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 2, params) == MLP_OK);
    for (k = 0; k < mlp_param_count(sizes, 2); ++k) params[k] = test_frand(&s);
    CHECK(mlp_buf_bytes(&m, MLP_Q7) <= sizeof qbuf);
    CHECK(mlp_quantize(&m, MLP_Q7, rand_get, &src, 64, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_OK);
    CHECK(qlayers[1].act == MLP_ACT_LINEAR);
    for (n = 100; n < 200; ++n) {
        rand_get(&src, n, x, t);
        CHECK(mlp_forward(&m, x, y, fws) == MLP_OK);
        CHECK(mlp_input_from_f32(&q, x, xq) == MLP_OK);
        CHECK(mlp_forward(&q, xq, yq, qws) == MLP_OK);
        if (mlp_argmax(&m, y) == mlp_argmax(&q, yq)) ++agree;
    }
    CHECK(agree >= 95);
}

static void test_quant_errors(void)
{
    static const uint16_t sizes[] = { 2, 3, 1 };
    static const uint8_t acts[] = { MLP_ACT_TANH, MLP_ACT_LINEAR };
    float params[13] = { 0.0f }, ws[32];
    int32_t qbuf[16];
    rand_src_t src = { 2, 1u, 1.0f };
    mlp_layer_t layers[2], qlayers[2], q2layers[2];
    mlp_model_t m, q, q2;

    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 2, params) == MLP_OK);
    CHECK(mlp_buf_bytes(&m, MLP_Q7) == 28u);        /* (6 -> 8) + 12, (3 -> 4) + 4 */
    CHECK(mlp_buf_bytes(&m, MLP_Q15) == 36u);       /* 12 + 12, (6 -> 8) + 4 */
    CHECK(mlp_buf_bytes(&m, MLP_F32) == mlp_param_count(sizes, 2) * 4u);
    CHECK(MLP_LAYER_BYTES_Q7(2, 3) + MLP_LAYER_BYTES_Q7(3, 1) == 28u);
    CHECK(MLP_LAYER_BYTES_Q15(2, 3) + MLP_LAYER_BYTES_Q15(3, 1) == 36u);
    CHECK(mlp_quantize(&m, MLP_Q15, rand_get, &src, 4, ws, &q, qlayers, qbuf, 35) == MLP_ERR_ARG);
    CHECK(mlp_quantize(&m, MLP_Q7, NULL, &src, 4, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_ERR_ARG);
    CHECK(mlp_quantize(&m, MLP_Q7, rand_get, &src, 0, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_ERR_ARG);
    CHECK(mlp_quantize(&m, MLP_F32, rand_get, &src, 4, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_ERR_DTYPE);
    CHECK(mlp_quantize(&m, MLP_Q7, failing_get, &src, 4, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_ERR_IO);
    CHECK(mlp_quantize(&m, MLP_Q7, rand_get, &src, 4, ws, &q, qlayers, qbuf, sizeof qbuf) == MLP_OK);
    CHECK(mlp_quantize(&q, MLP_Q15, rand_get, &src, 4, ws, &q2, q2layers, qbuf, sizeof qbuf) == MLP_ERR_DTYPE);
}

void test_quant(void)
{
    test_accuracy();
    test_overflow_bound();
    test_softmax_classifier();
    test_quant_errors();
}

#else
void test_quant(void) {}
#endif
