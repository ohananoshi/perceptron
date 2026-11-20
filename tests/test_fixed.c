/* test_fixed.c - Q7/Q15 kernels on hand-built layers with known integer results. */
#include "test.h"

static mlp_model_t one_layer(const mlp_layer_t *L, uint8_t dtype, int8_t in_frac)
{
    mlp_model_t m;

    m.layers = L;
    m.n_layers = 1;
    m.dtype = dtype;
    m.in_frac = in_frac;
    m.flags = 0;
    return m;
}

static void test_q7_linear(void)
{
    /* rows [1.0, -0.5] and [0.25, 1.984375] at w_frac 6; bias 1.0 at acc scale 2^-14 */
    static const int8_t w[4] = { 64, -32, 16, 127 };
    static const int32_t b[2] = { 1 << 14, 0 };
    const int16_t x[2] = { 256, 512 };                  /* 1.0, 2.0 in Q8 */
    int16_t y[2];
    mlp_layer_t L = { 2, 2, MLP_ACT_LINEAR, 6, 8, w, b };
    mlp_model_t m = one_layer(&L, MLP_Q7, 8);

    CHECK(mlp_forward(&m, x, y, NULL) == MLP_OK);
    CHECK(y[0] == 256);                                 /* 1.0 */
    CHECK(y[1] == 1080);                                /* 4.21875 */
}

static void test_q15_saturation(void)
{
    static const int16_t w[4] = { 16384, -16384, -16384, 16384 };
    static const int32_t b[2] = { 0, 0 };
    const int16_t x[2] = { 32767, -32768 };
    int16_t y[2];
    mlp_layer_t L = { 2, 2, MLP_ACT_LINEAR, 14, 15, w, b };
    mlp_model_t m = one_layer(&L, MLP_Q15, 15);

    CHECK(mlp_forward(&m, x, y, NULL) == MLP_OK);
    CHECK(y[0] == 32767);
    CHECK(y[1] == -32768);
}

static void test_q_piecewise(void)
{
    static const int16_t w[1] = { 16384 };              /* 1.0 at w_frac 14 */
    static const int32_t b[1] = { 0 };
    const int16_t pos[1] = { 1024 }, neg[1] = { -1024 };    /* +-1.0 in Q10 */
    int16_t y[1];
    mlp_layer_t L = { 1, 1, MLP_ACT_LINEAR, 14, 10, w, b };
    mlp_model_t m = one_layer(&L, MLP_Q15, 10);

    CHECK(mlp_forward(&m, neg, y, NULL) == MLP_OK && y[0] == -1024);
    L.act = MLP_ACT_RELU;
    CHECK(mlp_forward(&m, neg, y, NULL) == MLP_OK && y[0] == 0);
    CHECK(mlp_forward(&m, pos, y, NULL) == MLP_OK && y[0] == 1024);
    L.act = MLP_ACT_LEAKY_RELU;
    CHECK(mlp_forward(&m, neg, y, NULL) == MLP_OK && y[0] >= -11 && y[0] <= -10);  /* -10.24 */
    L.act = MLP_ACT_STEP;
    CHECK(mlp_forward(&m, pos, y, NULL) == MLP_OK && y[0] == 1024);
    CHECK(mlp_forward(&m, neg, y, NULL) == MLP_OK && y[0] == 0);
    L.act = MLP_ACT_SOFTMAX;
    CHECK(mlp_forward(&m, pos, y, NULL) == MLP_ERR_ACT);
}

static void test_q_sigmoid_tanh(void)
{
    static const int16_t w[1] = { 16384 };
    static const int32_t b[1] = { 0 };
    int16_t x[1], y[1];
    double worst_s = 0.0, worst_t = 0.0;
    mlp_layer_t L = { 1, 1, MLP_ACT_SIGMOID, 14, 15, w, b };
    mlp_model_t m = one_layer(&L, MLP_Q15, 11);
    int32_t v;

    for (v = -20000; v <= 20000; v += 37) {
        const double z = (double)v / 2048.0;
        double e;

        x[0] = (int16_t)v;
        L.act = MLP_ACT_SIGMOID;
        CHECK(mlp_forward(&m, x, y, NULL) == MLP_OK);
        e = fabs((double)y[0] / 32768.0 - 1.0 / (1.0 + exp(-z)));
        if (e > worst_s) worst_s = e;

        L.act = MLP_ACT_TANH;
        CHECK(mlp_forward(&m, x, y, NULL) == MLP_OK);
        e = fabs((double)y[0] / 32768.0 - tanh(z));
        if (e > worst_t) worst_t = e;
    }
    CHECK(worst_s < 1e-3);
    CHECK(worst_t < 2e-3);
}

static void test_q_chain(void)
{
    /* identity, identity, sum: checks the per-layer fraction hand-off and both
       ping-pong halves (input frac 8 -> 10 -> 7 -> 9) */
    static const int8_t eye[4] = { 64, 0, 0, 64 }, sum[2] = { 64, 64 };
    static const int32_t b2[2] = { 0, 0 }, b1[1] = { 0 };
    const mlp_layer_t layers[3] = {
        { 2, 2, MLP_ACT_LINEAR, 6, 10, eye, b2 },
        { 2, 2, MLP_ACT_RELU, 6, 7, eye, b2 },
        { 2, 1, MLP_ACT_LINEAR, 6, 9, sum, b1 },
    };
    const mlp_model_t m = { layers, 3, MLP_Q7, 8, 0 };
    const int16_t x[2] = { 256, -128 };                 /* 1.0, -0.5 */
    int16_t y[1], ws[4];
    float f[1];

    CHECK(mlp_forward_ws_bytes(&m) == 4u * sizeof(int16_t));
    CHECK(mlp_forward(&m, x, y, ws) == MLP_OK);
    CHECK(ws[0] == 1024 && ws[1] == -512);              /* layer 0 output, Q10 */
    CHECK(ws[2] == 128 && ws[3] == 0);                  /* layer 1 output after relu, Q7 */
    CHECK(y[0] == 512);                                 /* 1.0 in Q9 */
    CHECK(mlp_output_to_f32(&m, y, f) == MLP_OK);
    CHECK_NEAR(f[0], 1.0, 1e-6);
}

static void test_q_conversions(void)
{
    static const int8_t w[3] = { 1, 1, 1 };
    static const int32_t b[1] = { 0 };
    const mlp_layer_t L = { 3, 1, MLP_ACT_LINEAR, 0, 4, w, b };
    const mlp_model_t m = { &L, 1, MLP_Q7, 8, 0 };
    const float x[3] = { 1.0f, -0.5f, 200.0f };
    const int16_t q_out[1] = { -24 };
    int16_t q[3];
    float y[1];

    CHECK(mlp_input_from_f32(&m, x, q) == MLP_OK);
    CHECK(q[0] == 256 && q[1] == -128 && q[2] == 32767);
    CHECK(mlp_output_to_f32(&m, q_out, y) == MLP_OK);
    CHECK_NEAR(y[0], -1.5, 1e-7);
    CHECK((-8 >> 1) == -4);                             /* arithmetic shift assumption */
}

void test_fixed(void)
{
#if MLP_ENABLE_Q7
    test_q7_linear();
    test_q_chain();
    test_q_conversions();
#endif
#if MLP_ENABLE_Q15
    test_q15_saturation();
    test_q_piecewise();
    test_q_sigmoid_tanh();
#endif
}
