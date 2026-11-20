/* test_forward.c - float model setup and inference against the double reference. */
#include "test.h"
#include "mlp_act.h"

#if MLP_FAST_MATH
#define FWD_TOL 2e-3
#else
#define FWD_TOL 1e-5
#endif

static void test_sizes(void)
{
    static const uint16_t sizes[] = { 3, 5, 2 };

    CHECK(MLP_LAYER_PARAMS(3, 5) == 20u);
    CHECK(mlp_param_count(sizes, 2) == 20u + 12u);
    CHECK(MLP_FWD_WS_ELEMS(1, 0) == 1u);
    CHECK(MLP_FWD_WS_ELEMS(2, 7) == 7u);
    CHECK(MLP_FWD_WS_ELEMS(4, 7) == 14u);
}

/* Build a random float model, run it, compare with ref_forward(). */
static void check_against_ref(const uint16_t *sizes, const uint8_t *acts, uint8_t n_layers, uint32_t seed)
{
    float params[512], ws[2 * REF_MAX], x[REF_MAX], y[REF_MAX];
    double dparams[512], dx[REF_MAX], dy[REF_MAX];
    mlp_layer_t layers[8];
    mlp_model_t m;
    const uint32_t np = mlp_param_count(sizes, n_layers);
    uint32_t i;
    int trial;

    CHECK(np <= 512u);
    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, n_layers, params) == MLP_OK);
    for (i = 0; i < np; ++i) dparams[i] = params[i] = test_frand(&seed);
    CHECK(mlp_forward_ws_bytes(&m) <= sizeof ws);

    for (trial = 0; trial < 5; ++trial) {
        for (i = 0; i < sizes[0]; ++i) dx[i] = x[i] = 2.0f * test_frand(&seed);
        CHECK(mlp_forward(&m, x, y, ws) == MLP_OK);
        ref_forward(sizes, acts, n_layers, dparams, dx, dy);
        for (i = 0; i < sizes[n_layers]; ++i) CHECK_NEAR(y[i], dy[i], FWD_TOL);
    }
}

static void test_against_reference(void)
{
    static const uint8_t hidden[] = { MLP_ACT_LINEAR, MLP_ACT_RELU, MLP_ACT_LEAKY_RELU,
                                      MLP_ACT_SIGMOID, MLP_ACT_TANH };
    static const uint8_t output[] = { MLP_ACT_LINEAR, MLP_ACT_SIGMOID, MLP_ACT_TANH,
                                      MLP_ACT_SOFTMAX, MLP_ACT_STEP };
    static const uint16_t one[] = { 4, 3 };
    static const uint16_t two[] = { 3, 7, 2 };
    static const uint16_t deep[] = { 3, 6, 5, 4, 2 };   /* exercises ping-pong in both halves */
    unsigned h, o;

    for (o = 0; o < sizeof output; ++o) {
        uint8_t a1[1];
        a1[0] = output[o];
        check_against_ref(one, a1, 1, 11u + o);
        for (h = 0; h < sizeof hidden; ++h) {
            uint8_t a2[2], a4[4];
            a2[0] = hidden[h]; a2[1] = output[o];
            a4[0] = hidden[h]; a4[1] = hidden[(h + 1u) % sizeof hidden];
            a4[2] = hidden[(h + 2u) % sizeof hidden]; a4[3] = output[o];
            check_against_ref(two, a2, 2, 100u + 10u * o + h);
            check_against_ref(deep, a4, 4, 200u + 10u * o + h);
        }
    }
}

static void test_workspace_and_errors(void)
{
    static const uint16_t sizes1[] = { 2, 3 };
    static const uint16_t sizes3[] = { 2, 9, 4, 1 };
    static const uint8_t acts[] = { MLP_ACT_TANH, MLP_ACT_TANH, MLP_ACT_LINEAR };
    static const uint8_t bad_acts[] = { MLP_ACT_SOFTMAX, MLP_ACT_TANH, MLP_ACT_LINEAR };
    float params[64] = { 0.0f }, x[2] = { 0.5f, -0.5f }, y[4], ws[18];
    mlp_layer_t layers[3], broken[2];
    mlp_model_t m, mb;

    CHECK(mlp_model_init_f32(&m, layers, sizes1, acts, 1, params) == MLP_OK);
    CHECK(mlp_forward_ws_bytes(&m) == 0u);
    CHECK(mlp_forward(&m, x, y, NULL) == MLP_OK);      /* single layer needs no workspace */

    CHECK(mlp_model_init_f32(&m, layers, sizes3, acts, 3, params) == MLP_OK);
    CHECK(mlp_forward_ws_bytes(&m) == 2u * 9u * sizeof(float));
    CHECK(mlp_n_inputs(&m) == 2u);
    CHECK(mlp_n_outputs(&m) == 1u);
    CHECK(mlp_forward(&m, x, y, NULL) == MLP_ERR_ARG);
    CHECK(mlp_forward(&m, x, x, ws) == MLP_ERR_ARG);
    CHECK(mlp_forward(NULL, x, y, ws) == MLP_ERR_ARG);

    broken[0] = layers[0];
    broken[1] = layers[2];                              /* 9 outputs feeding a 4-input layer */
    mb.layers = broken; mb.n_layers = 2; mb.dtype = MLP_F32; mb.in_frac = 0; mb.flags = 0;
    CHECK(mlp_model_check(&mb) == MLP_ERR_SHAPE);
    mb.layers = layers; mb.n_layers = 1; mb.dtype = 9;
    CHECK(mlp_model_check(&mb) == MLP_ERR_DTYPE);
    CHECK(mlp_forward(&mb, x, y, ws) == MLP_ERR_DTYPE);

    CHECK(mlp_model_init_f32(&m, layers, sizes3, bad_acts, 3, params) == MLP_ERR_ACT);
    CHECK(mlp_model_init_f32(&m, layers, sizes3, acts, 0, params) == MLP_ERR_ARG);
}

static void test_argmax(void)
{
    static const uint16_t sizes[] = { 1, 4 };
    static const uint8_t acts[] = { MLP_ACT_LINEAR };
    float params[8] = { 0.0f };
    const float out[4] = { 0.1f, 0.7f, -2.0f, 0.69f };
    mlp_layer_t layers[1];
    mlp_model_t m;

    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 1, params) == MLP_OK);
    CHECK(mlp_argmax(&m, out) == 1u);
}

static void test_expf(void)
{
    double worst = 0.0;
    int i;

    for (i = -2000; i <= 2000; ++i) {
        const float x = (float)i * 0.01f;
        const double e = fabs((double)mlp_expf(x) / exp((double)x) - 1.0);
        if (e > worst) worst = e;
    }
#if MLP_FAST_MATH
    CHECK(worst < 1.2e-4);
#else
    CHECK(worst < 1e-6);
#endif
    CHECK(mlp_expf(-200.0f) >= 0.0f);
    CHECK(mlp_sigmoidf(200.0f) <= 1.0f && mlp_sigmoidf(-200.0f) >= 0.0f);
}

void test_forward(void)
{
    test_sizes();
    test_against_reference();
    test_workspace_and_errors();
    test_argmax();
    test_expf();
}
