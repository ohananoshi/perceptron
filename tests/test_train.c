/* test_train.c - backprop correctness (gradient check) and convergence. */
#include <string.h>
#include "test.h"
#include "mlp_train.h"

#if MLP_ENABLE_TRAIN

#if MLP_FAST_MATH
#define GRAD_ABS 2e-3
#define GRAD_REL 2e-2
#else
#define GRAD_ABS 1e-4
#define GRAD_REL 1e-3
#endif

typedef struct {
    const float *x, *t;
    uint16_t n_in, n_out;
    unsigned visits[16];
} table_t;

static int table_get(void *ctx, uint16_t idx, float *x, float *t)
{
    table_t *tab = (table_t *)ctx;

    memcpy(x, tab->x + (size_t)idx * tab->n_in, tab->n_in * sizeof(float));
    memcpy(t, tab->t + (size_t)idx * tab->n_out, tab->n_out * sizeof(float));
    if (idx < 16u) tab->visits[idx]++;
    return 0;
}

static int failing_get(void *ctx, uint16_t idx, float *x, float *t)
{
    (void)ctx; (void)idx; (void)x; (void)t;
    return -1;
}

/* With lr = 1 and no momentum, one step changes each parameter by exactly
   minus its gradient. Compare that with central differences of the double
   reference loss. */
static void grad_check(const uint16_t *sizes, const uint8_t *acts, uint8_t nl, uint8_t loss, uint32_t seed)
{
    float params[256], before[256], ws[256], x[REF_MAX], t[REF_MAX], L;
    double dparams[256], dx[REF_MAX], dt[REF_MAX], y[REF_MAX];
    const uint16_t n_out = sizes[nl];
    const uint8_t out_act = acts[nl - 1u];
    const double eps = 1e-5;
    mlp_layer_t layers[4];
    mlp_model_t m;
    mlp_sgd_t opt;
    uint32_t np, k, i;

    opt.lr = 1.0f; opt.momentum = 0.0f; opt.velocity = NULL; opt.shuffle = NULL; opt.loss = loss;
    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, nl, params) == MLP_OK);
    CHECK(mlp_train_ws_bytes(&m) <= sizeof ws);
    np = mlp_param_count(sizes, nl);
    for (k = 0; k < np; ++k) before[k] = params[k] = 0.8f * test_frand(&seed);   /* biases too */
    for (i = 0; i < sizes[0]; ++i) dx[i] = x[i] = test_frand(&seed);
    for (i = 0; i < n_out; ++i) {
        if (out_act == MLP_ACT_SOFTMAX) t[i] = i == 1u ? 1.0f : 0.0f;
        else if (out_act == MLP_ACT_SIGMOID) t[i] = 0.5f + 0.5f * test_frand(&seed);
        else t[i] = test_frand(&seed);
        dt[i] = t[i];
    }

    CHECK(mlp_train_sample(&m, params, &opt, x, t, ws, &L) == MLP_OK);
    for (k = 0; k < np; ++k) dparams[k] = before[k];
    ref_forward(sizes, acts, nl, dparams, dx, y);
    CHECK_NEAR(L, ref_loss(loss, out_act, y, dt, n_out), 1e-5 + 1e-4 * fabs((double)L));

    for (k = 0; k < np; ++k) {
        double lp, lm, num;

        dparams[k] = (double)before[k] + eps;
        ref_forward(sizes, acts, nl, dparams, dx, y);
        lp = ref_loss(loss, out_act, y, dt, n_out);
        dparams[k] = (double)before[k] - eps;
        ref_forward(sizes, acts, nl, dparams, dx, y);
        lm = ref_loss(loss, out_act, y, dt, n_out);
        dparams[k] = before[k];
        num = (lp - lm) / (2.0 * eps);
        CHECK_NEAR(before[k] - params[k], num, GRAD_ABS + GRAD_REL * fabs(num));
    }
}

static void test_gradients(void)
{
    static const uint8_t hidden[][2] = {
        { MLP_ACT_SIGMOID, MLP_ACT_TANH }, { MLP_ACT_RELU, MLP_ACT_LEAKY_RELU },
        { MLP_ACT_TANH, MLP_ACT_LINEAR }, { MLP_ACT_LEAKY_RELU, MLP_ACT_SIGMOID },
    };
    static const uint8_t outputs[][2] = {             /* activation, loss */
        { MLP_ACT_LINEAR, MLP_LOSS_MSE }, { MLP_ACT_SIGMOID, MLP_LOSS_MSE },
        { MLP_ACT_TANH, MLP_LOSS_MSE }, { MLP_ACT_SIGMOID, MLP_LOSS_CE },
        { MLP_ACT_SOFTMAX, MLP_LOSS_CE },
    };
    static const uint16_t sizes[] = { 3, 5, 4, 3 };
    static const uint16_t single[] = { 4, 3 };
    unsigned h, o;

    for (o = 0; o < sizeof outputs / sizeof outputs[0]; ++o) {
        uint8_t a1[1];
        a1[0] = outputs[o][0];
        grad_check(single, a1, 1, outputs[o][1], 7u + o);
        for (h = 0; h < sizeof hidden / sizeof hidden[0]; ++h) {
            uint8_t a3[3];
            a3[0] = hidden[h][0]; a3[1] = hidden[h][1]; a3[2] = outputs[o][0];
            grad_check(sizes, a3, 3, outputs[o][1], 1000u + 10u * o + h);
        }
    }
}

static void test_init_and_momentum(void)
{
    static const uint16_t sizes[] = { 4, 6, 2 };
    static const uint8_t acts[] = { MLP_ACT_RELU, MLP_ACT_LINEAR };
    float p1[48], p2[48], vel[48], ws[64];
    const float x[4] = { 0.3f, -0.2f, 0.9f, 0.1f }, t[2] = { 1.0f, -1.0f };
    mlp_layer_t l1[2], l2[2];
    mlp_model_t m1, m2;
    mlp_sgd_t sgd, mom;
    const uint32_t np = mlp_param_count(sizes, 2);
    int same = 1, any_v = 0, differs = 0;
    uint32_t k;

    CHECK(np == 44u);
    CHECK(mlp_model_init_f32(&m1, l1, sizes, acts, 2, p1) == MLP_OK);
    CHECK(mlp_model_init_f32(&m2, l2, sizes, acts, 2, p2) == MLP_OK);
    CHECK(mlp_init_params(&m1, p1, 42u) == MLP_OK);
    CHECK(mlp_init_params(&m2, p2, 42u) == MLP_OK);
    CHECK(memcmp(p1, p2, np * sizeof(float)) == 0);
    for (k = 0; k < 24u; ++k) CHECK(fabsf(p1[k]) <= 1.2248f);   /* He limit sqrt(6/4) */
    for (k = 24u; k < 30u; ++k) CHECK(p1[k] == 0.0f);           /* hidden biases */
    CHECK(mlp_init_params(&m2, p2, 43u) == MLP_OK);
    CHECK(memcmp(p1, p2, np * sizeof(float)) != 0);
    CHECK(mlp_init_params(&m2, p2, 42u) == MLP_OK);

    memset(vel, 0, sizeof vel);
    sgd.lr = 0.1f; sgd.momentum = 0.0f; sgd.velocity = NULL; sgd.shuffle = NULL; sgd.loss = MLP_LOSS_MSE;
    mom = sgd;
    mom.momentum = 0.9f;
    mom.velocity = vel;

    /* From zero velocity, the first momentum step equals plain SGD. */
    CHECK(mlp_train_sample(&m1, p1, &sgd, x, t, ws, NULL) == MLP_OK);
    CHECK(mlp_train_sample(&m2, p2, &mom, x, t, ws, NULL) == MLP_OK);
    for (k = 0; k < np; ++k) {
        if (fabsf(p1[k] - p2[k]) > 1e-7f) same = 0;
        if (vel[k] != 0.0f) any_v = 1;
    }
    CHECK(same && any_v);
    CHECK(mlp_train_sample(&m1, p1, &sgd, x, t, ws, NULL) == MLP_OK);
    CHECK(mlp_train_sample(&m2, p2, &mom, x, t, ws, NULL) == MLP_OK);
    for (k = 0; k < np; ++k) if (fabsf(p1[k] - p2[k]) > 1e-6f) differs = 1;
    CHECK(differs);
}

static int train_until(const mlp_model_t *m, float *params, const mlp_sgd_t *opt, table_t *tab,
                       uint16_t n, float *ws, float target, int max_epochs)
{
    float loss = 1.0f;
    int e;

    for (e = 0; e < max_epochs && loss > target; ++e)
        if (mlp_train_epoch(m, params, opt, table_get, tab, n, ws, &loss) != MLP_OK) return -1;
    return e;
}

static void test_xor(uint8_t out_act, uint8_t loss)
{
    static const float X[8] = { 0, 0, 0, 1, 1, 0, 1, 1 };
    static const float T[4] = { 0, 1, 1, 0 };
    static const uint16_t sizes[] = { 2, 4, 1 };
    uint8_t acts[2];
    float params[17], vel[17] = { 0.0f }, ws[32], y[1], wsf[4];
    uint32_t rng = 1234u;
    table_t tab;
    mlp_layer_t layers[2];
    mlp_model_t m;
    mlp_sgd_t opt;
    int epochs, i;

    acts[0] = MLP_ACT_TANH; acts[1] = out_act;
    memset(&tab, 0, sizeof tab);
    tab.x = X; tab.t = T; tab.n_in = 2; tab.n_out = 1;
    opt.lr = 0.2f; opt.momentum = 0.9f; opt.velocity = vel; opt.shuffle = &rng; opt.loss = loss;

    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 2, params) == MLP_OK);
    CHECK(mlp_init_params(&m, params, 7u) == MLP_OK);
    epochs = train_until(&m, params, &opt, &tab, 4, ws, 1e-3f, 3000);
    CHECK(epochs > 0 && epochs < 3000);
    for (i = 0; i < 4; ++i) {
        CHECK(mlp_forward(&m, X + 2 * i, y, wsf) == MLP_OK);
        CHECK_NEAR(y[0], T[i], 0.1);
    }
}

static void test_sine(void)
{
    enum { N = 64 };
    static const uint16_t sizes[] = { 1, 16, 1 };
    static const uint8_t acts[] = { MLP_ACT_TANH, MLP_ACT_LINEAR };
    float X[N], T[N], params[49], vel[49] = { 0.0f }, ws[64], wsf[16], x[1], y[1];
    uint32_t rng = 99u;
    double mse = 0.0;
    table_t tab;
    mlp_layer_t layers[2];
    mlp_model_t m;
    mlp_sgd_t opt;
    int i;

    for (i = 0; i < N; ++i) {
        X[i] = -1.0f + 2.0f * (float)i / (float)(N - 1);         /* x / pi */
        T[i] = (float)sin(3.14159265358979 * (double)X[i]);
    }
    memset(&tab, 0, sizeof tab);
    tab.x = X; tab.t = T; tab.n_in = 1; tab.n_out = 1;
    opt.lr = 0.003f; opt.momentum = 0.9f; opt.velocity = vel; opt.shuffle = &rng; opt.loss = MLP_LOSS_MSE;

    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 2, params) == MLP_OK);
    CHECK(mlp_init_params(&m, params, 3u) == MLP_OK);
    CHECK(train_until(&m, params, &opt, &tab, N, ws, 2e-5f, 3000) > 0);

    for (i = 0; i < 200; ++i) {
        x[0] = -1.0f + 2.0f * (float)i / 199.0f;
        CHECK(mlp_forward(&m, x, y, wsf) == MLP_OK);
        mse += pow((double)y[0] - sin(3.14159265358979 * (double)x[0]), 2.0);
    }
    mse /= 200.0;
    CHECK(mse < 2e-4);
}

/* Single-layer STEP network: backprop with the straight-through derivative is
   the classic perceptron learning rule, which converges on separable data. */
static void test_perceptron(void)
{
    static const float X[8] = { 0, 0, 0, 1, 1, 0, 1, 1 };
    static const float T_AND[4] = { 0, 0, 0, 1 }, T_OR[4] = { 0, 1, 1, 1 };
    static const uint16_t sizes[] = { 2, 1 };
    static const uint8_t acts[] = { MLP_ACT_STEP };
    const float *targets[2];
    float params[3], ws[16], y[1];
    table_t tab;
    mlp_layer_t layers[1];
    mlp_model_t m;
    mlp_sgd_t opt;
    int g, i;

    targets[0] = T_AND; targets[1] = T_OR;
    opt.lr = 0.1f; opt.momentum = 0.0f; opt.velocity = NULL; opt.shuffle = NULL; opt.loss = MLP_LOSS_MSE;
    for (g = 0; g < 2; ++g) {
        memset(&tab, 0, sizeof tab);
        tab.x = X; tab.t = targets[g]; tab.n_in = 2; tab.n_out = 1;
        memset(params, 0, sizeof params);
        CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 1, params) == MLP_OK);
        CHECK(train_until(&m, params, &opt, &tab, 4, ws, 0.0f, 50) < 50);
        for (i = 0; i < 4; ++i) {
            CHECK(mlp_forward(&m, X + 2 * i, y, NULL) == MLP_OK);
            CHECK(y[0] == targets[g][i]);
        }
    }
}

static void test_epoch_order(void)
{
    static const uint16_t sizes[] = { 1, 1 };
    static const uint8_t acts[] = { MLP_ACT_LINEAR };
    static const uint16_t counts[] = { 1, 2, 7, 10, 12, 16 };
    float X[16] = { 0.0f }, T[16] = { 0.0f }, params[2] = { 0.0f }, ws[8];
    uint32_t rng = 5u;
    table_t tab;
    mlp_layer_t layers[1];
    mlp_model_t m;
    mlp_sgd_t opt;
    unsigned c, i, rep;

    opt.lr = 0.01f; opt.momentum = 0.0f; opt.velocity = NULL; opt.shuffle = &rng; opt.loss = MLP_LOSS_MSE;
    CHECK(mlp_model_init_f32(&m, layers, sizes, acts, 1, params) == MLP_OK);
    for (c = 0; c < sizeof counts / sizeof counts[0]; ++c) {
        for (rep = 0; rep < 3; ++rep) {
            memset(&tab, 0, sizeof tab);
            tab.x = X; tab.t = T; tab.n_in = 1; tab.n_out = 1;
            CHECK(mlp_train_epoch(&m, params, &opt, table_get, &tab, counts[c], ws, NULL) == MLP_OK);
            for (i = 0; i < counts[c]; ++i) CHECK(tab.visits[i] == 1u);
        }
    }
}

static void test_train_errors(void)
{
    static const uint16_t sizes[] = { 2, 3 };
    static const uint8_t tanh_out[] = { MLP_ACT_TANH }, softmax_out[] = { MLP_ACT_SOFTMAX };
    float params[9] = { 0.0f }, ws[32];
    const float x[2] = { 0.0f, 1.0f }, t[3] = { 0.0f, 1.0f, 0.0f };
    mlp_layer_t layers[1];
    mlp_model_t m;
    mlp_sgd_t opt;

    opt.lr = 0.1f; opt.momentum = 0.0f; opt.velocity = NULL; opt.shuffle = NULL; opt.loss = MLP_LOSS_CE;
    CHECK(mlp_model_init_f32(&m, layers, sizes, tanh_out, 1, params) == MLP_OK);
    CHECK(mlp_train_sample(&m, params, &opt, x, t, ws, NULL) == MLP_ERR_ACT);
    opt.loss = 9;
    CHECK(mlp_train_sample(&m, params, &opt, x, t, ws, NULL) == MLP_ERR_ARG);
    opt.loss = MLP_LOSS_MSE;
    CHECK(mlp_train_sample(&m, params + 1, &opt, x, t, ws, NULL) == MLP_ERR_ARG);
    opt.momentum = 0.5f;
    CHECK(mlp_train_sample(&m, params, &opt, x, t, ws, NULL) == MLP_ERR_ARG);
    opt.momentum = 0.0f;
    CHECK(mlp_train_epoch(&m, params, &opt, failing_get, NULL, 3, ws, NULL) == MLP_ERR_IO);
    CHECK(mlp_model_init_f32(&m, layers, sizes, softmax_out, 1, params) == MLP_OK);
    CHECK(mlp_train_sample(&m, params, &opt, x, t, ws, NULL) == MLP_ERR_ACT);
    CHECK(mlp_loss(&m, MLP_LOSS_MSE, x, t) < 0.0f);
}

void test_train(void)
{
    test_gradients();
    test_init_and_momentum();
    test_xor(MLP_ACT_SIGMOID, MLP_LOSS_MSE);
    test_xor(MLP_ACT_SIGMOID, MLP_LOSS_CE);
    test_sine();
    test_perceptron();
    test_epoch_order();
    test_train_errors();
}

#else
void test_train(void) {}
#endif
