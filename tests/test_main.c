/* test_main.c - runs every suite. With a path argument it also writes the
   exported model header used by the second-stage test (test_export.c). */
#include "test.h"

int g_checks = 0;
int g_fails = 0;

float test_frand(uint32_t *state)
{
    uint32_t x = *state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return (float)(x >> 8) * (1.0f / 8388608.0f) - 1.0f;
}

static double ref_act(uint8_t act, double z)
{
    switch (act) {
    case MLP_ACT_RELU:       return z > 0.0 ? z : 0.0;
    case MLP_ACT_LEAKY_RELU: return z > 0.0 ? z : (double)MLP_LEAKY_ALPHA * z;
    case MLP_ACT_SIGMOID:    return 1.0 / (1.0 + exp(-z));
    case MLP_ACT_TANH:       return tanh(z);
    case MLP_ACT_STEP:       return z > 0.0 ? 1.0 : 0.0;
    default:                 return z;
    }
}

void ref_forward(const uint16_t *sizes, const uint8_t *acts, uint8_t n_layers,
                 const double *params, const double *x, double *y)
{
    double a[REF_MAX], z[REF_MAX];
    const double *p = params;
    unsigned l, i, j;

    for (i = 0; i < sizes[0]; ++i) a[i] = x[i];
    for (l = 0; l < n_layers; ++l) {
        const unsigned n_in = sizes[l], n_out = sizes[l + 1u];
        const double *b = p + n_in * n_out;

        for (i = 0; i < n_out; ++i) {
            double s = b[i];
            for (j = 0; j < n_in; ++j) s += p[i * n_in + j] * a[j];
            z[i] = s;
        }
        if (acts[l] == MLP_ACT_SOFTMAX) {
            double max = z[0], sum = 0.0;
            for (i = 1; i < n_out; ++i) if (z[i] > max) max = z[i];
            for (i = 0; i < n_out; ++i) { z[i] = exp(z[i] - max); sum += z[i]; }
            for (i = 0; i < n_out; ++i) a[i] = z[i] / sum;
        } else {
            for (i = 0; i < n_out; ++i) a[i] = ref_act(acts[l], z[i]);
        }
        p = b + n_out;
    }
    for (i = 0; i < sizes[n_layers]; ++i) y[i] = a[i];
}

double ref_loss(uint8_t loss, uint8_t out_act, const double *y, const double *t, uint16_t n)
{
    double s = 0.0;
    unsigned i;

    for (i = 0; i < n; ++i) {
        if (loss == 0) {                            /* MLP_LOSS_MSE */
            s += 0.5 * (y[i] - t[i]) * (y[i] - t[i]);
        } else if (out_act == MLP_ACT_SOFTMAX) {
            s -= t[i] * log(y[i]);
        } else {
            s -= t[i] * log(y[i]) + (1.0 - t[i]) * log(1.0 - y[i]);
        }
    }
    return s;
}

int main(int argc, char **argv)
{
    test_forward();
    test_fixed();
    test_train();
    test_quant();
    test_io();

    if (argc > 1 && test_export_generate(argv[1]) != 0) {
        fprintf(stderr, "failed to write %s\n", argv[1]);
        return 1;
    }

    printf("%s%s%s: %d checks, %d failed\n", argv[0], MLP_FAST_MATH ? " (MLP_FAST_MATH)" : "",
           MLP_HARVARD ? " (MLP_HARVARD)" : "", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
