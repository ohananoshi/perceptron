/* test_export.c - second stage: compile the header written by mlp_export_c()
   and check the exported models reproduce the in-memory outputs exactly. */
#include <stdio.h>
#include <string.h>
#include "mlp.h"
#include "gen_model.h"

static int g_fails = 0;

static void check(int ok, const char *what, int i)
{
    if (!ok) {
        ++g_fails;
        fprintf(stderr, "test_export: %s mismatch at sample %d\n", what, i);
    }
}

int main(void)
{
    float y[2], fws[8];
    int16_t q[2], qws[8];
    int i;

    check(gen_f32.flags == MLP_MODEL_ROM && gen_q7.dtype == MLP_Q7 && gen_q15.dtype == MLP_Q15, "header", -1);
    for (i = 0; i < GEN_N; ++i) {
        check(mlp_forward(&gen_f32, gen_x + 3 * i, y, fws) == MLP_OK
              && memcmp(y, gen_y_f32 + 2 * i, sizeof y) == 0, "f32", i);
        check(mlp_forward(&gen_q7, gen_x_q7 + 3 * i, q, qws) == MLP_OK
              && memcmp(q, gen_y_q7 + 2 * i, sizeof q) == 0, "q7", i);
        check(mlp_forward(&gen_q15, gen_x_q15 + 3 * i, q, qws) == MLP_OK
              && memcmp(q, gen_y_q15 + 2 * i, sizeof q) == 0, "q15", i);
    }
    printf("test_export: %d samples x 3 models, %d failed\n", GEN_N, g_fails);
    return g_fails == 0 ? 0 : 1;
}
