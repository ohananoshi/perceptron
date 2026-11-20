/* host_sine.c - fit sin(x) on [-pi, pi] with a 1-16-1 network, then compare
   the float32, Q15 and Q7 versions. (This replaces the old teste.c experiment.)

   Usage: host_sine [out.csv]     CSV columns: x, sin(x), float32, Q15, Q7
*/
#include <math.h>
#include <stdio.h>
#include "mlp.h"
#include "mlp_quant.h"
#include "mlp_train.h"

#define N_TRAIN 64
#define N_EVAL 200
#define PI 3.14159265358979

/* Input is x / pi, so the network sees [-1, 1]. */
static int sine_sample(void *ctx, uint16_t idx, float *x, float *t)
{
    (void)ctx;
    x[0] = -1.0f + 2.0f * (float)idx / (float)(N_TRAIN - 1);
    t[0] = (float)sin(PI * (double)x[0]);
    return 0;
}

int main(int argc, char **argv)
{
    static const uint16_t sizes[] = { 1, 16, 1 };
    static const uint8_t acts[] = { MLP_ACT_TANH, MLP_ACT_LINEAR };
    static float params[MLP_LAYER_PARAMS(1, 16) + MLP_LAYER_PARAMS(16, 1)];
    static float velocity[sizeof params / sizeof params[0]];
    static float train_ws[64], fwd_ws[MLP_FWD_WS_ELEMS(2, 16)];
    static int32_t q15buf[(MLP_LAYER_BYTES_Q15(1, 16) + MLP_LAYER_BYTES_Q15(16, 1)) / 4];
    static int32_t q7buf[(MLP_LAYER_BYTES_Q7(1, 16) + MLP_LAYER_BYTES_Q7(16, 1)) / 4];
    static int16_t qws[MLP_FWD_WS_ELEMS(2, 16)];
    mlp_layer_t layers[2], l15[2], l7[2];
    mlp_model_t net, q15, q7;
    double se[3] = { 0.0, 0.0, 0.0 };
    uint32_t rng = 99u;
    mlp_sgd_t opt;
    float loss = 1.0f;
    FILE *csv = NULL;
    int epoch, i;

    if (mlp_model_init_f32(&net, layers, sizes, acts, 2, params) != MLP_OK
        || mlp_train_ws_bytes(&net) > sizeof train_ws
        || mlp_quant_ws_bytes(&net) > sizeof train_ws
        || mlp_init_params(&net, params, 3u) != MLP_OK)
        return 1;

    opt.lr = 0.003f;
    opt.momentum = 0.9f;
    opt.velocity = velocity;
    opt.shuffle = &rng;
    opt.loss = MLP_LOSS_MSE;
    for (epoch = 0; epoch < 3000 && loss > 1e-5f; ++epoch)
        if (mlp_train_epoch(&net, params, &opt, sine_sample, NULL, N_TRAIN, train_ws, &loss) != MLP_OK) return 1;
    printf("trained in %d epochs, mean loss %.2e\n", epoch, (double)loss);

    if (mlp_quantize(&net, MLP_Q15, sine_sample, NULL, N_TRAIN, train_ws, &q15, l15, q15buf, sizeof q15buf) != MLP_OK
        || mlp_quantize(&net, MLP_Q7, sine_sample, NULL, N_TRAIN, train_ws, &q7, l7, q7buf, sizeof q7buf) != MLP_OK)
        return 1;

    if (argc > 1 && (csv = fopen(argv[1], "w")) == NULL) {
        fprintf(stderr, "cannot write %s\n", argv[1]);
        return 1;
    }
    if (csv != NULL) fputs("x,sin,f32,q15,q7\n", csv);

    for (i = 0; i < N_EVAL; ++i) {
        const float x = -1.0f + 2.0f * (float)i / (float)(N_EVAL - 1);
        const double ref = sin(PI * (double)x);
        float y[3];
        int16_t xq[1], yq[1];

        mlp_forward(&net, &x, &y[0], fwd_ws);
        mlp_input_from_f32(&q15, &x, xq);
        mlp_forward(&q15, xq, yq, qws);
        mlp_output_to_f32(&q15, yq, &y[1]);
        mlp_input_from_f32(&q7, &x, xq);
        mlp_forward(&q7, xq, yq, qws);
        mlp_output_to_f32(&q7, yq, &y[2]);

        se[0] += ((double)y[0] - ref) * ((double)y[0] - ref);
        se[1] += ((double)y[1] - ref) * ((double)y[1] - ref);
        se[2] += ((double)y[2] - ref) * ((double)y[2] - ref);
        if (csv != NULL)
            fprintf(csv, "%.5f,%.6f,%.6f,%.6f,%.6f\n", PI * (double)x, ref,
                    (double)y[0], (double)y[1], (double)y[2]);
    }
    if (csv != NULL && fclose(csv) != 0) return 1;

    printf("MSE vs sin(x) over %d points: float32 %.2e, Q15 %.2e, Q7 %.2e\n",
           N_EVAL, se[0] / N_EVAL, se[1] / N_EVAL, se[2] / N_EVAL);
    printf("weights: float32 %lu bytes, Q15 %lu bytes, Q7 %lu bytes\n",
           (unsigned long)mlp_buf_bytes(&net, MLP_F32), (unsigned long)mlp_buf_bytes(&net, MLP_Q15),
           (unsigned long)mlp_buf_bytes(&net, MLP_Q7));
    return 0;
}
