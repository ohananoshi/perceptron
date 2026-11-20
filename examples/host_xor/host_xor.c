/* host_xor.c - train XOR on the PC, quantize it to Q7 and export a header
   that an MCU compiles straight into flash.

   Usage: host_xor [header.h]     (examples/arduino_infer/xor_model.h is made this way)
*/
#include <stdio.h>
#include <string.h>
#include "mlp.h"
#include "mlp_io.h"
#include "mlp_quant.h"
#include "mlp_train.h"

static const float X[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
static const float T[4] = { 0, 1, 1, 0 };

static int xor_sample(void *ctx, uint16_t idx, float *x, float *t)
{
    (void)ctx;
    x[0] = X[idx][0];
    x[1] = X[idx][1];
    t[0] = T[idx];
    return 0;
}

int main(int argc, char **argv)
{
    static const uint16_t sizes[] = { 2, 4, 1 };
    static const uint8_t acts[] = { MLP_ACT_TANH, MLP_ACT_SIGMOID };

    /* Every buffer is static and sized up front: no malloc anywhere. */
    static float params[MLP_LAYER_PARAMS(2, 4) + MLP_LAYER_PARAMS(4, 1)];
    static float velocity[sizeof params / sizeof params[0]];
    static float train_ws[32], fwd_ws[MLP_FWD_WS_ELEMS(2, 4)];
    static int32_t qbuf[(MLP_LAYER_BYTES_Q7(2, 4) + MLP_LAYER_BYTES_Q7(4, 1)) / 4];
    static int16_t qws[MLP_FWD_WS_ELEMS(2, 4)];
    mlp_layer_t layers[2], qlayers[2];
    mlp_model_t net, qnet;
    uint32_t rng = 1234u;
    mlp_sgd_t opt;
    float loss = 1.0f;
    int epoch, i;

    if (mlp_model_init_f32(&net, layers, sizes, acts, 2, params) != MLP_OK
        || mlp_train_ws_bytes(&net) > sizeof train_ws
        || mlp_quant_ws_bytes(&net) > sizeof train_ws
        || mlp_init_params(&net, params, 7u) != MLP_OK)
        return 1;

    opt.lr = 0.2f;
    opt.momentum = 0.9f;
    opt.velocity = velocity;
    opt.shuffle = &rng;
    opt.loss = MLP_LOSS_CE;
    for (epoch = 0; epoch < 5000 && loss > 1e-3f; ++epoch)
        if (mlp_train_epoch(&net, params, &opt, xor_sample, NULL, 4, train_ws, &loss) != MLP_OK) return 1;
    printf("trained in %d epochs, mean loss %.5f\n", epoch, (double)loss);

    if (mlp_quantize(&net, MLP_Q7, xor_sample, NULL, 4, train_ws, &qnet, qlayers, qbuf, sizeof qbuf) != MLP_OK)
        return 1;

    printf("  x1 x2 | float32 | Q7\n");
    for (i = 0; i < 4; ++i) {
        float y, yq_f;
        int16_t xq[2], yq;

        mlp_forward(&net, X[i], &y, fwd_ws);
        mlp_input_from_f32(&qnet, X[i], xq);
        mlp_forward(&qnet, xq, &yq, qws);
        mlp_output_to_f32(&qnet, &yq, &yq_f);
        printf("  %.0f  %.0f  | %.4f  | %.4f\n", (double)X[i][0], (double)X[i][1], (double)y, (double)yq_f);
    }
    printf("weights: float32 %lu bytes, Q7 %lu bytes; inference workspace %lu bytes\n",
           (unsigned long)mlp_buf_bytes(&net, MLP_F32), (unsigned long)mlp_buf_bytes(&qnet, MLP_Q7),
           (unsigned long)mlp_forward_ws_bytes(&qnet));

    if (argc > 1) {
        FILE *f = fopen(argv[1], "w");
        if (f == NULL || mlp_export_c(f, &qnet, "xor_model") != MLP_OK || fclose(f) != 0) {
            fprintf(stderr, "cannot write %s\n", argv[1]);
            return 1;
        }
        printf("wrote %s\n", argv[1]);
    }
    return 0;
}
