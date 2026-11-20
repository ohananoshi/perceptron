/* mlp_train.h - float32 training: online backpropagation with SGD.

   Works on models built with mlp_model_init_f32(); `params` is the same flat
   buffer that model was built on. Needs MLP_ENABLE_TRAIN.

   Losses:
     MLP_LOSS_MSE  0.5 * sum((y - t)^2), any output activation except softmax.
     MLP_LOSS_CE   cross-entropy, sigmoid or softmax output only.
*/
#ifndef MLP_TRAIN_H
#define MLP_TRAIN_H

#include "mlp.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MLP_LOSS_MSE = 0,
    MLP_LOSS_CE  = 1
} mlp_loss_t;

typedef struct {
    float lr;               /* learning rate */
    float momentum;         /* 0 for plain SGD */
    float *velocity;        /* mlp_param_count() floats, zeroed before the first step;
                               only needed when momentum != 0 */
    uint32_t *shuffle;      /* xorshift state for shuffling each epoch, or NULL for in-order */
    uint8_t loss;           /* mlp_loss_t */
} mlp_sgd_t;

/* xorshift32: tiny deterministic PRNG. *state must not be 0. */
uint32_t mlp_xorshift32(uint32_t *state);

/* Random weights (Xavier uniform, or He uniform before relu/leaky relu)
   and zero biases. The same seed always gives the same weights. */
mlp_status_t mlp_init_params(const mlp_model_t *m, float *params, uint32_t seed);

/* Workspace for mlp_train_sample() and mlp_train_epoch(), in bytes
   (float-aligned): every layer output, two delta buffers and one sample. */
size_t mlp_train_ws_bytes(const mlp_model_t *m);

/* One SGD step on sample (x, t). *loss (may be NULL) receives the loss before the step. */
mlp_status_t mlp_train_sample(const mlp_model_t *m, float *params, const mlp_sgd_t *opt,
                              const float *x, const float *t, void *ws, float *loss);

/* One pass over n samples fetched through `get`. *mean_loss (may be NULL)
   receives the average per-sample loss. A non-zero return from `get` aborts
   with MLP_ERR_IO. */
mlp_status_t mlp_train_epoch(const mlp_model_t *m, float *params, const mlp_sgd_t *opt,
                             mlp_sample_fn get, void *ctx, uint16_t n, void *ws, float *mean_loss);

/* Loss of output y against target t, as defined above; negative for invalid arguments. */
float mlp_loss(const mlp_model_t *m, uint8_t loss, const float *y, const float *t);

#ifdef __cplusplus
}
#endif

#endif /* MLP_TRAIN_H */
