/* mlp_internal.h - helpers shared between the library's .c files. Not public API. */
#ifndef MLP_INTERNAL_H
#define MLP_INTERNAL_H

#include "mlp.h"

/* Widest hidden layer (0 for a single-layer model). */
uint16_t mlpi_max_hidden(const mlp_model_t *m);

/* Widest layer output, hidden or not. */
uint16_t mlpi_max_width(const mlp_model_t *m);

/* Sum of n_out over all layers. */
uint32_t mlpi_total_outputs(const mlp_model_t *m);

int mlpi_dtype_enabled(uint8_t dtype);

/* Bytes one layer's weights + bias occupy in a model buffer, with the bias
   4-byte aligned. *b_off (if not NULL) receives the bias offset. */
size_t mlpi_layer_bytes(uint16_t n_in, uint16_t n_out, uint8_t dtype, size_t *b_off);

/* 2^e as a float, built directly in the exponent field (e clamped to [-126, 127]). */
float mlpi_pow2f(int e);

#if MLP_ENABLE_F32
/* y = act(W x + b) for one float layer. */
void mlpi_layer_f32(const mlp_layer_t *L, const float *x, float *y, uint8_t rom);

void mlpi_forward_f32(const mlp_model_t *m, const float *in, float *out, float *ws);

/* Forward pass keeping every layer's output: acts needs mlpi_total_outputs() floats,
   layer l's output starts after the outputs of layers 0..l-1. */
void mlpi_forward_f32_all(const mlp_model_t *m, const float *in, float *acts);
#endif

#if MLP_ENABLE_Q7 || MLP_ENABLE_Q15
mlp_status_t mlpi_forward_q(const mlp_model_t *m, const int16_t *in, int16_t *out, int16_t *ws);
#endif

#endif /* MLP_INTERNAL_H */
