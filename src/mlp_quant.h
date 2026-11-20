/* mlp_quant.h - float32 model -> Q7 / Q15 model. Needs MLP_ENABLE_QUANT.

   Usually run on the PC after training, followed by mlp_export_c(), but it
   has no host dependencies and also runs on the device.

   Formats are chosen per layer:
     input / relu / leaky / linear outputs: from the largest magnitude seen on
       the calibration samples, with 25% headroom (values beyond saturate);
     sigmoid and tanh outputs: Q15; step outputs: Q14;
     weights: the most fractional bits that fit, then reduced until the
       worst-case accumulator |bias| + sum|w| * 32768 stays below 2^30 for
       every neuron, so no int16 input can overflow the int32 accumulator.
   A softmax output layer becomes linear: the fixed-point model returns
   logits, which have the same argmax.
*/
#ifndef MLP_QUANT_H
#define MLP_QUANT_H

#include "mlp.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Workspace for mlp_quantize(), in bytes (float-aligned). */
size_t mlp_quant_ws_bytes(const mlp_model_t *src);

/* Quantize float model `src` to `dtype` (MLP_Q7 or MLP_Q15).
   calib/ctx/n_calib: representative inputs (targets are ignored), at least one.
   dst_layers: src->n_layers entries. buf: buf_bytes >= mlp_buf_bytes(src, dtype),
   4-byte aligned (declare it as an int32_t array). Both stay referenced by *dst.
   Returns MLP_ERR_ARG if buf is too small. */
mlp_status_t mlp_quantize(const mlp_model_t *src, uint8_t dtype,
                          mlp_sample_fn calib, void *ctx, uint16_t n_calib, void *ws,
                          mlp_model_t *dst, mlp_layer_t *dst_layers, void *buf, size_t buf_bytes);

#ifdef __cplusplus
}
#endif

#endif /* MLP_QUANT_H */
