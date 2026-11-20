/* mlp.h - multilayer perceptron inference for microcontrollers.

   Author: Guilherme Arruda
   GitHub: https://github.com/ohananoshi/perceptron

   C99, no dynamic allocation: the caller provides every buffer, and the
   *_bytes() functions and MLP_*() macros say how large each must be.

   A model is a chain of fully connected layers. Each layer stores its weights
   as one contiguous row-major n_out x n_in array plus a separate bias array,
   and applies one activation to all of its outputs.

   Number types (mlp_model_t.dtype):
     MLP_F32  float weights, float activations.
     MLP_Q7   int8 weights, int16 activations, int32 accumulator.
     MLP_Q15  int16 weights, int16 activations, int32 accumulator.
   Fixed-point values are plain integers with a power-of-two scale per layer:
   real = q * 2^-frac. Weights use mlp_layer_t.w_frac, layer outputs use
   out_frac, the model input uses mlp_model_t.in_frac, and biases are int32 at
   the accumulator scale 2^-(w_frac + input frac).
*/
#ifndef MLP_H
#define MLP_H

#include <stddef.h>
#include <stdint.h>
#include "mlp_port.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MLP_VERSION_MAJOR 0
#define MLP_VERSION_MINOR 2

typedef enum {
    MLP_OK = 0,
    MLP_ERR_ARG,        /* NULL pointer, bad size or buffer too small */
    MLP_ERR_SHAPE,      /* layer sizes do not chain */
    MLP_ERR_DTYPE,      /* number type disabled in mlp_config.h or not allowed here */
    MLP_ERR_ACT,        /* activation not supported for this operation */
    MLP_ERR_FORMAT,     /* malformed or corrupted model blob */
    MLP_ERR_IO,         /* a user callback reported failure */
    MLP_ERR_RANGE       /* quantization cannot keep the accumulator in range */
} mlp_status_t;

/* Values are part of the blob format: append only. */
typedef enum {
    MLP_F32 = 0,
    MLP_Q7  = 1,
    MLP_Q15 = 2
} mlp_dtype_t;

/* Values are part of the blob format: append only. */
typedef enum {
    MLP_ACT_LINEAR     = 0,
    MLP_ACT_RELU       = 1,
    MLP_ACT_LEAKY_RELU = 2,
    MLP_ACT_SIGMOID    = 3,
    MLP_ACT_TANH       = 4,
    MLP_ACT_STEP       = 5,     /* 1 if x > 0 else 0 */
    MLP_ACT_SOFTMAX    = 6      /* float only, output layer only */
} mlp_act_t;

typedef struct {
    uint16_t n_in;
    uint16_t n_out;
    uint8_t  act;           /* mlp_act_t, kept as uint8_t so the layout does not depend on enum size */
    int8_t   w_frac;        /* fixed-point only: weight fractional bits */
    int8_t   out_frac;      /* fixed-point only: output fractional bits */
    const void *w;          /* n_out x n_in, row-major: float, int8_t or int16_t */
    const void *b;          /* n_out: float, or int32_t at accumulator scale */
} mlp_layer_t;

/* mlp_model_t.flags */
#define MLP_MODEL_ROM 0x01u     /* weights and biases are MLP_ROM data (matters on AVR only) */

typedef struct {
    const mlp_layer_t *layers;
    uint8_t n_layers;
    uint8_t dtype;          /* mlp_dtype_t */
    int8_t  in_frac;        /* fixed-point only: input fractional bits */
    uint8_t flags;
} mlp_model_t;

/* Fills x (n_inputs floats) and t (n_outputs floats) with sample `idx`.
   Used for training and calibration, so data can stream from flash, an SD
   card or a sensor instead of sitting in RAM. Return 0 on success. */
typedef int (*mlp_sample_fn)(void *ctx, uint16_t idx, float *x, float *t);

/* ---- Sizing --------------------------------------------------------------- */

/* Number of float parameters (weights + biases) of one layer. */
#define MLP_LAYER_PARAMS(n_in, n_out) ((uint32_t)(n_out) * ((uint32_t)(n_in) + 1u))

/* Bytes of one layer in a Q7 / Q15 model buffer: weights padded to 4 bytes,
   then int32 biases. Sum them to size a static buffer for mlp_quantize() or
   mlp_load(); declare it as int32_t[bytes / 4] for alignment. */
#define MLP_LAYER_BYTES_Q7(n_in, n_out) \
    ((((size_t)(n_in) * (size_t)(n_out) + 3u) & ~(size_t)3u) + 4u * (size_t)(n_out))
#define MLP_LAYER_BYTES_Q15(n_in, n_out) \
    ((((size_t)(n_in) * (size_t)(n_out) * 2u + 3u) & ~(size_t)3u) + 4u * (size_t)(n_out))

/* Elements of forward workspace: one buffer for one hidden layer, two
   (ping-pong) for more. max_hidden is the widest hidden layer.
   The element type is float for MLP_F32 and int16_t for MLP_Q7/MLP_Q15. */
#define MLP_FWD_WS_ELEMS(n_layers, max_hidden) \
    ((n_layers) <= 1 ? 1u : ((n_layers) == 2 ? (size_t)(max_hidden) : 2u * (size_t)(max_hidden)))

/* Total float parameters for `sizes` (n_layers + 1 entries: input, hidden..., output). */
uint32_t mlp_param_count(const uint16_t *sizes, uint8_t n_layers);

/* Bytes of a weight + bias buffer holding m's layers as `dtype` (each layer's
   weights, then its bias 4-byte aligned). Sizes the output of mlp_quantize()
   and the buffer for mlp_load(). For MLP_F32 this is mlp_param_count() * 4. */
size_t mlp_buf_bytes(const mlp_model_t *m, uint8_t dtype);

/* Forward workspace for mlp_forward(), in bytes (0 for a single layer). */
size_t mlp_forward_ws_bytes(const mlp_model_t *m);

uint16_t mlp_n_inputs(const mlp_model_t *m);
uint16_t mlp_n_outputs(const mlp_model_t *m);

/* ---- Model setup ---------------------------------------------------------- */

/* Build a float model on top of a flat parameter buffer, laid out as
   [W0, b0, W1, b1, ...] with mlp_param_count(sizes, n_layers) floats.
   `layers` must hold n_layers entries and stays referenced by the model.
   The parameters are not initialised: see mlp_init_params() in mlp_train.h. */
mlp_status_t mlp_model_init_f32(mlp_model_t *m, mlp_layer_t *layers,
                                const uint16_t *sizes, const uint8_t *acts,
                                uint8_t n_layers, float *params);

/* Validate layer chaining, dtype and activations. */
mlp_status_t mlp_model_check(const mlp_model_t *m);

/* ---- Inference ------------------------------------------------------------ */

/* Run the network. For MLP_F32, `in` and `out` are float arrays; for
   MLP_Q7/MLP_Q15 they are int16_t arrays in Q(in_frac) and Q(last out_frac).
   `ws` needs mlp_forward_ws_bytes(m) bytes, aligned for the element type, and
   may be NULL for a single-layer model. `in` and `out` must not overlap. */
mlp_status_t mlp_forward(const mlp_model_t *m, const void *in, void *out, void *ws);

/* Index of the largest output. */
uint16_t mlp_argmax(const mlp_model_t *m, const void *out);

/* Convert float inputs to the model's fixed-point input format and fixed-point
   outputs back to float. Return MLP_ERR_DTYPE for float models. */
mlp_status_t mlp_input_from_f32(const mlp_model_t *m, const float *x, int16_t *q);
mlp_status_t mlp_output_to_f32(const mlp_model_t *m, const int16_t *q, float *y);

#ifdef __cplusplus
}
#endif

#endif /* MLP_H */
