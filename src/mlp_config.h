/* mlp_config.h - compile-time feature switches.

   Every option has a default here. Override them either with -D flags or by
   pointing MLP_CONFIG_FILE at your own header:

       -DMLP_CONFIG_FILE=\"my_mlp_config.h\"

   Disabled features are compiled out entirely, so an inference-only build
   for a small MCU can drop training, quantization and unused number types.
*/
#ifndef MLP_CONFIG_H
#define MLP_CONFIG_H

#ifdef MLP_CONFIG_FILE
#include MLP_CONFIG_FILE
#endif

/* Number types available to mlp_forward(). */
#ifndef MLP_ENABLE_F32
#define MLP_ENABLE_F32 1        /* float32 weights and activations */
#endif
#ifndef MLP_ENABLE_Q7
#define MLP_ENABLE_Q7 1         /* int8 weights, int16 activations */
#endif
#ifndef MLP_ENABLE_Q15
#define MLP_ENABLE_Q15 1        /* int16 weights, int16 activations */
#endif

/* On-device training (float32 backprop). Requires MLP_ENABLE_F32. */
#ifndef MLP_ENABLE_TRAIN
#define MLP_ENABLE_TRAIN 1
#endif

/* float -> Q7/Q15 conversion. Requires MLP_ENABLE_F32. */
#ifndef MLP_ENABLE_QUANT
#define MLP_ENABLE_QUANT 1
#endif

/* mlp_export_c(), which needs <stdio.h>. Off by default: MCUs rarely want it. */
#ifndef MLP_ENABLE_STDIO
#define MLP_ENABLE_STDIO 0
#endif

/* Replace expf() with a bit-level approximation (max relative error ~1e-4).
   Worth it on chips without an FPU, where expf() is a long soft-float routine. */
#ifndef MLP_FAST_MATH
#define MLP_FAST_MATH 0
#endif

/* Unroll dot products 4x. Helps 32-bit cores with pipelined FPUs/MACs;
   costs a little flash. Set to 0 on 8-bit parts to save code size. */
#ifndef MLP_UNROLL
#if defined(__AVR__)
#define MLP_UNROLL 0
#else
#define MLP_UNROLL 1
#endif
#endif

/* Negative slope of MLP_ACT_LEAKY_RELU. */
#ifndef MLP_LEAKY_ALPHA
#define MLP_LEAKY_ALPHA 0.01f
#endif

/* Debug checks (MLP_ASSERT) and trace output (MLP_LOG). */
#ifndef MLP_DEBUG
#define MLP_DEBUG 0
#endif

#if (MLP_ENABLE_TRAIN || MLP_ENABLE_QUANT) && !MLP_ENABLE_F32
#error "MLP_ENABLE_TRAIN and MLP_ENABLE_QUANT require MLP_ENABLE_F32"
#endif

#endif /* MLP_CONFIG_H */
