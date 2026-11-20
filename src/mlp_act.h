/* mlp_act.h - float activation functions. Not public API.

   Derivatives are written in terms of the activation's output y, so training
   only has to keep layer outputs, never the pre-activation values.
*/
#ifndef MLP_ACT_H
#define MLP_ACT_H

#include <math.h>
#include "mlp.h"

#if MLP_FAST_MATH
/* exp(x) = 2^n * 2^r with n = floor(x / ln 2) and r in [0, 1).
   2^n is written straight into the IEEE-754 exponent field through a union,
   2^r comes from a cubic fitted with p(0) = 1, p(1) = 2 (max relative error 1.03e-4). */
MLP_INLINE float mlp_expf(float x)
{
    union { float f; uint32_t u; } v;
    float t, r;
    int32_t n;

    if (x < -87.0f) x = -87.0f;         /* keep 2^n a normal number */
    if (x > 88.0f) x = 88.0f;
    t = x * 1.44269504f;
    n = (int32_t)t;
    if ((float)n > t) --n;              /* truncation -> floor for negative t */
    r = t - (float)n;
    v.u = (uint32_t)(n + 127) << 23;
    return v.f * (1.0f + r * (0.6954244f + r * (0.22630766f + r * 0.07826794f)));
}

MLP_INLINE float mlp_tanhf(float x)
{
    return 2.0f / (1.0f + mlp_expf(-2.0f * x)) - 1.0f;
}
#else
MLP_INLINE float mlp_expf(float x) { return expf(x); }
MLP_INLINE float mlp_tanhf(float x) { return tanhf(x); }
#endif

MLP_INLINE float mlp_sigmoidf(float x)
{
    return 1.0f / (1.0f + mlp_expf(-x));
}

/* Apply `act` in place to y[0..n). One switch per layer, tight loop per case. */
MLP_INLINE void mlp_act_apply(uint8_t act, float *y, uint_fast16_t n)
{
    uint_fast16_t i;

    switch (act) {
    case MLP_ACT_RELU:
        for (i = 0; i < n; ++i) if (y[i] < 0.0f) y[i] = 0.0f;
        break;
    case MLP_ACT_LEAKY_RELU:
        for (i = 0; i < n; ++i) if (y[i] < 0.0f) y[i] *= MLP_LEAKY_ALPHA;
        break;
    case MLP_ACT_SIGMOID:
        for (i = 0; i < n; ++i) y[i] = mlp_sigmoidf(y[i]);
        break;
    case MLP_ACT_TANH:
        for (i = 0; i < n; ++i) y[i] = mlp_tanhf(y[i]);
        break;
    case MLP_ACT_STEP:
        for (i = 0; i < n; ++i) y[i] = y[i] > 0.0f ? 1.0f : 0.0f;
        break;
    case MLP_ACT_SOFTMAX: {
        float max = y[0], sum = 0.0f, inv;
        for (i = 1; i < n; ++i) if (y[i] > max) max = y[i];
        for (i = 0; i < n; ++i) { y[i] = mlp_expf(y[i] - max); sum += y[i]; }
        inv = 1.0f / sum;
        for (i = 0; i < n; ++i) y[i] *= inv;
        break;
    }
    default:                            /* MLP_ACT_LINEAR */
        break;
    }
}

/* d(act)/dx at the point where act(x) = y.
   STEP returns 1 (straight-through): for a single-layer network this turns
   backprop into the classic perceptron learning rule.
   SOFTMAX is only trained with cross-entropy, which never calls this. */
MLP_INLINE float mlp_act_deriv(uint8_t act, float y)
{
    switch (act) {
    case MLP_ACT_RELU:       return y > 0.0f ? 1.0f : 0.0f;
    case MLP_ACT_LEAKY_RELU: return y > 0.0f ? 1.0f : MLP_LEAKY_ALPHA;
    case MLP_ACT_SIGMOID:    return y * (1.0f - y);
    case MLP_ACT_TANH:       return 1.0f - y * y;
    default:                 return 1.0f;
    }
}

#endif /* MLP_ACT_H */
