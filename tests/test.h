/* test.h - minimal test harness and shared reference helpers. */
#ifndef TEST_H
#define TEST_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include "mlp.h"

extern int g_checks;
extern int g_fails;

#define CHECK(c) do {                                                         \
        ++g_checks;                                                           \
        if (!(c)) {                                                           \
            ++g_fails;                                                        \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #c); \
        }                                                                     \
    } while (0)

#define CHECK_NEAR(a, b, tol) do {                                            \
        const double a_ = (double)(a), b_ = (double)(b);                      \
        ++g_checks;                                                           \
        if (!(fabs(a_ - b_) <= (double)(tol))) {                              \
            ++g_fails;                                                        \
            fprintf(stderr, "%s:%d: |%s - %s| = |%.9g - %.9g| > %g\n",        \
                    __FILE__, __LINE__, #a, #b, a_, b_, (double)(tol));       \
        }                                                                     \
    } while (0)

/* Widest layer the double reference supports. */
#define REF_MAX 64

/* Deterministic uniform float in [-1, 1). */
float test_frand(uint32_t *state);

/* Plain double-precision forward pass over a flat [W0,b0,W1,b1,...] buffer:
   the "obviously correct" reference the library is checked against. */
void ref_forward(const uint16_t *sizes, const uint8_t *acts, uint8_t n_layers,
                 const double *params, const double *x, double *y);

/* Loss matching the library's definitions (0.5*sum sq. error, or cross-entropy). */
double ref_loss(uint8_t loss, uint8_t out_act, const double *y, const double *t, uint16_t n);

void test_forward(void);
void test_fixed(void);
void test_train(void);
void test_quant(void);
void test_io(void);
int test_export_generate(const char *path);

#endif /* TEST_H */
