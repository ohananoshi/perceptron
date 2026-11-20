/* test_io.c - blob round trips, corruption handling, and the generator for
   the exported-header test. */
#include <string.h>
#include "test.h"
#include "mlp_io.h"
#include "mlp_quant.h"
#include "mlp_train.h"

#if MLP_ENABLE_QUANT && MLP_ENABLE_Q7 && MLP_ENABLE_Q15

typedef struct {
    uint8_t *data;
    size_t cap, pos;
} mem_t;

static int mem_write(void *ctx, const uint8_t *d, uint16_t n)
{
    mem_t *m = (mem_t *)ctx;

    if (m->pos + n > m->cap) return -1;
    memcpy(m->data + m->pos, d, n);
    m->pos += n;
    return 0;
}

static int mem_read(void *ctx, uint8_t *d, uint16_t n)
{
    mem_t *m = (mem_t *)ctx;

    if (m->pos + n > m->cap) return -1;
    memcpy(d, m->data + m->pos, n);
    m->pos += n;
    return 0;
}

static int calib_get(void *ctx, uint16_t idx, float *x, float *t)
{
    uint32_t s = 1000u + idx;
    uint16_t i, n = *(const uint16_t *)ctx;

    for (i = 0; i < n; ++i) x[i] = test_frand(&s);
    t[0] = 0.0f;
    return 0;
}

/* A float model and its Q7 / Q15 versions: 3-5-2, tanh then sigmoid. */
typedef struct {
    float params[32];
    int32_t q7buf[32], q15buf[32];
    mlp_layer_t lf[2], l7[2], l15[2];
    mlp_model_t f, q7, q15;
} models_t;

static void build_models(models_t *mm)
{
    static const uint16_t sizes[] = { 3, 5, 2 };
    static const uint8_t acts[] = { MLP_ACT_TANH, MLP_ACT_SIGMOID };
    static const uint16_t n_in = 3;
    float ws[64];
    uint32_t s = 2024u, k;

    CHECK(mlp_model_init_f32(&mm->f, mm->lf, sizes, acts, 2, mm->params) == MLP_OK);
    for (k = 0; k < mlp_param_count(sizes, 2); ++k) mm->params[k] = 1.5f * test_frand(&s);
    CHECK(mlp_buf_bytes(&mm->f, MLP_Q15) <= sizeof mm->q15buf);
    CHECK(mlp_quantize(&mm->f, MLP_Q7, calib_get, (void *)&n_in, 32, ws, &mm->q7, mm->l7, mm->q7buf, sizeof mm->q7buf) == MLP_OK);
    CHECK(mlp_quantize(&mm->f, MLP_Q15, calib_get, (void *)&n_in, 32, ws, &mm->q15, mm->l15, mm->q15buf, sizeof mm->q15buf) == MLP_OK);
}

/* Save, load and check the loaded model gives identical outputs. */
static void round_trip(const mlp_model_t *m)
{
    uint8_t blob[512];
    int32_t buf[64];
    mlp_layer_t layers[4];
    mlp_model_t got;
    mem_t mem;
    float x[3], yf1[2], yf2[2], fws[8];
    int16_t xq[3], yq1[2], yq2[2], qws[8];
    uint32_t s = 9u;
    int n, i;

    mem.data = blob; mem.cap = sizeof blob; mem.pos = 0;
    CHECK(mlp_save(m, mem_write, &mem) == MLP_OK);
    CHECK(mem.pos == mlp_blob_bytes(m));
    mem.cap = mem.pos; mem.pos = 0;
    CHECK(mlp_load(mem_read, &mem, &got, layers, 4, buf, sizeof buf) == MLP_OK);
    CHECK(mem.pos == mem.cap);
    CHECK(got.dtype == m->dtype && got.in_frac == m->in_frac && got.n_layers == m->n_layers);

    for (n = 0; n < 8; ++n) {
        for (i = 0; i < 3; ++i) x[i] = test_frand(&s);
        if (m->dtype == MLP_F32) {
            CHECK(mlp_forward(m, x, yf1, fws) == MLP_OK);
            CHECK(mlp_forward(&got, x, yf2, fws) == MLP_OK);
            CHECK(memcmp(yf1, yf2, sizeof yf1) == 0);
        } else {
            CHECK(mlp_input_from_f32(m, x, xq) == MLP_OK);
            CHECK(mlp_forward(m, xq, yq1, qws) == MLP_OK);
            CHECK(mlp_forward(&got, xq, yq2, qws) == MLP_OK);
            CHECK(memcmp(yq1, yq2, sizeof yq1) == 0);
        }
    }

#if MLP_ENABLE_TRAIN
    if (m->dtype == MLP_F32) {                          /* loaded float models keep training */
        const float t[2] = { 0.0f, 1.0f };
        float tws[32];
        mlp_sgd_t opt;
        opt.lr = 0.1f; opt.momentum = 0.0f; opt.velocity = NULL; opt.shuffle = NULL; opt.loss = MLP_LOSS_MSE;
        CHECK(mlp_train_sample(&got, (float *)(void *)buf, &opt, x, t, tws, NULL) == MLP_OK);
    }
#endif
}

static void test_blob_errors(const mlp_model_t *m)
{
    uint8_t blob[512];
    int32_t buf[32];
    mlp_layer_t layers[4];
    mlp_model_t got;
    mem_t mem;
    size_t len;

    mem.data = blob; mem.cap = sizeof blob; mem.pos = 0;
    CHECK(mlp_save(m, mem_write, &mem) == MLP_OK);
    len = mem.pos;

    mem.cap = len; mem.pos = 0;
    blob[20] ^= 0x10u;                                  /* corrupt a data byte */
    CHECK(mlp_load(mem_read, &mem, &got, layers, 4, buf, sizeof buf) == MLP_ERR_FORMAT);
    blob[20] ^= 0x10u;

    mem.pos = 0;
    blob[0] = 'X';
    CHECK(mlp_load(mem_read, &mem, &got, layers, 4, buf, sizeof buf) == MLP_ERR_FORMAT);
    blob[0] = 'M';

    mem.cap = len - 3u; mem.pos = 0;                    /* truncated */
    CHECK(mlp_load(mem_read, &mem, &got, layers, 4, buf, sizeof buf) == MLP_ERR_IO);

    mem.cap = len; mem.pos = 0;
    CHECK(mlp_load(mem_read, &mem, &got, layers, 1, buf, sizeof buf) == MLP_ERR_ARG);
    mem.pos = 0;
    CHECK(mlp_load(mem_read, &mem, &got, layers, 4, buf, 8) == MLP_ERR_ARG);
    mem.pos = 0;
    CHECK(mlp_load(mem_read, &mem, &got, layers, 4, buf, sizeof buf) == MLP_OK);

    mem.cap = 10; mem.pos = 0;                          /* writer runs out of space */
    CHECK(mlp_save(m, mem_write, &mem) == MLP_ERR_IO);
}

/* MLP_MODEL_ROM switches to the ROM read paths (pgm_read_* on AVR). Built
   with -DMLP_HARVARD=1 on the host those paths run with plain reads, so they
   must match the RAM paths exactly, for inference and for mlp_save(). */
static void test_rom_paths(const mlp_model_t *m)
{
    uint8_t blob_ram[512], blob_rom[512];
    mlp_model_t rom = *m;
    mem_t a, b;
    float x[3], yf1[2], yf2[2], fws[8];
    int16_t xq[3], yq1[2], yq2[2], qws[8];
    uint32_t s = 77u;
    int n, i;

    rom.flags = MLP_MODEL_ROM;
    for (n = 0; n < 8; ++n) {
        for (i = 0; i < 3; ++i) x[i] = test_frand(&s);
        if (m->dtype == MLP_F32) {
            CHECK(mlp_forward(m, x, yf1, fws) == MLP_OK && mlp_forward(&rom, x, yf2, fws) == MLP_OK);
            CHECK(memcmp(yf1, yf2, sizeof yf1) == 0);
        } else {
            CHECK(mlp_input_from_f32(m, x, xq) == MLP_OK);
            CHECK(mlp_forward(m, xq, yq1, qws) == MLP_OK && mlp_forward(&rom, xq, yq2, qws) == MLP_OK);
            CHECK(memcmp(yq1, yq2, sizeof yq1) == 0);
        }
    }
    a.data = blob_ram; a.cap = sizeof blob_ram; a.pos = 0;
    b.data = blob_rom; b.cap = sizeof blob_rom; b.pos = 0;
    CHECK(mlp_save(m, mem_write, &a) == MLP_OK && mlp_save(&rom, mem_write, &b) == MLP_OK);
    CHECK(a.pos == b.pos && memcmp(blob_ram, blob_rom, a.pos) == 0);
}

void test_io(void)
{
    static models_t mm;

    build_models(&mm);
    test_rom_paths(&mm.f);
    test_rom_paths(&mm.q7);
    test_rom_paths(&mm.q15);
    round_trip(&mm.f);
    round_trip(&mm.q7);
    round_trip(&mm.q15);
    test_blob_errors(&mm.f);
    test_blob_errors(&mm.q7);
}

/* ---- Generator for tests/test_export.c ----------------------------------- */

#if MLP_ENABLE_STDIO
static void write_floats(FILE *f, const char *name, const float *v, unsigned n)
{
    unsigned i;

    fprintf(f, "static const float %s[%u] = {", name, n);
    for (i = 0; i < n; ++i) fprintf(f, "%s%.8ef", i ? ", " : " ", (double)v[i]);
    fputs(" };\n", f);
}

static void write_q(FILE *f, const char *name, const int16_t *v, unsigned n)
{
    unsigned i;

    fprintf(f, "static const int16_t %s[%u] = {", name, n);
    for (i = 0; i < n; ++i) fprintf(f, "%s%d", i ? ", " : " ", (int)v[i]);
    fputs(" };\n", f);
}

int test_export_generate(const char *path)
{
    enum { N = 6 };
    static models_t mm;
    float x[N * 3], yf[N * 2], fws[8];
    int16_t xq7[N * 3], yq7[N * 2], xq15[N * 3], yq15[N * 2], qws[8];
    uint32_t s = 4242u;
    FILE *f;
    int i, ok;

    build_models(&mm);
    for (i = 0; i < N * 3; ++i) x[i] = test_frand(&s);
    for (i = 0; i < N; ++i) {
        mlp_forward(&mm.f, x + 3 * i, yf + 2 * i, fws);
        mlp_input_from_f32(&mm.q7, x + 3 * i, xq7 + 3 * i);
        mlp_forward(&mm.q7, xq7 + 3 * i, yq7 + 2 * i, qws);
        mlp_input_from_f32(&mm.q15, x + 3 * i, xq15 + 3 * i);
        mlp_forward(&mm.q15, xq15 + 3 * i, yq15 + 2 * i, qws);
    }

    f = fopen(path, "w");
    if (f == NULL) return -1;
    ok = mlp_export_c(f, &mm.f, "gen_f32") == MLP_OK
      && mlp_export_c(f, &mm.q7, "gen_q7") == MLP_OK
      && mlp_export_c(f, &mm.q15, "gen_q15") == MLP_OK;
    fprintf(f, "\n#define GEN_N %d\n", N);
    write_floats(f, "gen_x", x, N * 3);
    write_floats(f, "gen_y_f32", yf, N * 2);
    write_q(f, "gen_x_q7", xq7, N * 3);
    write_q(f, "gen_y_q7", yq7, N * 2);
    write_q(f, "gen_x_q15", xq15, N * 3);
    write_q(f, "gen_y_q15", yq15, N * 2);
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}
#else
int test_export_generate(const char *path) { (void)path; return -1; }
#endif

#else
void test_io(void) {}
int test_export_generate(const char *path) { (void)path; return -1; }
#endif
