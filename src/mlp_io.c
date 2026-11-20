/* mlp_io.c - blob save/load and C header export. */
#include <string.h>
#include "mlp_io.h"
#include "mlp_internal.h"

#define BLOB_VERSION 1u
#define BLOB_HEADER 8u
#define BLOB_LAYER 7u

static uint8_t mlpi_wsize(uint8_t dtype)
{
    return dtype == MLP_Q7 ? 1u : (dtype == MLP_Q15 ? 2u : 4u);
}

/* Raw bits of element i of a weight (esize 1/2/4) or bias (esize 4) array. */
static uint32_t mlpi_elem_bits(const void *base, uint32_t i, uint8_t esize, uint8_t is_float, uint8_t rom)
{
#if !MLP_HARVARD
    (void)rom;
#define MLPI_RD(rd, p) (*(p))
#else
#define MLPI_RD(rd, p) (rom ? rd(p) : *(p))
#endif
    if (esize == 1u) return (uint8_t)MLPI_RD(MLP_RD_I8, (const int8_t *)base + i);
    if (esize == 2u) return (uint16_t)MLPI_RD(MLP_RD_I16, (const int16_t *)base + i);
    if (is_float) {
        const float f = MLPI_RD(MLP_RD_F32, (const float *)base + i);
        uint32_t u;
        memcpy(&u, &f, sizeof u);
        return u;
    }
    return (uint32_t)MLPI_RD(MLP_RD_I32, (const int32_t *)base + i);
#undef MLPI_RD
}

size_t mlp_blob_bytes(const mlp_model_t *m)
{
    size_t n = BLOB_HEADER + 2u;
    uint_fast8_t l;

    if (m == NULL || m->layers == NULL) return 0;
    for (l = 0; l < m->n_layers; ++l) {
        const mlp_layer_t *L = &m->layers[l];
        n += BLOB_LAYER + (size_t)L->n_in * L->n_out * mlpi_wsize(m->dtype) + (size_t)L->n_out * 4u;
    }
    return n;
}

/* ---- Writing ------------------------------------------------------------- */

typedef struct {
    mlp_write_fn wr;
    void *ctx;
    uint16_t s1, s2;
    int err;
} mlpi_writer_t;

static void mlpi_put(mlpi_writer_t *w, const uint8_t *d, uint16_t n)
{
    uint16_t i;

    if (w->err) return;
    for (i = 0; i < n; ++i) {
        w->s1 = (uint16_t)((w->s1 + d[i]) % 255u);
        w->s2 = (uint16_t)((w->s2 + w->s1) % 255u);
    }
    if (w->wr(w->ctx, d, n) != 0) w->err = 1;
}

static void mlpi_put_le(mlpi_writer_t *w, uint32_t v, uint8_t n)
{
    uint8_t b[4];
    uint8_t i;

    for (i = 0; i < n; ++i) b[i] = (uint8_t)(v >> (8u * i));
    mlpi_put(w, b, n);
}

mlp_status_t mlp_save(const mlp_model_t *m, mlp_write_fn wr, void *ctx)
{
    static const uint8_t magic[4] = { 'M', 'L', 'P', '1' };
    const uint8_t rom = (uint8_t)(m != NULL ? m->flags & MLP_MODEL_ROM : 0u);
    mlpi_writer_t w;
    uint8_t sum[2];
    uint_fast8_t l;
    mlp_status_t st;

    if (m == NULL || wr == NULL) return MLP_ERR_ARG;
    st = mlp_model_check(m);
    if (st != MLP_OK) return st;

    w.wr = wr; w.ctx = ctx; w.s1 = 0; w.s2 = 0; w.err = 0;
    mlpi_put(&w, magic, 4);
    mlpi_put_le(&w, BLOB_VERSION, 1);
    mlpi_put_le(&w, m->dtype, 1);
    mlpi_put_le(&w, m->n_layers, 1);
    mlpi_put_le(&w, (uint8_t)m->in_frac, 1);
    for (l = 0; l < m->n_layers; ++l) {
        const mlp_layer_t *L = &m->layers[l];
        mlpi_put_le(&w, L->n_in, 2);
        mlpi_put_le(&w, L->n_out, 2);
        mlpi_put_le(&w, L->act, 1);
        mlpi_put_le(&w, (uint8_t)L->w_frac, 1);
        mlpi_put_le(&w, (uint8_t)L->out_frac, 1);
    }
    for (l = 0; l < m->n_layers && !w.err; ++l) {
        const mlp_layer_t *L = &m->layers[l];
        const uint8_t es = mlpi_wsize(m->dtype), is_float = m->dtype == MLP_F32;
        const uint32_t nw = (uint32_t)L->n_in * L->n_out;
        uint32_t i;

        for (i = 0; i < nw; ++i) mlpi_put_le(&w, mlpi_elem_bits(L->w, i, es, is_float, rom), es);
        for (i = 0; i < L->n_out; ++i) mlpi_put_le(&w, mlpi_elem_bits(L->b, i, 4u, is_float, rom), 4u);
    }
    sum[0] = (uint8_t)w.s1;
    sum[1] = (uint8_t)w.s2;
    if (!w.err && wr(ctx, sum, 2) != 0) w.err = 1;
    return w.err ? MLP_ERR_IO : MLP_OK;
}

/* ---- Reading ------------------------------------------------------------- */

typedef struct {
    mlp_read_fn rd;
    void *ctx;
    uint16_t s1, s2;
    int err;
} mlpi_reader_t;

static void mlpi_get(mlpi_reader_t *r, uint8_t *d, uint16_t n)
{
    uint16_t i;

    if (r->err || r->rd(r->ctx, d, n) != 0) {
        r->err = 1;
        memset(d, 0, n);
        return;
    }
    for (i = 0; i < n; ++i) {
        r->s1 = (uint16_t)((r->s1 + d[i]) % 255u);
        r->s2 = (uint16_t)((r->s2 + r->s1) % 255u);
    }
}

static uint32_t mlpi_get_le(mlpi_reader_t *r, uint8_t n)
{
    uint8_t b[4];
    uint32_t v = 0;
    uint8_t i;

    mlpi_get(r, b, n);
    for (i = 0; i < n; ++i) v |= (uint32_t)b[i] << (8u * i);
    return v;
}

mlp_status_t mlp_load(mlp_read_fn rd, void *ctx, mlp_model_t *m,
                      mlp_layer_t *layers, uint8_t max_layers, void *buf, size_t buf_bytes)
{
    mlpi_reader_t r;
    uint8_t hdr[BLOB_HEADER], sum[2];
    uint8_t *p = (uint8_t *)buf;
    size_t need = 0;
    uint8_t dtype, n_layers, es;
    uint16_t s1, s2;
    uint_fast8_t l;

    if (rd == NULL || m == NULL || layers == NULL || buf == NULL) return MLP_ERR_ARG;
    r.rd = rd; r.ctx = ctx; r.s1 = 0; r.s2 = 0; r.err = 0;

    mlpi_get(&r, hdr, BLOB_HEADER);
    if (r.err) return MLP_ERR_IO;
    if (memcmp(hdr, "MLP1", 4) != 0 || hdr[4] != BLOB_VERSION || hdr[5] > MLP_Q15 || hdr[6] == 0u)
        return MLP_ERR_FORMAT;
    dtype = hdr[5];
    n_layers = hdr[6];
    if (!mlpi_dtype_enabled(dtype)) return MLP_ERR_DTYPE;
    if (n_layers > max_layers) return MLP_ERR_ARG;
    es = mlpi_wsize(dtype);

    for (l = 0; l < n_layers; ++l) {
        mlp_layer_t *L = &layers[l];
        L->n_in = (uint16_t)mlpi_get_le(&r, 2);
        L->n_out = (uint16_t)mlpi_get_le(&r, 2);
        L->act = (uint8_t)mlpi_get_le(&r, 1);
        L->w_frac = (int8_t)(uint8_t)mlpi_get_le(&r, 1);
        L->out_frac = (int8_t)(uint8_t)mlpi_get_le(&r, 1);
        if (r.err) return MLP_ERR_IO;
        if (L->n_in == 0u || L->n_out == 0u) return MLP_ERR_FORMAT;
        need += mlpi_layer_bytes(L->n_in, L->n_out, dtype, NULL);
    }
    if (need > buf_bytes) return MLP_ERR_ARG;

    for (l = 0; l < n_layers && !r.err; ++l) {
        mlp_layer_t *L = &layers[l];
        const uint32_t nw = (uint32_t)L->n_in * L->n_out;
        size_t b_off;
        uint32_t i;

        (void)mlpi_layer_bytes(L->n_in, L->n_out, dtype, &b_off);
        if (dtype == MLP_Q7) {
            int8_t *w = (int8_t *)(void *)p;
            for (i = 0; i < nw; ++i) w[i] = (int8_t)(uint8_t)mlpi_get_le(&r, es);
        } else if (dtype == MLP_Q15) {
            int16_t *w = (int16_t *)(void *)p;
            for (i = 0; i < nw; ++i) w[i] = (int16_t)(uint16_t)mlpi_get_le(&r, es);
        } else {
            float *w = (float *)(void *)p;
            for (i = 0; i < nw; ++i) {
                const uint32_t u = mlpi_get_le(&r, es);
                memcpy(&w[i], &u, sizeof u);
            }
        }
        {
            uint8_t *b = p + b_off;
            for (i = 0; i < L->n_out; ++i) {
                const uint32_t u = mlpi_get_le(&r, 4u);
                memcpy(b + 4u * i, &u, sizeof u);       /* float or int32: same bytes */
            }
        }
        L->w = p;
        L->b = p + b_off;
        p += mlpi_layer_bytes(L->n_in, L->n_out, dtype, NULL);
    }

    s1 = r.s1;
    s2 = r.s2;
    mlpi_get(&r, sum, 2);
    if (r.err) return MLP_ERR_IO;
    if (sum[0] != s1 || sum[1] != s2) return MLP_ERR_FORMAT;

    m->layers = layers;
    m->n_layers = n_layers;
    m->dtype = dtype;
    m->in_frac = (int8_t)hdr[7];
    m->flags = 0;
    return mlp_model_check(m);
}

/* ---- C header export ----------------------------------------------------- */

#if MLP_ENABLE_STDIO
#include <ctype.h>
#include <math.h>

static const char *mlpi_act_name(uint8_t act)
{
    static const char *const names[] = {
        "MLP_ACT_LINEAR", "MLP_ACT_RELU", "MLP_ACT_LEAKY_RELU", "MLP_ACT_SIGMOID",
        "MLP_ACT_TANH", "MLP_ACT_STEP", "MLP_ACT_SOFTMAX"
    };
    return names[act <= MLP_ACT_SOFTMAX ? act : 0];
}

/* One array: `type name[n] MLP_ROM = { ... };`, 8 values per line. */
static int mlpi_export_array(FILE *f, const char *type, const char *name, char kind, unsigned idx,
                             const void *data, uint32_t n, uint8_t esize, uint8_t is_float, uint8_t rom)
{
    uint32_t i;

    fprintf(f, "static const %s %s_%c%u[%lu] MLP_ROM = {", type, name, kind, idx, (unsigned long)n);
    for (i = 0; i < n; ++i) {
        const uint32_t u = mlpi_elem_bits(data, i, esize, is_float, rom);

        fputs(i % 8u == 0u ? "\n    " : " ", f);
        if (is_float) {
            float v;
            memcpy(&v, &u, sizeof v);
            if (!isfinite(v)) return -1;
            fprintf(f, "%.8ef,", (double)v);            /* 9 significant digits: exact round trip */
        } else if (esize == 1u) {
            fprintf(f, "%d,", (int)(int8_t)(uint8_t)u);
        } else if (esize == 2u) {
            fprintf(f, "%d,", (int)(int16_t)(uint16_t)u);
        } else {
            fprintf(f, "%ld,", (long)(int32_t)u);
        }
    }
    fputs("\n};\n", f);
    return 0;
}

mlp_status_t mlp_export_c(FILE *f, const mlp_model_t *m, const char *name)
{
    static const char *const wtype[] = { "float", "int8_t", "int16_t" };
    static const char *const dname[] = { "MLP_F32", "MLP_Q7", "MLP_Q15" };
    const uint8_t rom = (uint8_t)(m != NULL ? m->flags & MLP_MODEL_ROM : 0u);
    const char *c;
    uint_fast8_t l;
    mlp_status_t st;

    if (f == NULL || m == NULL || name == NULL || name[0] == '\0') return MLP_ERR_ARG;
    st = mlp_model_check(m);
    if (st != MLP_OK) return st;

    fprintf(f, "/* Generated by mlp_export_c(): %s model, %u layers (%u",
            dname[m->dtype], (unsigned)m->n_layers, (unsigned)m->layers[0].n_in);
    for (l = 0; l < m->n_layers; ++l) fprintf(f, "-%u", (unsigned)m->layers[l].n_out);
    fprintf(f, "), %lu bytes of weights. */\n#ifndef ", (unsigned long)mlp_buf_bytes(m, m->dtype));
    for (c = name; *c != '\0'; ++c) fputc(toupper((unsigned char)*c), f);
    fputs("_H\n#define ", f);
    for (c = name; *c != '\0'; ++c) fputc(toupper((unsigned char)*c), f);
    fputs("_H\n\n#include \"mlp.h\"\n\n", f);

    for (l = 0; l < m->n_layers; ++l) {
        const mlp_layer_t *L = &m->layers[l];
        const uint8_t is_float = m->dtype == MLP_F32;

        if (mlpi_export_array(f, wtype[m->dtype], name, 'w', (unsigned)l, L->w,
                              (uint32_t)L->n_in * L->n_out, mlpi_wsize(m->dtype), is_float, rom) != 0
            || mlpi_export_array(f, is_float ? "float" : "int32_t", name, 'b', (unsigned)l, L->b,
                                 L->n_out, 4u, is_float, rom) != 0)
            return MLP_ERR_RANGE;
    }

    fprintf(f, "\nstatic const mlp_layer_t %s_layers[%u] = {\n", name, (unsigned)m->n_layers);
    for (l = 0; l < m->n_layers; ++l) {
        const mlp_layer_t *L = &m->layers[l];
        fprintf(f, "    { %u, %u, %s, %d, %d, %s_w%u, %s_b%u },\n",
                (unsigned)L->n_in, (unsigned)L->n_out, mlpi_act_name(L->act),
                (int)L->w_frac, (int)L->out_frac, name, (unsigned)l, name, (unsigned)l);
    }
    fprintf(f, "};\n\nstatic const mlp_model_t %s = { %s_layers, %u, %s, %d, MLP_MODEL_ROM };\n\n#endif\n",
            name, name, (unsigned)m->n_layers, dname[m->dtype], (int)m->in_frac);
    return ferror(f) ? MLP_ERR_IO : MLP_OK;
}
#endif /* MLP_ENABLE_STDIO */
