/* mlp_io.h - model persistence.

   mlp_save() / mlp_load() stream a model through your own read/write
   callbacks, so it can live in EEPROM, external flash, an SD card or a file.
   Blob format (little-endian, independent of the target's byte order):

     "MLP1"  u8 version=1  u8 dtype  u8 n_layers  i8 in_frac
     per layer: u16 n_in  u16 n_out  u8 act  i8 w_frac  i8 out_frac
     per layer: weights (n_in*n_out x f32 / i8 / i16), biases (n_out x f32 / i32)
     u8 u8  Fletcher-16 checksum of everything before it

   mlp_export_c() (MLP_ENABLE_STDIO only) writes a C header with the model as
   MLP_ROM arrays, ready to compile into firmware.
*/
#ifndef MLP_IO_H
#define MLP_IO_H

#include "mlp.h"
#if MLP_ENABLE_STDIO
#include <stdio.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Return 0 on success. */
typedef int (*mlp_write_fn)(void *ctx, const uint8_t *data, uint16_t n);
typedef int (*mlp_read_fn)(void *ctx, uint8_t *data, uint16_t n);

/* Exact size of the blob mlp_save() writes. */
size_t mlp_blob_bytes(const mlp_model_t *m);

mlp_status_t mlp_save(const mlp_model_t *m, mlp_write_fn wr, void *ctx);

/* Read a blob into `layers` (max_layers entries) and `buf` (4-byte aligned,
   at least mlp_buf_bytes() of the saved model). A loaded float model uses the
   mlp_model_init_f32() layout, so it can keep training with params = buf.
   Returns MLP_ERR_FORMAT for a bad header or checksum, MLP_ERR_IO when the
   callback fails, MLP_ERR_ARG when layers or buf are too small. */
mlp_status_t mlp_load(mlp_read_fn rd, void *ctx, mlp_model_t *m,
                      mlp_layer_t *layers, uint8_t max_layers, void *buf, size_t buf_bytes);

#if MLP_ENABLE_STDIO
/* Write a header defining `name` (an mlp_model_t) and its arrays.
   `name` must be a valid C and C++ identifier. */
mlp_status_t mlp_export_c(FILE *f, const mlp_model_t *m, const char *name);
#endif

#ifdef __cplusplus
}
#endif

#endif /* MLP_IO_H */
