/* mlp_port.h - compiler and platform abstraction.

   MLP_ROM        storage qualifier for constant model data (PROGMEM on AVR)
   MLP_RD_*(p)    read one value from MLP_ROM data
   MLP_HARVARD    1 when ROM data is not reachable through normal pointers
   MLP_INLINE     static inline
   MLP_RESTRICT   C99 restrict (or the C++ extension)
   MLP_ASSERT(c)  debug check, compiled out unless MLP_DEBUG
   MLP_LOG(...)   debug trace, compiled out unless MLP_DEBUG and defined by you

   Every macro can be predefined (e.g. from MLP_CONFIG_FILE) to port the
   library to a new compiler.
*/
#ifndef MLP_PORT_H
#define MLP_PORT_H

#include "mlp_config.h"

#ifndef MLP_INLINE
#define MLP_INLINE static inline
#endif

#ifndef MLP_RESTRICT
#ifdef __cplusplus
#define MLP_RESTRICT __restrict
#else
#define MLP_RESTRICT restrict
#endif
#endif

/* ---- Constant data in program memory ---------------------------------------
   ARM, ESP32, RISC-V: const data already lives in flash and is read through
   ordinary pointers, so nothing is needed.
   PIC XC8: const data goes to program memory automatically.
   PIC XC16: define MLP_ROM as __attribute__((space(auto_psv))) if needed.
   AVR: const data is copied to SRAM at startup unless it is marked PROGMEM,
   and PROGMEM data must be read with pgm_read_*(). Only the low 64 KB of
   flash is reachable this way. */
#if defined(__AVR__) && !defined(MLP_ROM)
#include <avr/pgmspace.h>
#define MLP_HARVARD 1
#define MLP_ROM PROGMEM
#define MLP_RD_F32(p) pgm_read_float(p)
#define MLP_RD_I8(p)  ((int8_t)pgm_read_byte(p))
#define MLP_RD_I16(p) ((int16_t)pgm_read_word(p))
#define MLP_RD_I32(p) ((int32_t)pgm_read_dword(p))
#endif

#ifndef MLP_HARVARD
#define MLP_HARVARD 0
#endif
#ifndef MLP_ROM
#define MLP_ROM
#endif
#ifndef MLP_RD_F32
#define MLP_RD_F32(p) (*(p))
#define MLP_RD_I8(p)  (*(p))
#define MLP_RD_I16(p) (*(p))
#define MLP_RD_I32(p) (*(p))
#endif

/* ---- Debug ------------------------------------------------------------- */
#if MLP_DEBUG
#ifndef MLP_ASSERT
#include <assert.h>
#define MLP_ASSERT(c) assert(c)
#endif
#endif
#ifndef MLP_ASSERT
#define MLP_ASSERT(c) ((void)0)
#endif

/* Define MLP_LOG yourself (e.g. to printf or Serial.printf) to get traces. */
#if !MLP_DEBUG || !defined(MLP_LOG)
#undef MLP_LOG
#define MLP_LOG(...) ((void)0)
#endif

/* The blob format and the fast-math tricks assume IEEE-754 binary32 floats.
   (Old PIC XC8 versions default to 24-bit float: select 32-bit.) */
typedef char mlp_assert_float_is_32bit[(sizeof(float) == 4) ? 1 : -1];

#endif /* MLP_PORT_H */
