# perceptron: MLP neural networks for microcontrollers

A C99 framework for running and training multilayer perceptrons on small
hardware: Arduino (AVR), PIC, STM32 and ESP32.

- **No dynamic allocation.** You provide every buffer. Size macros and
  `*_bytes()` functions say exactly how much each needs.
- **Three number types.** float32, Q15 (int16 weights) and Q7 (int8 weights).
  Fixed-point inference uses only integer multiply-accumulate: no FPU, no
  division, no floats.
- **Weights stay in flash.** Exported models are `const` arrays marked
  `MLP_ROM`, which means `PROGMEM` on AVR.
- **Train on the PC or on the device.** Online backprop with SGD and optional
  momentum. Data is streamed through a callback, so samples never have to sit
  in RAM.
- **Train, then deploy.** Train in float, quantize with calibration data, and
  export a C header to compile into firmware.
- Everything optional can be compiled out: training, quantization, unused
  number types, stdio.

## Quick start

```c
#include "mlp.h"
#include "mlp_train.h"

static const uint16_t sizes[] = { 2, 4, 1 };                  /* input, hidden, output */
static const uint8_t  acts[]  = { MLP_ACT_TANH, MLP_ACT_SIGMOID };
static float params[MLP_LAYER_PARAMS(2, 4) + MLP_LAYER_PARAMS(4, 1)];
static float ws[MLP_FWD_WS_ELEMS(2, 4)];
static mlp_layer_t layers[2];
static mlp_model_t net;

mlp_model_init_f32(&net, layers, sizes, acts, 2, params);
mlp_init_params(&net, params, 42);                             /* random weights */
/* ... train with mlp_train_epoch(), or load trained weights ... */
float x[2] = { 1.0f, 0.0f }, y[1];
mlp_forward(&net, x, y, ws);
```

To go from the PC to an MCU, see [examples/host_xor](examples/host_xor/host_xor.c).
It trains XOR, quantizes it to Q7, and writes
[xor_model.h](examples/arduino_infer/xor_model.h).
[examples/arduino_infer](examples/arduino_infer/arduino_infer.ino) then runs
that header from flash:

```c
#include <mlp.h>
#include "xor_model.h"                 /* generated: int8 weights in PROGMEM */

static int16_t ws[MLP_FWD_WS_ELEMS(2, 4)];
int16_t xq[2], yq[1];
mlp_input_from_f32(&xor_model, x, xq);
mlp_forward(&xor_model, xq, yq, ws);
```

## Memory

Measured with `make bench` for an 8-16-4 network (212 parameters):

| | Old `perceptron.h` design (estimate, 32-bit ARM) | float32 | Q15 | Q7 |
|---|---|---|---|---|
| Weights + biases | ~1.7 KB of `double` in RAM | 848 B in flash | 464 B in flash | 272 B in flash |
| Other inference RAM | ~0.8 KB: per-neuron structs, malloc headers, per-call buffers | 64 B workspace | 32 B workspace | 32 B workspace |

The inference workspace is one buffer as wide as the widest hidden layer, or
two such buffers when there are two or more hidden layers. Layer 0 reads your
input directly, and the last layer writes straight into your output.

## Speed

- **float32** is the right choice on chips with a single-precision FPU:
  Cortex-M4F/M7 and ESP32 (Xtensa).
- **Q7/Q15** are for chips without one: AVR, PIC, Cortex-M0/M3 and ESP32-C3.
  There, every float operation is a software routine, and the integer path is
  expected to be several times faster.
- On a PC (hardware FPU with SIMD), float32 wins:
  - 8-16-4: 152 ns float32, 213 ns Q7.
  - 64-32-10: 346 ns float32, 857 ns Q7.
- Measure on your own board with the snippets under [Porting](#porting).

## API

| Header | What it does |
|---|---|
| [mlp.h](src/mlp.h) | Types, model setup (`mlp_model_init_f32`, `mlp_model_check`), inference (`mlp_forward`, `mlp_argmax`), fixed-point conversion (`mlp_input_from_f32`, `mlp_output_to_f32`), sizing (`mlp_param_count`, `mlp_buf_bytes`, `mlp_forward_ws_bytes`, `MLP_LAYER_PARAMS`, `MLP_LAYER_BYTES_Q7/Q15`, `MLP_FWD_WS_ELEMS`) |
| [mlp_train.h](src/mlp_train.h) | `mlp_init_params` (Xavier/He), `mlp_train_sample`, `mlp_train_epoch`, `mlp_loss`, `mlp_xorshift32` |
| [mlp_quant.h](src/mlp_quant.h) | `mlp_quantize`: float32 to Q7/Q15, with calibration |
| [mlp_io.h](src/mlp_io.h) | `mlp_save` / `mlp_load` (checksummed blob through read/write callbacks, for EEPROM, flash or SD), `mlp_export_c` (C header) |

All functions return an `mlp_status_t`. None of them allocate, print, or call
`exit()`.

**Activations:**
- `LINEAR`, `RELU`, `LEAKY_RELU`, `SIGMOID`, `TANH`, `STEP`.
- `SOFTMAX` is float only and only allowed on the output layer.

**Losses:**
- MSE (`0.5 * sum((y - t)^2)`) works with any output activation except softmax.
- Cross-entropy needs a sigmoid or softmax output.

**Training and quantization data** come through a callback:

```c
int get_sample(void *ctx, uint16_t idx, float *x, float *t);   /* return 0 on success */
```

A single-layer `STEP` network trained with MSE uses the classic perceptron
learning rule, so the old `neuron_training()` behaviour is still available.

## Fixed-point format

A value `q` with `f` fractional bits means `q * 2^-f`. The scale is chosen once
per layer, which works like a block floating-point format:

- **Input:** int16, with `mlp_model_t.in_frac` fractional bits.
- **Weights:** int8 (Q7) or int16 (Q15), with `w_frac` bits.
- **Biases:** int32 at the accumulator scale `w_frac + input frac`.
- **Accumulator:** `int32 acc = bias + Σ w·x`. It is then shifted with
  rounding and saturation to the layer's `out_frac`.
- **Sigmoid and tanh:** a 33-entry interpolated lookup table (66 bytes, in
  flash). Error is below 7.5e-4.

`mlp_quantize()` picks every format for you. It also reduces weight precision
until `|bias| + Σ|w|·32768 < 2^30` for every neuron, so no possible int16 input
can overflow the accumulator.

On the sine example (1-16-1), the MSE against sin(x) is:
- float32: 9.5e-5
- Q15: 5.7e-5
- Q7: 4.8e-4

## Configuration

Set these with `-D` flags, or put them in a header and pass
`-DMLP_CONFIG_FILE=\"my_config.h\"`. See [mlp_config.h](src/mlp_config.h).

| Option | Default | |
|---|---|---|
| `MLP_ENABLE_F32` / `_Q7` / `_Q15` | 1 | Number types available to `mlp_forward` |
| `MLP_ENABLE_TRAIN`, `MLP_ENABLE_QUANT` | 1 | Training and quantization |
| `MLP_ENABLE_STDIO` | 0 | `mlp_export_c()` |
| `MLP_FAST_MATH` | 0 | Bit-level `expf` (writes 2ⁿ straight into the float's exponent field through a union, plus a cubic). Relative error 1.03e-4 |
| `MLP_UNROLL` | 1 (0 on AVR) | 4x unrolled dot products |
| `MLP_LEAKY_ALPHA` | 0.01f | Leaky-relu slope |
| `MLP_DOT_F32`, `MLP_DOT_Q7`, `MLP_DOT_Q15` | built-in | Hooks for platform dot products |

Unused functions are dropped by `-ffunction-sections -Wl,--gc-sections`, which
Arduino, STM32CubeIDE and ESP-IDF use by default.

## Porting

- **AVR (Arduino Uno/Nano/Mega):**
  - `MLP_ROM` is `PROGMEM`, and weights are read with `pgm_read_*`.
  - Exported models carry `MLP_MODEL_ROM`, so they run from flash.
  - Models built in RAM (for training) run from RAM.
  - Only the low 64 KB of flash is reachable this way.
  - Use Q7: float math is done in software, and SRAM is 2 KB on the Uno.
- **PIC:**
  - XC8 already puts `const` data in program memory.
  - On XC16, define `MLP_ROM` as `__attribute__((space(auto_psv)))` if needed.
  - Select 32-bit IEEE `float`: there is a compile-time check for it.
- **STM32:**
  - M4F/M7: use float32.
  - M0/M0+/M3: use Q7/Q15.
  - With CMSIS-DSP, pass this file via `MLP_CONFIG_FILE`:

    ```c
    #include "arm_math.h"
    static inline float my_dot(const float *a, const float *b, uint_fast16_t n)
    { float r; arm_dot_prod_f32(a, b, (uint32_t)n, &r); return r; }
    #define MLP_DOT_F32(w, x, n) my_dot((w), (x), (n))
    ```

  - Count cycles with the DWT counter (M3 and up):

    ```c
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    uint32_t c0 = DWT->CYCCNT; mlp_forward(&model, x, y, ws); uint32_t cycles = DWT->CYCCNT - c0;
    ```

- **ESP32:**
  - Xtensa has a single-precision FPU, so float32 is a good default.
  - ESP32-C3 (RISC-V) has no FPU: use Q7/Q15.
  - `dsps_dotprod_f32` from esp-dsp can plug into `MLP_DOT_F32`.
  - Time with `esp_cpu_get_cycle_count()`.
- **Arduino, any board:** time with `micros()`, as in the example sketch.

Fixed-point code assumes right shifts of negative numbers are arithmetic. C99
leaves that to the compiler, but GCC, Clang, IAR, Keil and XC all do it, and
the tests check it.

## Build and test (host)

```bash
make test
```

`make test` runs:
- The unit tests under AddressSanitizer and UBSan with `-std=c99 -Wall -Wextra
  -Wpedantic -Wconversion -Wdouble-promotion -Werror`, in normal and
  `MLP_FAST_MATH` builds. They include a gradient check against finite
  differences for every activation/loss pair.
- A second stage that compiles an exported header and checks it reproduces
  the in-memory outputs bit for bit.

| Target | Does |
|---|---|
| `make examples` | Builds `build/host_xor` and `build/host_sine` |
| `make bench` | Time per inference and memory for each number type |
| `make cxx-check` | Headers and an exported model compile as C++ |
| `make size-arm`, `make size-avr` | Flash/RAM of an inference-only build. Needs `gcc-arm-none-eabi`, or `gcc-avr avr-libc` |
| `make lib` | `build/libmlp.a` |

## Migrating from the old `perceptron.h`

| Old | New |
|---|---|
| `p_neuron` / `p_layer` / `p_network` (one heap allocation per neuron) | `mlp_layer_t` / `mlp_model_t` over your own buffers |
| `neuron_start`, `layer_start`, `network_start` | `mlp_model_init_f32` + `mlp_init_params` |
| `neuron_process`, `layer_process`, `network_process` | `mlp_forward` |
| `neuron_training`, `layer_training` | `mlp_train_sample` / `mlp_train_epoch` (single-layer `MLP_ACT_STEP` gives the perceptron rule) |
| `step_1`, `sigmoid_1` | `MLP_ACT_STEP`, `MLP_ACT_SIGMOID` |
| `step_2` (−1/+1) | No direct equivalent: use `MLP_ACT_TANH`, or `STEP` and map 0 to −1 |
| `array_gen_*`, `array_print` | Removed: use plain C arrays |

## Layout

```
src/        library (Arduino library layout)
tests/      unit tests, C++ header check, exported-header test
examples/   host_xor, host_sine, arduino_infer
bench/      host benchmark
```
