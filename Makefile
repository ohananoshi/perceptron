# Host build, tests, benchmark, examples and optional cross-compiled size reports.
#
#   make test        unit tests under ASan/UBSan, normal and MLP_FAST_MATH builds,
#                    plus the second-stage exported-header test
#   make bench       time per inference (F32 / Q7 / Q15)
#   make examples    host examples (XOR, sine)
#   make cxx-check   public headers and an exported model compile as C++
#   make size-arm    flash/RAM of an inference-only build for Cortex-M0 and M4F
#   make size-avr    same for ATmega328P

CC      ?= cc
CXX     ?= c++
BUILD   := build

WARN    := -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -Wshadow \
           -Wstrict-prototypes -Wmissing-prototypes -Wcast-align -Werror
CFLAGS  := -std=c99 $(WARN) -Isrc
OPT     ?= -O2 -g
SAN     := -fsanitize=address,undefined -fno-sanitize-recover=all
LDLIBS  := -lm

LIB_SRC  := $(wildcard src/*.c)
LIB_HDR  := $(wildcard src/*.h)
TEST_SRC := $(filter-out tests/test_export.c,$(wildcard tests/*.c))
TEST_HDR := $(wildcard tests/*.h)

# Inference-only configuration used for the size reports.
SIZE_DEFS := -DMLP_ENABLE_TRAIN=0 -DMLP_ENABLE_QUANT=0
SIZE_SRC  := src/mlp.c src/mlp_f32.c src/mlp_fixed.c

.PHONY: all lib test bench examples cxx-check size-arm size-avr clean

all: lib test

lib: $(BUILD)/libmlp.a

$(BUILD)/lib/%.o: src/%.c $(LIB_HDR) | $(BUILD)/lib
	$(CC) $(CFLAGS) -Os -ffunction-sections -fdata-sections -c $< -o $@

$(BUILD)/libmlp.a: $(LIB_SRC:src/%.c=$(BUILD)/lib/%.o)
	$(AR) rcs $@ $^

$(BUILD)/test_all: $(LIB_SRC) $(LIB_HDR) $(TEST_SRC) $(TEST_HDR) | $(BUILD)
	$(CC) $(CFLAGS) $(OPT) $(SAN) -DMLP_ENABLE_STDIO=1 $(LIB_SRC) $(TEST_SRC) $(LDLIBS) -o $@

$(BUILD)/test_all_fast: $(LIB_SRC) $(LIB_HDR) $(TEST_SRC) $(TEST_HDR) | $(BUILD)
	$(CC) $(CFLAGS) $(OPT) $(SAN) -DMLP_ENABLE_STDIO=1 -DMLP_FAST_MATH=1 $(LIB_SRC) $(TEST_SRC) $(LDLIBS) -o $@

# MLP_HARVARD=1 on the host compiles the AVR PROGMEM read paths with plain
# reads, so their logic is tested without an AVR toolchain.
$(BUILD)/test_all_harvard: $(LIB_SRC) $(LIB_HDR) $(TEST_SRC) $(TEST_HDR) | $(BUILD)
	$(CC) $(CFLAGS) $(OPT) $(SAN) -DMLP_ENABLE_STDIO=1 -DMLP_HARVARD=1 $(LIB_SRC) $(TEST_SRC) $(LDLIBS) -o $@

# The first run writes gen_model.h; the second stage compiles it and checks
# that the exported models (flagged MLP_MODEL_ROM) reproduce the in-memory
# outputs bit for bit, through the RAM and the Harvard read paths.
test: $(BUILD)/test_all $(BUILD)/test_all_fast $(BUILD)/test_all_harvard
	$(BUILD)/test_all $(BUILD)/gen_model.h
	$(BUILD)/test_all_fast
	$(BUILD)/test_all_harvard
	$(CC) $(CFLAGS) $(OPT) $(SAN) -I$(BUILD) $(LIB_SRC) tests/test_export.c $(LDLIBS) -o $(BUILD)/test_export
	$(BUILD)/test_export
	$(CC) $(CFLAGS) $(OPT) $(SAN) -DMLP_HARVARD=1 -I$(BUILD) $(LIB_SRC) tests/test_export.c $(LDLIBS) -o $(BUILD)/test_export_harvard
	$(BUILD)/test_export_harvard

$(BUILD)/bench: bench/bench_host.c $(LIB_SRC) $(LIB_HDR) | $(BUILD)
	$(CC) $(CFLAGS) -O2 -D_POSIX_C_SOURCE=199309L $(LIB_SRC) bench/bench_host.c $(LDLIBS) -o $@

bench: $(BUILD)/bench
	$(BUILD)/bench

EXAMPLE_BUILD = $(CC) $(CFLAGS) -O2 -DMLP_ENABLE_STDIO=1 $(LIB_SRC) $< $(LDLIBS) -o $@

$(BUILD)/host_xor: examples/host_xor/host_xor.c $(LIB_SRC) $(LIB_HDR) | $(BUILD)
	$(EXAMPLE_BUILD)

$(BUILD)/host_sine: examples/host_sine/host_sine.c $(LIB_SRC) $(LIB_HDR) | $(BUILD)
	$(EXAMPLE_BUILD)

examples: $(BUILD)/host_xor $(BUILD)/host_sine

cxx-check:
	$(CXX) -std=c++11 -Wall -Wextra -Werror -fsyntax-only -Isrc -x c++ tests/cxx_check.cpp

size-arm: | $(BUILD)
	@command -v arm-none-eabi-gcc >/dev/null || { echo "arm-none-eabi-gcc not found (apt install gcc-arm-none-eabi)"; exit 1; }
	@for cpu in "cortex-m0 -mfloat-abi=soft" "cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16"; do \
	    set -- $$cpu; echo "== $$1 =="; \
	    for f in $(SIZE_SRC); do \
	        arm-none-eabi-gcc -std=c99 -Isrc -Os -mthumb -mcpu=$$cpu -ffunction-sections -fdata-sections \
	            $(SIZE_DEFS) -c $$f -o $(BUILD)/$$(basename $$f .c).arm.o || exit 1; \
	    done; \
	    arm-none-eabi-size -t $(BUILD)/*.arm.o; rm -f $(BUILD)/*.arm.o; \
	done

size-avr: | $(BUILD)
	@command -v avr-gcc >/dev/null || { echo "avr-gcc not found (apt install gcc-avr avr-libc)"; exit 1; }
	@for f in $(SIZE_SRC); do \
	    avr-gcc -std=c99 -Isrc -Os -mmcu=atmega328p -ffunction-sections -fdata-sections \
	        $(SIZE_DEFS) -c $$f -o $(BUILD)/$$(basename $$f .c).avr.o || exit 1; \
	done; \
	avr-size -t $(BUILD)/*.avr.o; rm -f $(BUILD)/*.avr.o

$(BUILD) $(BUILD)/lib:
	mkdir -p $@

clean:
	rm -rf $(BUILD)
