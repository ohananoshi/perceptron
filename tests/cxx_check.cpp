// cxx_check.cpp - the public headers and an exported model must compile as
// C++, because Arduino sketches (.ino) are C++.
#include "mlp.h"
#include "mlp_io.h"
#include "mlp_quant.h"
#include "mlp_train.h"
#include "../examples/arduino_infer/xor_model.h"

int cxx_check(void);

int cxx_check(void)
{
    int16_t x[2] = { 0, 16384 }, y[1], ws[MLP_FWD_WS_ELEMS(2, 4)];
    return mlp_forward(&xor_model, x, y, ws) == MLP_OK ? y[0] : -1;
}
