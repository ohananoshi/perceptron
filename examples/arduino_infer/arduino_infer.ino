/* arduino_infer.ino - run the XOR network trained and quantized on the PC by
   examples/host_xor. Its weights are int8 in flash (PROGMEM on AVR), so the
   only RAM the network needs is the 8-byte workspace below.

   Regenerate the model with:  make examples && build/host_xor examples/arduino_infer/xor_model.h
*/
#include <mlp.h>
#include "xor_model.h"

static int16_t ws[MLP_FWD_WS_ELEMS(2, 4)];

void setup()
{
    static const float inputs[4][2] = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };

    Serial.begin(115200);
    while (!Serial) {}

    for (int i = 0; i < 4; ++i) {
        int16_t xq[2], yq[1];
        float y;

        mlp_input_from_f32(&xor_model, inputs[i], xq);
        const unsigned long t0 = micros();
        mlp_forward(&xor_model, xq, yq, ws);
        const unsigned long dt = micros() - t0;
        mlp_output_to_f32(&xor_model, yq, &y);

        Serial.print((int)inputs[i][0]);
        Serial.print(" xor ");
        Serial.print((int)inputs[i][1]);
        Serial.print(" = ");
        Serial.print(y, 4);
        Serial.print("  (");
        Serial.print(dt);
        Serial.println(" us)");
    }
}

void loop()
{
}
