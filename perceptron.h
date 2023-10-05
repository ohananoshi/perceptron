/* Header: perceptron.h

    Tools for MLP Neural Network development.

    Author: Guilherme Arruda

    GitHub: https://github.com/ohananoshi/perceptron

    Created on: 05 Jan 2023

    Last updated: 05 Jan 2023
*/


//=================================== HEADERS ==========================================

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>

//=============================== DATA STRUCTURES =======================================

typedef struct{
    uint8_t weight_counter;
    double bias_input;
    double* input_weights;
    double (*activation_function)(double);
}p_neuron;

typedef struct{
    uint8_t neuron_counter;
    p_neuron* neuron; 
}p_layer;

//====================================== FUNCTIONS =======================================

double dot_product(double* a, double* b, uint16_t size){
    double result;

    for(uint16_t i = 0; i < size; i++){
        result += a[i]*b[i];
    }

    return result;
}

p_neuron neuron_start(double* measurements, uint8_t element_count, double (*function)(double)){
    p_neuron output;
    output.activation_function = function;
    output.bias_input = measurements[0];
    output.weight_counter = (element_count - 1);

    for(uint16_t i = 2; i < element_count; i++){
        output.input_weights[i] = measurements[i];
    }

    return output;
}

p_layer layer_start(uint8_t neuron_count, p_neuron* neurons, double** start_values, uint8_t data_len, double (*function)(double)){
    p_layer output_layer;
    output_layer.neuron_counter = neuron_count;
    output_layer.neuron = neurons;

    for(uint8_t i = 0; i < neuron_count; i++){
        output_layer.neuron[i] = neuron_start(start_values[i], data_len, function);
    }

    return output_layer;
}

double neuron_process(p_neuron neuron, double* data_input, uint8_t data_len){
    double output;

    for(uint8_t i = 0; i < data_len; i++){
        output += data_input[i]*neuron.input_weights[i];
    }

    return neuron.activation_function(output);
}

double* layer_process(p_layer layer, double* input_data, uint8_t input_len){
    double* layer_output = (double*)calloc(layer.neuron_counter, sizeof(double));

    for(uint8_t i = 0; i < layer.neuron_counter; i++){
        layer_output[i] = neuron_process(layer.neuron[i], input_data, input_len);
    }

    return layer_output;
}
