/* Header: perceptron.h

    Tools for MLP Neural Network development.

    Author: Guilherme Arruda

    GitHub: https://github.com/ohananoshi/perceptron

    Created on: 05 Jan 2023

    Last updated: 05 Jan 2023
*/


//=================================== HEADERS ==========================================
#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <math.h>

//==================================== MACROS ===========================================

//#define DEBUG_ALL
//#define DEBUG_NEURON_START
//#define DEBUG_NEURON_PROCESS
//#define DEBUG_LAYER_START
//#define DEBUG_LAYER_PROCESS
//#define DEBUG_NETWORK_START
//#define DEBUG_NETWORK_PROCESS
//#define DEBUG_NEURON_TRAINING
//#define DEBUG_LAYER_TRAINING

#define USE_UTILS

//==================================== MACRO FUNCTIONS ==================================

//=============================== DATA STRUCTURES =======================================

typedef struct{
    uint8_t weight_counter;
    double* weights;
    double (*activation_function)(double);
}p_neuron;

typedef struct{
    uint8_t neuron_counter;
    p_neuron* neuron; 
}p_layer;

typedef struct{
    uint8_t layer_counter;
    p_layer* layer;
}p_network;

//============================== NEURON ACTIVATION FUNCTIONS =============================

double step_1(double x){
    return x <= 0.0 ? 0.0:1.0;
}

double step_2(double x){
    return x <= 0 ? -1.0:1.0;
}

double sigmoid_1(double x){
    return 1/(1 + exp(-x));
}

//DERIVATIVES ---------------------------------------------------------------------------

double sigmoid_1d(double x){
    return sigmoid_1(x)*(1-sigmoid_1(x));
}

//====================================== FUNCTIONS =======================================

//UTIL TEST FUNCTIONS --------------------------------------------------------------------

#ifdef USE_UTILS

uint8_t* array_int_gen(uint8_t len, ...){
    uint8_t* arr = (uint8_t*)calloc(len, sizeof(uint8_t));

    va_list numbers;
    va_start(numbers, len);

    int aux;

    for(uint8_t i = 0; i < len; i++){
        aux = va_arg(numbers, int);
        if(aux > 255) fprintf(stderr, "Passed value is incompatible");

        arr[i] = (uint8_t)aux;
    }

    va_end(numbers);

    return arr;
}

double* array_gen_1(uint8_t len, ...){
    double* arr = (double*)calloc(len, sizeof(double));

    va_list numbers;
    va_start(numbers, len);

    for(uint8_t i = 0; i < len; i++){
        arr[i] = va_arg(numbers, double);
    }

    va_end(numbers);

    return arr;
}

double** array_gen_2(uint8_t rows, uint8_t colunms, ...){
    double** arr = (double**)calloc(rows, sizeof(double*));

    va_list number;
    va_start(number, colunms);

    for(uint8_t i = 0; i < rows; i++){
        arr[i] = (double*)calloc(colunms, sizeof(double));

        for(uint8_t j = 0; j < colunms; j++){
            arr[i][j] = va_arg(number, double);
        }
    }
    va_end(number);

    return arr;
}

double*** array_gen_3(uint8_t slices, uint8_t* rows, uint8_t* colunms, ...){
    
    double*** output = (double***)calloc(slices, sizeof(double**));

    va_list numbers;
    va_start(numbers, colunms);

     printf("--------------------\n");
      printf("--------------------\n");
    
    for(uint8_t i = 0; i < slices; i++){
        output[i] = (double**)calloc(rows[i], sizeof(double*));

        for(uint8_t j = 0; j < rows[i]; j++){
            output[i][j] = (double*)calloc(colunms[i], sizeof(double));

            for(uint8_t k = 0; k < colunms[i]; k++){
                output[i][j][k] = va_arg(numbers, double);
                printf("%f ", output[i][j][k]);
            }
            printf("\n");
        }
        printf("--------------------\n");
    }

    va_end(numbers);

    return output;
}

void array_print(double* arr, uint8_t len){
    for(uint8_t i = 0; i < len; i++){
        printf("%f ", arr[i]);
    }
    printf("\n");
}

double dot_product(double* a, double* b, uint16_t size){
    double result;

    for(uint16_t i = 0; i < size; i++){
        result += a[i]*b[i];
    }

    return result;
}

#endif

//CORE FUNCTIONS -------------------------------------------------------------------------

double* scalar_product(double a, double* array, uint8_t len){
    double* arr = (double*)calloc(len, sizeof(double));

    for(uint8_t i = 0; i < len; i++){
        arr[i] = a*array[i];
    }

    return arr;
}

double* array_sum(double* a, double* b, uint8_t len){
    double* sum = (double*)calloc(len, sizeof(double));

    for(uint8_t i = 0; i < len; i++){
        sum[i] = a[i] + b[i];
    }

    return sum;
}

p_neuron neuron_start(double* weights, uint8_t weight_count, double (*function)(double)){
    p_neuron output;

    output.activation_function = function;
    output.weight_counter = weight_count;
    output.weights = (double*)calloc(3, sizeof(double));

    for(uint16_t i = 0; i < weight_count; i++){
        output.weights[i] = weights[i];
    }

    #ifdef DEBUG_NEURON_START
        printf("NEURON START DEBUG\n");
        printf("WEIGHT COUNTER: %d\n", output.weight_counter);
        printf("WEIGHT ARRAY: ");array_print(output.weights, output.weight_counter);
    #endif

    return output;
}

double neuron_process(p_neuron neuron, double* data_input){
    
    //if(data_len != neuron.weight_counter) fprintf(stderr, "Data lenght and neuron weight number are different.");
    
    double output = 0;

    #ifdef DEBUG_NEURON_PROCESS
        printf("NEURON PROCESS DEBUG\n");
        printf("input array: ");array_print(data_input, neuron.weight_counter);
    #endif

    for(uint8_t i = 0; i < neuron.weight_counter; i++){

        output += data_input[i]*neuron.weights[i];

        #ifdef DEBUG_NEURON_PROCESS
           printf("input: %f input weight: %f sum: %f\n",data_input[i],neuron.weights[i], output);
        #endif 
    }

    #ifdef DEBUG_NEURON_PROCESS
        printf("->%f \n", neuron.activation_function(output));
    #endif

    return neuron.activation_function(output);
}

p_layer layer_start(uint8_t neuron_count, double** start_weights, uint8_t weight_count, double (*function)(double)){
    p_layer output_layer;

    output_layer.neuron_counter = neuron_count;
    output_layer.neuron = (p_neuron*)calloc(neuron_count, sizeof(p_neuron));

    #ifdef DEBUG_LAYER_START
        printf("DEBUG LAYER START\n");
        printf("NEURON COUNT: %d\n", output_layer.neuron_counter);
    #endif

    for(uint8_t i = 0; i < neuron_count; i++){

        #ifdef DEBUG_LAYER_START
            printf("NEURON %d\n", i);
        #endif

        output_layer.neuron[i] = neuron_start(start_weights[i], weight_count, function);
    }

    return output_layer;
}

double* layer_process(p_layer layer, double* input_data){
    double* layer_output = (double*)calloc(layer.neuron_counter, sizeof(double));

    #ifdef DEBUG_LAYER_PROCESS
            printf("DEBUG_LAYER_PROCESS\n");
    #endif

    for(uint8_t i = 0; i < layer.neuron_counter; i++){

        #ifdef DEBUG_LAYER_PROCESS
            printf("NEURON %d\n", i);
        #endif

        layer_output[i] = neuron_process(layer.neuron[i], input_data);
    }

    return layer_output;
}

p_network network_start(uint8_t layer_count, uint8_t* layer_neuron_count, uint8_t* neuron_weight_counter,double*** start_weights, double(*function[])(double)){
    p_network output_network;

    output_network.layer_counter = layer_count;
    output_network.layer = (p_layer*)calloc(layer_count, sizeof(p_layer));

    #ifdef DEBUG_NETWORK_START
        printf("DEBUG NETWORK START\nLAYER COUNT: %d\n", layer_count);
    #endif

    for(uint8_t i = 0; i < layer_count; i++){

        #ifdef DEBUG_NETWORK_START
            printf("LAYER: %d\n", i);
        #endif

        output_network.layer[i] = layer_start(layer_neuron_count[i], start_weights[i], neuron_weight_counter[i], function[i]);
    }

    return output_network;
}

double* iterative_compose(p_network network, double* input){

    double* result = input;

    for(uint8_t i = 0; i < network.layer_counter; i++){
        result = layer_process(network.layer[i], result);

        #ifdef DEBUG_NETWORK_PROCESS
            printf("LAYER: %d\nRESULT: ", i);
            array_print(result, network.layer[i].neuron_counter);
            printf("@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        #endif
    } 

    return result;
}

double* network_process(p_network network, double* input_data){
    double* network_output = (double*)calloc(network.layer[network.layer_counter - 1].neuron_counter, sizeof(double));

    #ifdef DEBUG_NETWORK_PROCESS
        printf("DEBUG NETWORK PROCESS\n@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
    #endif

    network_output = iterative_compose(network, input_data);

    return network_output;
}

//SUPERVISIONED TRAINING FUNCTIONS -------------------------------------------------------

double* neuron_training(p_neuron neuron, double learning_rate, double** input, uint16_t sample_count, double* output, uint16_t max_training_cycle){
    double* error = (double*)calloc(sample_count, sizeof(double));
    uint16_t err_flag, cycle_counter;

    do{
        err_flag = 0;

        #ifdef DEBUG_NEURON_TRAINING
            printf("----- DEBUG_NEURON_TRAINING -----\n");
            printf("Round: %d, err_flag start: %d\n Initial weights: ", cycle_counter, err_flag);
            array_print(neuron.weights,3);
        #endif

        for(uint8_t i = 0; i < sample_count; i++){
            error[i] = output[i] - neuron_process(neuron, input[i]);

            #ifdef DEBUG_NEURON_TRAINING
                printf("INPUT[%d]: ",i);array_print(input[i], neuron.weight_counter);
                printf("NEURON OUT: %f  EXPECTED OUT: %f  ERROR[%d]: %f\n", neuron_process(neuron, input[i]), output[i], i, error[i]);
            #endif

            if(round(error[i]) != round(0.0)){
                err_flag++;

                #ifdef DEBUG_NEURON_TRAINING
                    printf("err_flag: %d \nold_weights: ",err_flag);array_print(neuron.weights,neuron.weight_counter);
                #endif

                neuron.weights = array_sum(neuron.weights, scalar_product(learning_rate*error[i],input[i],neuron.weight_counter), neuron.weight_counter);

                #ifdef DEBUG_NEURON_TRAINING
                    printf("new_weights: ");array_print(neuron.weights,3);
                #endif
            }  
        }
        cycle_counter++;
    }while((err_flag != 0) || (cycle_counter > max_training_cycle));

    if(cycle_counter > max_training_cycle) printf("\n\nMAXIMUM TRAINING EPOCHS REACHED\n\n");

    cycle_counter = 0;
    free(error);

    return neuron.weights;
}

p_neuron* layer_training(p_layer layer, double learning_rate, double** input, uint16_t sample_count, double** output, uint16_t max_training_cycles){
    
    #ifdef DEBUG_LAYER_TRAINING
            printf("=========== DEBUG_LAYER_TRAINING =============\n");
    #endif

    for(uint16_t i = 0; i < layer.neuron_counter; i++){

        #ifdef DEBUG_LAYER_TRAINING
            printf("------------- NEURON: %d ---------------\n\n", i);
        #endif

        layer.neuron[i].weights = neuron_training(layer.neuron[i], learning_rate, input, sample_count, output[i], max_training_cycles); 
    }

    return layer.neuron;
}


/*NOT YET
p_layer* network_training(p_network network, double learning_rate, double** input, uint16_t sample_count, double** output, uint16_t max_training_cycles){
    double** out = (double**)calloc(sample_count, sizeof(double*));
    double** error = (double**)calloc(sample_count, sizeof(double*));
    
    for(uint16_t i = 0; i < 0; i++){
        out[i] = network_process(network, input[i]);
        error[i] = array_sum(output[i],scalar_product(-1.0,out[i],network.layer[i].neuron_counter),network.layer[i].neuron_counter);
    }


}
*/