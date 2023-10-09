### C framework for MLP (Multi Layer Perceptron) supervised training

### You can easy create and train individually neurons, layers or networks (in future).

#### Datatypes
- #### p_neuron
- #### p_layer
- #### p_network

#### Pre-defined acrivation functions
- #### step_1
    Returns 0 if x < 0 and 1 if x >= 0;
- #### step_2
    Returns -1 if x < 0 and 1 if x >= 0;
- #### sigmoid_1
    Applies: $$f\left(x\right)=\frac{1}{1+e^{-ax}}$$
- #### sigmoid_1d
    Is the derivative of sigmoid_1. Applies:
    $$f´\left(x\right)=f\left(x\right)\left(1-f\left(x\right)\right)$$

#### Core functions

- #### neuron_start
```C
p_neuron neuron_start(double* weights, uint8_t weight_count, double (*function)(double))

Initialize a neuron with initial weight values and an activation function.
```
- #### neuron_process
```C
double neuron_process(p_neuron neuron, double* data_input)

Process the input array according with neuron weights and activation function, and returns the result.
```
- #### layer_start
```C
p_layer layer_start(uint8_t neuron_count, double** start_weights, uint8_t weight_count, double (*function)(double))

Initialize a set of neurons that will belong to the same layer, each neuron will have the same activation funtion.
```
- #### layer_process
```C
double* layer_process(p_layer layer, double* input_data)

Process the input array and returns an array with respective neuron response.
```
- #### network_start
```C
p_network network_start(uint8_t layer_count, uint8_t* layer_neuron_count, uint8_t* neuron_weight_counter,double*** start_weights, double(*function[])(double))

Initialize a neural network with desirable layers, neurons per layer, and activation function per layer.
```
- #### network_process
```C
double* network_process(p_network network, double* input_data)

Process the input array and returns the neural network response.
```
#### Supervised Training Functions
- #### neuron_training
```C
double* neuron_training(p_neuron neuron, double learning_rate, double** input, uint16_t sample_count, double* output, uint16_t max_training_cycle)

It receives the training data samples and returns, after training is successful or the number of training cycles exceeds the maximum, an array with the adjusted weights.
```
- #### layer_training
```C
p_neuron* layer_training(p_layer layer, double learning_rate, double** input, uint16_t sample_count, double** output, uint16_t max_training_cycles)

It receives the training data samples and returns, after training is successful or the number of training cycles exceeds the maximum, an array with the adjusted neurons.
```
- #### network_training (coming_soon)