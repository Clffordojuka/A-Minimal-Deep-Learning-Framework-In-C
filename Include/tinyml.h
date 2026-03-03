#ifndef TINYML_H
#define TINYML_H

#include <stdlib.h>

/*
------------------------------------
Tensor Structure
------------------------------------
*/

typedef struct {
    int rows;
    int cols;
    double *data;
} Tensor;

/*
------------------------------------
Tensor Memory
------------------------------------
*/

Tensor tensor_create(int rows, int cols);
void tensor_free(Tensor *t);

/*
------------------------------------
Basic Operations
------------------------------------
*/

Tensor tensor_matmul(Tensor *a, Tensor *b);
Tensor tensor_add(Tensor *a, Tensor *b);
Tensor tensor_add_bias(Tensor *a, Tensor *bias);

/*
------------------------------------
Activations
------------------------------------
*/

void tensor_relu(Tensor *t);
Tensor tensor_relu_backward(Tensor *grad, Tensor *z);

/*
------------------------------------
Utilities
------------------------------------
*/

void tensor_random(Tensor *t);
void tensor_print(Tensor *t);

/*
================================================
NEURAL NETWORK LAYERS
================================================
*/

/*
------------------------------------
Dense (Fully Connected) Layer
------------------------------------
*/

typedef struct {

    Tensor weights;
    Tensor bias;

    Tensor input;     // saved input for backprop
    Tensor z;         // pre-activation
    Tensor output;    // activation output

} DenseLayer;

/*
------------------------------------
Layer API
------------------------------------
*/

DenseLayer dense_create(int input_size, int output_size);

Tensor dense_forward(DenseLayer *layer, Tensor *input);

Tensor dense_backward(DenseLayer *layer,
                      Tensor *grad_output,
                      double learning_rate);

void dense_free(DenseLayer *layer);

#endif