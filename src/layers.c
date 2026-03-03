#include <stdio.h>
#include "../include/tinyml.h"

/*
------------------------------------
Create Dense Layer
------------------------------------
*/

DenseLayer dense_create(int input_size, int output_size) {

    DenseLayer layer;

    layer.weights = tensor_create(input_size, output_size);
    layer.bias = tensor_create(1, output_size);

    tensor_random(&layer.weights);
    tensor_random(&layer.bias);

    return layer;
}

/*
------------------------------------
Forward Pass
------------------------------------
*/

Tensor dense_forward(DenseLayer *layer, Tensor *input) {

    layer->input = *input;

    Tensor z = tensor_matmul(input, &layer->weights);
    Tensor out = tensor_add_bias(&z, &layer->bias);

    layer->z = out;

    tensor_relu(&out);

    layer->output = out;

    return out;
}

/*
------------------------------------
Backward Pass
------------------------------------
*/

Tensor dense_backward(DenseLayer *layer,
                      Tensor *grad_output,
                      double lr) {

    Tensor grad_z =
        tensor_relu_backward(grad_output, &layer->z);

    Tensor grad_w =
        tensor_matmul(&layer->input, &grad_z);

    for(int i=0;i<layer->weights.rows * layer->weights.cols;i++)
        layer->weights.data[i] -=
            lr * grad_w.data[i];

    for(int i=0;i<layer->bias.cols;i++)
        layer->bias.data[i] -=
            lr * grad_z.data[i];

    return grad_z;
}

/*
------------------------------------
Free Layer
------------------------------------
*/

void dense_free(DenseLayer *layer) {

    tensor_free(&layer->weights);
    tensor_free(&layer->bias);
}