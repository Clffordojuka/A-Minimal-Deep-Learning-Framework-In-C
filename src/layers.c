#include <stdio.h>
#include "../include/tinyml.h"

/*
------------------------------------
Create Dense Layer
------------------------------------
*/

DenseLayer dense_create(int input_size, int output_size)
{
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

Tensor dense_forward(DenseLayer *layer, Tensor *input)
{
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
                      double lr)
{
    /* ReLU gradient */

    Tensor grad_z =
        tensor_relu_backward(grad_output, &layer->z);

    /* Compute weight gradient */

    Tensor input_T =
        tensor_transpose(&layer->input);

    Tensor grad_w =
        tensor_matmul(&input_T, &grad_z);

    /* Update weights */

    int weight_size =
        layer->weights.rows * layer->weights.cols;

    for(int i = 0; i < weight_size; i++)
    {
        layer->weights.data[i] -=
            lr * grad_w.data[i];
    }

    /* Update bias */

    for(int i = 0; i < layer->bias.cols; i++)
    {
        layer->bias.data[i] -=
            lr * grad_z.data[i];
    }

    /* Compute gradient for previous layer */

    Tensor weights_T =
        tensor_transpose(&layer->weights);

    Tensor grad_input =
        tensor_matmul(&grad_z, &weights_T);

    /* Free temporary tensors */

    tensor_free(&grad_w);
    tensor_free(&input_T);
    tensor_free(&weights_T);

    return grad_input;
}

/*
------------------------------------
Free Layer
------------------------------------
*/

void dense_free(DenseLayer *layer)
{
    tensor_free(&layer->weights);
    tensor_free(&layer->bias);
}