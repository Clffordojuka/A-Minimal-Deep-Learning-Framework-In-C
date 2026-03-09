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

    /* gradient buffers */
    layer.grad_weights = tensor_create(input_size, output_size);
    layer.grad_bias = tensor_create(1, output_size);

    dense_zero_grad(&layer);

    return layer;
}

/*
------------------------------------
Forward Pass (Linear only)
------------------------------------
*/

Tensor dense_forward(DenseLayer *layer, Tensor *input)
{
    layer->input = *input;

    Tensor z = tensor_matmul(input, &layer->weights);
    Tensor z_bias = tensor_add_bias(&z, &layer->bias);

    tensor_free(&z);

    layer->z = z_bias;
    layer->output = z_bias;

    return z_bias;
}

/*
------------------------------------
Backward Pass (Accumulate gradients)
------------------------------------
*/

Tensor dense_backward(DenseLayer *layer,
                      Tensor *grad_output)
{
    Tensor input_T = tensor_transpose(&layer->input);
    Tensor grad_w = tensor_matmul(&input_T, grad_output);

    /* accumulate weight gradients */
    int weight_size =
        layer->weights.rows * layer->weights.cols;

    for (int i = 0; i < weight_size; i++)
    {
        layer->grad_weights.data[i] += grad_w.data[i];
    }

    /* accumulate bias gradients */
    for (int i = 0; i < layer->bias.cols; i++)
    {
        layer->grad_bias.data[i] += grad_output->data[i];
    }

    /* propagate gradient to previous layer */
    Tensor weights_T = tensor_transpose(&layer->weights);
    Tensor grad_input = tensor_matmul(grad_output, &weights_T);

    tensor_free(&grad_w);
    tensor_free(&input_T);
    tensor_free(&weights_T);

    return grad_input;
}

/*
------------------------------------
Apply accumulated gradients
------------------------------------
*/

void dense_apply_gradients(DenseLayer *layer,
                           double learning_rate,
                           int batch_size)
{
    int weight_size =
        layer->weights.rows * layer->weights.cols;

    for (int i = 0; i < weight_size; i++)
    {
        layer->weights.data[i] -=
            learning_rate *
            (layer->grad_weights.data[i] / batch_size);
    }

    for (int i = 0; i < layer->bias.cols; i++)
    {
        layer->bias.data[i] -=
            learning_rate *
            (layer->grad_bias.data[i] / batch_size);
    }
}

/*
------------------------------------
Zero gradient buffers
------------------------------------
*/

void dense_zero_grad(DenseLayer *layer)
{
    int weight_size =
        layer->grad_weights.rows * layer->grad_weights.cols;

    for (int i = 0; i < weight_size; i++)
    {
        layer->grad_weights.data[i] = 0.0;
    }

    for (int i = 0; i < layer->grad_bias.cols; i++)
    {
        layer->grad_bias.data[i] = 0.0;
    }
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
    tensor_free(&layer->grad_weights);
    tensor_free(&layer->grad_bias);
}