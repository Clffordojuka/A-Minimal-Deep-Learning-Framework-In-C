#include <stdio.h>
#include "../include/tinyml.h"

/*
------------------------------------
Initialize Network
------------------------------------
*/

void network_init(NeuralNetwork *net)
{
    net->num_layers = 0;
}

/*
------------------------------------
Add Layer
------------------------------------
*/

void network_add(NeuralNetwork *net, DenseLayer layer)
{
    if (net->num_layers >= MAX_LAYERS)
    {
        printf("Error: too many layers\n");
        exit(1);
    }

    net->layers[net->num_layers++] = layer;
}

/*
------------------------------------
Forward Pass
------------------------------------
*/

Tensor network_forward(NeuralNetwork *net, Tensor *input)
{
    Tensor current = *input;
    Tensor next;

    for (int i = 0; i < net->num_layers; i++)
    {
        next = dense_forward(&net->layers[i], &current);

        /* ReLU only on hidden layers */
        if (i < net->num_layers - 1)
        {
            tensor_relu(&next);
        }

        current = next;
    }

    return current;
}

/*
------------------------------------
Backward Pass (Accumulate only)
------------------------------------
*/

void network_backward(NeuralNetwork *net,
                      Tensor *grad)
{
    Tensor current_grad = *grad;
    Tensor new_grad;

    for (int i = net->num_layers - 1; i >= 0; i--)
    {
        if (i < net->num_layers - 1)
        {
            Tensor relu_grad =
                tensor_relu_backward(&current_grad,
                                     &net->layers[i].z);

            if (i != net->num_layers - 1)
            {
                tensor_free(&current_grad);
            }

            current_grad = relu_grad;
        }

        new_grad =
            dense_backward(&net->layers[i],
                           &current_grad);

        if (i != net->num_layers - 1)
        {
            tensor_free(&current_grad);
        }

        current_grad = new_grad;
    }

    tensor_free(&current_grad);
}

/*
------------------------------------
Apply gradients to all layers
------------------------------------
*/

void network_step(NeuralNetwork *net,
                  double learning_rate,
                  int batch_size)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        dense_apply_gradients(&net->layers[i],
                              learning_rate,
                              batch_size);
    }
}

/*
------------------------------------
Zero gradients for all layers
------------------------------------
*/

void network_zero_grad(NeuralNetwork *net)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        dense_zero_grad(&net->layers[i]);
    }
}

/*
------------------------------------
Free Network
------------------------------------
*/

void network_free(NeuralNetwork *net)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        dense_free(&net->layers[i]);
    }
}