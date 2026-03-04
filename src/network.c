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
    Tensor out = *input;

    for (int i = 0; i < net->num_layers; i++)
    {
        out = dense_forward(&net->layers[i], &out);
    }

    return out;
}

/*
------------------------------------
Backward Pass
------------------------------------
*/

void network_backward(NeuralNetwork *net,
                      Tensor *grad,
                      double lr)
{
    Tensor current_grad = *grad;

    for (int i = net->num_layers - 1; i >= 0; i--)
    {
        Tensor new_grad =
            dense_backward(&net->layers[i],
                           &current_grad,
                           lr);

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