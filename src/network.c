#include <stdio.h>
#include <stdlib.h>
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
Apply gradients to all layers (SGD + L2)
------------------------------------
*/

void network_step(NeuralNetwork *net,
                  double learning_rate,
                  int batch_size,
                  double l2_lambda)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        dense_apply_gradients(&net->layers[i],
                              learning_rate,
                              batch_size,
                              l2_lambda);
    }
}

/*
------------------------------------
Apply gradients to all layers (Adam + L2)
------------------------------------
*/

void network_step_adam(NeuralNetwork *net,
                       double learning_rate,
                       double beta1,
                       double beta2,
                       double epsilon,
                       int timestep,
                       int batch_size,
                       double l2_lambda)
{
    for (int i = 0; i < net->num_layers; i++)
    {
        dense_apply_gradients_adam(&net->layers[i],
                                   learning_rate,
                                   beta1,
                                   beta2,
                                   epsilon,
                                   timestep,
                                   batch_size,
                                   l2_lambda);
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
Save Model
------------------------------------
*/

void network_save(NeuralNetwork *net, const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (fp == NULL)
    {
        printf("Error: could not open file for saving: %s\n", filename);
        exit(1);
    }

    fwrite(&net->num_layers, sizeof(int), 1, fp);

    for (int i = 0; i < net->num_layers; i++)
    {
        DenseLayer *layer = &net->layers[i];

        fwrite(&layer->weights.rows, sizeof(int), 1, fp);
        fwrite(&layer->weights.cols, sizeof(int), 1, fp);

        fwrite(layer->weights.data,
               sizeof(double),
               layer->weights.rows * layer->weights.cols,
               fp);

        fwrite(layer->bias.data,
               sizeof(double),
               layer->bias.cols,
               fp);
    }

    fclose(fp);

    printf("Model saved to %s\n", filename);
}

/*
------------------------------------
Load Model
------------------------------------
*/

void network_load(NeuralNetwork *net, const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    if (fp == NULL)
    {
        printf("Error: could not open file for loading: %s\n", filename);
        exit(1);
    }

    int saved_num_layers = 0;
    fread(&saved_num_layers, sizeof(int), 1, fp);

    if (saved_num_layers != net->num_layers)
    {
        printf("Error: model layer count mismatch\n");
        printf("Saved model layers: %d, current network layers: %d\n",
               saved_num_layers, net->num_layers);
        fclose(fp);
        exit(1);
    }

    for (int i = 0; i < net->num_layers; i++)
    {
        DenseLayer *layer = &net->layers[i];

        int saved_rows = 0;
        int saved_cols = 0;

        fread(&saved_rows, sizeof(int), 1, fp);
        fread(&saved_cols, sizeof(int), 1, fp);

        if (saved_rows != layer->weights.rows ||
            saved_cols != layer->weights.cols)
        {
            printf("Error: layer %d shape mismatch while loading\n", i);
            printf("Saved: (%d x %d), Current: (%d x %d)\n",
                   saved_rows, saved_cols,
                   layer->weights.rows, layer->weights.cols);
            fclose(fp);
            exit(1);
        }

        fread(layer->weights.data,
              sizeof(double),
              layer->weights.rows * layer->weights.cols,
              fp);

        fread(layer->bias.data,
              sizeof(double),
              layer->bias.cols,
              fp);
    }

    fclose(fp);

    printf("Model loaded from %s\n", filename);
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