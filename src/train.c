#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tinyml.h"

/*
====================================
Mean Squared Error
====================================
*/

double mse_loss(Tensor *pred, Tensor *target)
{
    double loss = 0.0;

    for (int i = 0; i < pred->rows * pred->cols; i++)
    {
        double diff = pred->data[i] - target->data[i];
        loss += diff * diff;
    }

    return loss / (pred->rows * pred->cols);
}

/*
====================================
MSE Backward
====================================
*/

Tensor mse_backward(Tensor *pred, Tensor *target)
{
    Tensor grad = tensor_create(pred->rows, pred->cols);

    for (int i = 0; i < pred->rows * pred->cols; i++)
    {
        grad.data[i] =
            2.0 * (pred->data[i] - target->data[i]) /
            (pred->rows * pred->cols);
    }

    return grad;
}

/*
====================================
Training Loop
====================================
*/

void train(NeuralNetwork *net,
           Dataset *dataset,
           TrainingConfig config)
{
    printf("Training started\n");

    int samples = dataset->num_samples;
    int features = dataset->num_features;

    printf("Samples: %d\n", samples);
    printf("Features: %d\n", features);

    Tensor input = tensor_create(1, features);
    Tensor target = tensor_create(1, 1);

    for (int epoch = 0; epoch < config.epochs; epoch++)
    {
        printf("Epoch %d started\n", epoch);

        dataset_shuffle(dataset);

        double epoch_loss = 0;

        for (int i = 0; i < samples; i++)
        {
            /* Load sample */

            for (int j = 0; j < features; j++)
                input.data[j] =
                    dataset->X.data[i * features + j];

            target.data[0] = dataset->y.data[i];

            /* Forward pass */

            Tensor pred = network_forward(net, &input);

            if (pred.data == NULL)
            {
                printf("ERROR: forward returned null tensor\n");
                exit(1);
            }

            /* Loss */

            double loss = mse_loss(&pred, &target);

            epoch_loss += loss;

            /* Backprop */

            Tensor grad = mse_backward(&pred, &target);

            network_backward(net,
                             &grad,
                             config.learning_rate);

            tensor_free(&pred);
            tensor_free(&grad);

            if (i % 5000 == 0)
                printf("Processed sample %d\n", i);
        }

        printf("Epoch %d | Loss %.6f\n",
               epoch,
               epoch_loss / samples);
    }

    tensor_free(&input);
    tensor_free(&target);

    printf("Training finished\n");
}

/*
====================================
Evaluation - MSE
====================================
*/

double evaluate_mse(NeuralNetwork *net,
                    Dataset *dataset)
{
    int samples = dataset->num_samples;
    int features = dataset->num_features;

    Tensor input = tensor_create(1, features);

    double total_loss = 0;

    for (int i = 0; i < samples; i++)
    {
        for (int j = 0; j < features; j++)
            input.data[j] =
                dataset->X.data[i * features + j];

        Tensor pred = network_forward(net, &input);

        double diff =
            pred.data[0] - dataset->y.data[i];

        total_loss += diff * diff;

        tensor_free(&pred);
    }

    tensor_free(&input);

    return total_loss / samples;
}


/*
====================================
Evaluation - RMSE
====================================
*/

double evaluate_rmse(NeuralNetwork *net,
                     Dataset *dataset)
{
    double mse =
        evaluate_mse(net, dataset);

    return sqrt(mse);
}