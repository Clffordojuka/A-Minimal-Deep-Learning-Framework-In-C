#include "../include/tinyml.h"
#include <stdio.h>

int main()
{
    Dataset full =
        dataset_load_csv("data/housing.csv",
                         20640,
                         8);

    dataset_normalize(&full);

    Dataset train_ds;
    Dataset test_ds;

    dataset_split(&full,
                  &train_ds,
                  &test_ds,
                  0.8);

    NeuralNetwork net;

    network_init(&net);

    /* deeper architecture */
    network_add(&net,
        dense_create(8, 32));

    network_add(&net,
        dense_create(32, 16));

    network_add(&net,
        dense_create(16, 1));

    TrainingConfig config;

    config.epochs = 20;
    config.batch_size = 32;
    config.learning_rate = 0.0005;
    config.l2_lambda = 1e-4;

    train(&net, &train_ds, config);

    printf("\nEvaluating model...\n");

    double mse =
        evaluate_mse(&net, &test_ds);

    double rmse =
        evaluate_rmse(&net, &test_ds);

    printf("Test MSE: %.6f\n", mse);
    printf("Test RMSE: %.6f\n", rmse);

    dataset_free(&full);
    dataset_free(&train_ds);
    dataset_free(&test_ds);

    network_free(&net);

    return 0;
}