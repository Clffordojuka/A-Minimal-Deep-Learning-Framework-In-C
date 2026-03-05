#include <stdio.h>
#include "../include/tinyml.h"

int main()
{
   Dataset ds = dataset_load_csv(
    "data/housing.csv",
    20640,
    8
);

dataset_normalize(&ds);

    printf("Loaded dataset: %d samples\n",
           ds.num_samples);

    NeuralNetwork net;

    network_init(&net);

    network_add(&net, dense_create(8, 16));
    network_add(&net, dense_create(16, 1));

    TrainingConfig config;

    config.epochs = 5;
    config.learning_rate = 0.00001;

    train(&net, &ds, config);

    dataset_free(&ds);
    network_free(&net);

    return 0;
}