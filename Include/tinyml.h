#ifndef TINYML_H
#define TINYML_H

#define MAX_LAYERS 32

#include <stdlib.h>
#include <stdio.h>

/*
====================================
Tensor Structure
====================================
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
Tensor tensor_transpose(Tensor *t);

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

typedef struct {

    Tensor weights;
    Tensor bias;

    Tensor input;
    Tensor z;
    Tensor output;

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


/*
====================================
NEURAL NETWORK
====================================
*/

typedef struct {

    int num_layers;
    DenseLayer layers[MAX_LAYERS];

} NeuralNetwork;

/*
------------------------------------
Network API
------------------------------------
*/

void network_init(NeuralNetwork *net);

void network_add(NeuralNetwork *net, DenseLayer layer);

Tensor network_forward(NeuralNetwork *net, Tensor *input);

void network_backward(NeuralNetwork *net,
                      Tensor *grad,
                      double learning_rate);

void network_free(NeuralNetwork *net);


/*
================================================
OPTIMIZERS
================================================
*/

typedef struct {

    double learning_rate;

} SGD;

/*
------------------------------------
Optimizer API
------------------------------------
*/

SGD sgd_create(double learning_rate);

void sgd_update(Tensor *param,
                Tensor *grad,
                SGD *opt);


/*
================================================
DATASET
================================================
*/

typedef struct {

    Tensor X;
    Tensor y;

    int num_samples;
    int num_features;

} Dataset;

/*
------------------------------------
Dataset API
------------------------------------
*/

Dataset dataset_create(int samples, int features);

void dataset_free(Dataset *ds);

Dataset dataset_load_csv(const char *filename,
                         int num_samples,
                         int num_features);

void dataset_shuffle(Dataset *ds);

/*
------------------------------------
Dataset Preprocessing
------------------------------------
*/

void dataset_normalize(Dataset *ds);


/*
------------------------------------
Dataset Split
------------------------------------
*/

void dataset_split(Dataset *full,
                   Dataset *train,
                   Dataset *test,
                   double train_ratio);


/*
------------------------------------
Evaluation
------------------------------------
*/

double evaluate_mse(NeuralNetwork *net,
                    Dataset *dataset);

double evaluate_rmse(NeuralNetwork *net,
                     Dataset *dataset);


/*
================================================
TRAINING ENGINE
================================================
*/

typedef struct {

    int epochs;
    int batch_size;
    double learning_rate;

} TrainingConfig;

/*
------------------------------------
Loss Functions
------------------------------------
*/

double mse_loss(Tensor *pred, Tensor *target);

Tensor mse_backward(Tensor *pred, Tensor *target);

/*
------------------------------------
Training
------------------------------------
*/

void train(NeuralNetwork *net,
           Dataset *dataset,
           TrainingConfig config);

#endif