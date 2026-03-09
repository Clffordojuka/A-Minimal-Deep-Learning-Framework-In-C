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

    /* batch gradient buffers */
    Tensor grad_weights;
    Tensor grad_bias;

    /* Adam moment buffers */
    Tensor m_weights;
    Tensor v_weights;
    Tensor m_bias;
    Tensor v_bias;

} DenseLayer;

/*
------------------------------------
Layer API
------------------------------------
*/

DenseLayer dense_create(int input_size, int output_size);

Tensor dense_forward(DenseLayer *layer, Tensor *input);

/* accumulate gradients only */
Tensor dense_backward(DenseLayer *layer,
                      Tensor *grad_output);

/* apply accumulated gradients using SGD + L2 */
void dense_apply_gradients(DenseLayer *layer,
                           double learning_rate,
                           int batch_size,
                           double l2_lambda);

/* apply accumulated gradients using Adam + L2 */
void dense_apply_gradients_adam(DenseLayer *layer,
                                double learning_rate,
                                double beta1,
                                double beta2,
                                double epsilon,
                                int timestep,
                                int batch_size,
                                double l2_lambda);

/* zero gradient buffers */
void dense_zero_grad(DenseLayer *layer);

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

/* accumulate gradients only */
void network_backward(NeuralNetwork *net,
                      Tensor *grad);

/* apply gradients to all layers using SGD + L2 */
void network_step(NeuralNetwork *net,
                  double learning_rate,
                  int batch_size,
                  double l2_lambda);

/* apply gradients to all layers using Adam + L2 */
void network_step_adam(NeuralNetwork *net,
                       double learning_rate,
                       double beta1,
                       double beta2,
                       double epsilon,
                       int timestep,
                       int batch_size,
                       double l2_lambda);

/* zero gradients for all layers */
void network_zero_grad(NeuralNetwork *net);

/* save / load trained weights */
void network_save(NeuralNetwork *net, const char *filename);
void network_load(NeuralNetwork *net, const char *filename);

void network_free(NeuralNetwork *net);

/*
================================================
OPTIMIZERS
================================================
*/

typedef struct {
    double learning_rate;
} SGD;

typedef struct {
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;
    int timestep;
} Adam;

/*
------------------------------------
Optimizer API
------------------------------------
*/

SGD sgd_create(double learning_rate);

void sgd_update(Tensor *param,
                Tensor *grad,
                SGD *opt);

Adam adam_create(double learning_rate,
                 double beta1,
                 double beta2,
                 double epsilon);

void adam_update(Tensor *param,
                 Tensor *grad,
                 Tensor *m,
                 Tensor *v,
                 Adam *opt);

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
    double l2_lambda;

    int early_stopping_patience;
    const char *checkpoint_path;

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
           Dataset *train_dataset,
           Dataset *val_dataset,
           TrainingConfig config);

#endif