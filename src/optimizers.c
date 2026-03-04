#include <stdio.h>
#include "../include/tinyml.h"

/*
====================================
Create SGD Optimizer
====================================
*/

SGD sgd_create(double learning_rate)
{
    SGD opt;
    opt.learning_rate = learning_rate;
    return opt;
}

/*
====================================
SGD Parameter Update
====================================

param = param - lr * gradient
*/

void sgd_update(Tensor *param, Tensor *grad, SGD *opt)
{
    int size = param->rows * param->cols;

    for (int i = 0; i < size; i++)
    {
        param->data[i] -= opt->learning_rate * grad->data[i];
    }
}