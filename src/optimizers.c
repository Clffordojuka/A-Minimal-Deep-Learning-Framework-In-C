#include <stdio.h>
#include <math.h>
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

/*
====================================
Create Adam Optimizer
====================================
*/

Adam adam_create(double learning_rate,
                 double beta1,
                 double beta2,
                 double epsilon)
{
    Adam opt;
    opt.learning_rate = learning_rate;
    opt.beta1 = beta1;
    opt.beta2 = beta2;
    opt.epsilon = epsilon;
    opt.timestep = 0;
    return opt;
}

/*
====================================
Adam Parameter Update
====================================

m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g^2

m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)

param = param - lr * m_hat / (sqrt(v_hat) + epsilon)
*/

void adam_update(Tensor *param,
                 Tensor *grad,
                 Tensor *m,
                 Tensor *v,
                 Adam *opt)
{
    int size = param->rows * param->cols;

    opt->timestep += 1;

    for (int i = 0; i < size; i++)
    {
        double g = grad->data[i];

        m->data[i] =
            opt->beta1 * m->data[i] +
            (1.0 - opt->beta1) * g;

        v->data[i] =
            opt->beta2 * v->data[i] +
            (1.0 - opt->beta2) * g * g;

        double m_hat =
            m->data[i] /
            (1.0 - pow(opt->beta1, opt->timestep));

        double v_hat =
            v->data[i] /
            (1.0 - pow(opt->beta2, opt->timestep));

        param->data[i] -=
            opt->learning_rate *
            m_hat / (sqrt(v_hat) + opt->epsilon);
    }
}