#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "../include/tinyml.h"

/*
------------------------------------
Create Tensor
------------------------------------
*/

Tensor tensor_create(int rows, int cols)
{
    Tensor t;

    t.rows = rows;
    t.cols = cols;

    t.data = (double*)malloc(sizeof(double) * rows * cols);

    if (!t.data)
    {
        printf("Tensor allocation failed\n");
        exit(1);
    }

    return t;
}

/*
------------------------------------
Free Tensor
------------------------------------
*/

void tensor_free(Tensor *t)
{
    if (t->data != NULL)
    {
        free(t->data);
        t->data = NULL;
    }
}

/*
------------------------------------
Xavier Random Initialization
------------------------------------
*/

void tensor_random(Tensor *t)
{
    static int seeded = 0;

    if (!seeded)
    {
        srand((unsigned int)time(NULL));
        seeded = 1;
    }

    double limit = sqrt(6.0 / (t->rows + t->cols));

    for (int i = 0; i < t->rows * t->cols; i++)
    {
        double r = (double)rand() / RAND_MAX;
        t->data[i] = (2.0 * r - 1.0) * limit;
    }
}

/*
------------------------------------
Matrix Multiplication
------------------------------------
*/

Tensor tensor_matmul(Tensor *a, Tensor *b)
{
    if (a->cols != b->rows)
    {
        printf("ERROR: matmul dimension mismatch: (%d x %d) * (%d x %d)\n",
               a->rows, a->cols,
               b->rows, b->cols);
        exit(1);
    }

    Tensor out = tensor_create(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < b->cols; j++)
        {
            double sum = 0;

            for (int k = 0; k < a->cols; k++)
            {
                sum += a->data[i * a->cols + k] *
                       b->data[k * b->cols + j];
            }

            out.data[i * out.cols + j] = sum;
        }
    }

    return out;
}

/*
------------------------------------
Tensor Add
------------------------------------
*/

Tensor tensor_add(Tensor *a, Tensor *b)
{
    Tensor out = tensor_create(a->rows, a->cols);

    for (int i = 0; i < a->rows * a->cols; i++)
    {
        out.data[i] = a->data[i] + b->data[i];
    }

    return out;
}

/*
------------------------------------
Add Bias
------------------------------------
*/

Tensor tensor_add_bias(Tensor *a, Tensor *bias)
{
    Tensor out = tensor_create(a->rows, a->cols);

    for (int i = 0; i < a->rows; i++)
    {
        for (int j = 0; j < a->cols; j++)
        {
            out.data[i * a->cols + j] =
                a->data[i * a->cols + j] +
                bias->data[j];
        }
    }

    return out;
}

/*
------------------------------------
Transpose
------------------------------------
*/

Tensor tensor_transpose(Tensor *t)
{
    Tensor out = tensor_create(t->cols, t->rows);

    for (int i = 0; i < t->rows; i++)
    {
        for (int j = 0; j < t->cols; j++)
        {
            out.data[j * out.cols + i] =
                t->data[i * t->cols + j];
        }
    }

    return out;
}

/*
------------------------------------
ReLU Activation
------------------------------------
*/

void tensor_relu(Tensor *t)
{
    for (int i = 0; i < t->rows * t->cols; i++)
    {
        if (t->data[i] < 0)
            t->data[i] = 0;
    }
}

/*
------------------------------------
ReLU Backward
------------------------------------
*/

Tensor tensor_relu_backward(Tensor *grad, Tensor *z)
{
    Tensor out = tensor_create(z->rows, z->cols);

    for (int i = 0; i < z->rows * z->cols; i++)
    {
        out.data[i] =
            z->data[i] > 0 ?
            grad->data[i] : 0;
    }

    return out;
}

/*
------------------------------------
Tensor Print
------------------------------------
*/

void tensor_print(Tensor *t)
{
    for (int i = 0; i < t->rows; i++)
    {
        for (int j = 0; j < t->cols; j++)
        {
            printf("%.4f ", t->data[i * t->cols + j]);
        }

        printf("\n");
    }

    printf("\n");
}