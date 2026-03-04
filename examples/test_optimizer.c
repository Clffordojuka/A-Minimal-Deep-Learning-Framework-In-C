#include "../include/tinyml.h"
#include <stdio.h>

int main()
{
    Tensor param = tensor_create(1,3);
    Tensor grad  = tensor_create(1,3);

    param.data[0] = 1.0;
    param.data[1] = 2.0;
    param.data[2] = 3.0;

    grad.data[0] = 0.1;
    grad.data[1] = 0.1;
    grad.data[2] = 0.1;

    SGD opt = sgd_create(0.01);

    sgd_step(&opt, &param, &grad);

    tensor_print(&param);

    tensor_free(&param);
    tensor_free(&grad);

    return 0;
}