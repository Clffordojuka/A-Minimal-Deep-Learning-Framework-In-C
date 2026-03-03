#include <stdio.h>
#include "../include/tinyml.h"

int main() {

    Tensor input = tensor_create(1,3);

    tensor_random(&input);

    DenseLayer layer = dense_create(3,4);

    Tensor out = dense_forward(&layer,&input);

    printf("Input:\n");
    tensor_print(&input);

    printf("Output:\n");
    tensor_print(&out);

    dense_free(&layer);
    tensor_free(&input);
    tensor_free(&out);

    return 0;
}