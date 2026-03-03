#include <stdio.h>
#include "../include/tinyml.h"

int main() {

    Tensor A = tensor_create(2,3);
    Tensor B = tensor_create(3,2);

    tensor_random(&A);
    tensor_random(&B);

    Tensor C = tensor_matmul(&A,&B);

    printf("Matrix A\n");
    tensor_print(&A);

    printf("Matrix B\n");
    tensor_print(&B);

    printf("A x B\n");
    tensor_print(&C);

    tensor_free(&A);
    tensor_free(&B);
    tensor_free(&C);

    return 0;
}