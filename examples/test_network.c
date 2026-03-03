#include <stdio.h>
#include "../include/tinyml.h"

int main()
{
    Tensor input = tensor_create(1,3);

    tensor_random(&input);

    NeuralNetwork net;

    network_init(&net);

    network_add(&net, dense_create(3,5));
    network_add(&net, dense_create(5,4));
    network_add(&net, dense_create(4,1));

    Tensor out = network_forward(&net,&input);

    printf("Input:\n");
    tensor_print(&input);

    printf("Output:\n");
    tensor_print(&out);

    network_free(&net);
    tensor_free(&input);
    tensor_free(&out);

    return 0;
}