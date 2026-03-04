#include "../include/tinyml.h"
#include <stdio.h>

int main()
{
    Dataset ds = dataset_load_csv("data/housing.csv", 30000, 8);

    printf("Samples: %d\n", ds.num_samples);
    printf("Features: %d\n", ds.num_features);

    printf("First row:\n");

    for(int i = 0; i < ds.num_features; i++)
        printf("%f ", ds.X.data[i]);

    printf("\nTarget: %f\n", ds.y.data[0]);

    dataset_free(&ds);

    return 0;
}