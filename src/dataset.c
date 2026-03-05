#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../include/tinyml.h"

/*
====================================
Create Dataset
====================================
*/

Dataset dataset_create(int samples, int features)
{
    Dataset ds;

    ds.num_samples = samples;
    ds.num_features = features;

    ds.X = tensor_create(samples, features);
    ds.y = tensor_create(samples, 1);

    return ds;
}

/*
====================================
Free Dataset
====================================
*/

void dataset_free(Dataset *ds)
{
    tensor_free(&ds->X);
    tensor_free(&ds->y);
}

/*
====================================
Load CSV Dataset
====================================
*/

Dataset dataset_load_csv(const char *filename,
                         int num_samples,
                         int num_features)
{
    printf("Attempting to open dataset: %s\n", filename);

    FILE *file = fopen(filename, "r");

    if (!file)
    {
        perror("Error opening dataset");
        printf("Current working directory may be incorrect.\n");
        exit(1);
    }

    printf("Dataset opened successfully.\n");

    Dataset ds = dataset_create(num_samples, num_features);

    char line[4096];

    /* Skip header */
    if (!fgets(line, sizeof(line), file))
    {
        printf("Failed to read CSV header\n");
        exit(1);
    }

    printf("CSV Header: %s\n", line);

    int row = 0;

    while (fgets(line, sizeof(line), file))
    {
        if (row >= num_samples)
            break;

        /* Remove newline */
        line[strcspn(line, "\r\n")] = '\0';

        char *token = strtok(line, ",");

        if (!token)
        {
            printf("Empty row at %d\n", row);
            continue;
        }

        /* Read feature columns */
        for (int col = 0; col < num_features; col++)
        {
            if (!token)
            {
                printf("Missing feature at row %d col %d\n", row, col);
                exit(1);
            }

            double value = atof(token);

            ds.X.data[row * num_features + col] = value;

            token = strtok(NULL, ",");
        }

        /* Target column */
        if (!token)
        {
            printf("Missing target at row %d\n", row);
            exit(1);
        }

        ds.y.data[row] = atof(token);

        /* Skip any remaining columns (like categorical ocean_proximity) */

        row++;

        /* Print first few rows for debugging */
        if (row < 5)
        {
            printf("Row %d loaded\n", row);
        }
    }

    fclose(file);

    printf("Loaded dataset: %d samples\n", row);

    if (row == 0)
    {
        printf("Dataset appears empty!\n");
        exit(1);
    }

    return ds;
}

/*
------------------------------------
Normalize Dataset (Standardization)
------------------------------------
*/

void dataset_normalize(Dataset *ds)
{
    printf("Normalizing dataset...\n");

    for(int j = 0; j < ds->num_features; j++)
    {
        double mean = 0.0;
        double std = 0.0;

        /* compute mean */
        for(int i = 0; i < ds->num_samples; i++)
        {
            mean += ds->X.data[i * ds->num_features + j];
        }

        mean /= ds->num_samples;

        /* compute variance */
        for(int i = 0; i < ds->num_samples; i++)
        {
            double val = ds->X.data[i * ds->num_features + j];
            std += (val - mean) * (val - mean);
        }

        std = sqrt(std / ds->num_samples);

        /* avoid division by zero */
        if(std == 0)
            std = 1;

        /* normalize column */
        for(int i = 0; i < ds->num_samples; i++)
        {
            int idx = i * ds->num_features + j;

            ds->X.data[idx] =
                (ds->X.data[idx] - mean) / std;
        }

        printf("Feature %d normalized (mean=%.3f std=%.3f)\n",
               j, mean, std);
    }

    printf("Dataset normalization complete\n");
}

/*
====================================
Shuffle Dataset
====================================
*/

void dataset_shuffle(Dataset *ds)
{
    for (int i = ds->num_samples - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);

        for (int k = 0; k < ds->num_features; k++)
        {
            double tmp = ds->X.data[i * ds->num_features + k];

            ds->X.data[i * ds->num_features + k] =
                ds->X.data[j * ds->num_features + k];

            ds->X.data[j * ds->num_features + k] = tmp;
        }

        double tmpy = ds->y.data[i];
        ds->y.data[i] = ds->y.data[j];
        ds->y.data[j] = tmpy;
    }
}