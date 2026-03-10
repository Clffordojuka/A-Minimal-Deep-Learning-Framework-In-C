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

        line[strcspn(line, "\r\n")] = '\0';

        char *token = strtok(line, ",");

        if (!token)
        {
            printf("Empty row at %d\n", row);
            continue;
        }

        for (int col = 0; col < num_features; col++)
        {
            if (!token)
            {
                printf("Missing feature at row %d col %d\n", row, col);
                exit(1);
            }

            ds.X.data[row * num_features + col] = atof(token);
            token = strtok(NULL, ",");
        }

        if (!token)
        {
            printf("Missing target at row %d\n", row);
            exit(1);
        }

        ds.y.data[row] = atof(token);

        row++;

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
====================================
Normalization Stats
====================================
*/

NormalizationStats normalization_stats_create(int num_features)
{
    NormalizationStats stats;

    stats.num_features = num_features;
    stats.feature_mean = (double *)malloc(sizeof(double) * num_features);
    stats.feature_std = (double *)malloc(sizeof(double) * num_features);
    stats.target_mean = 0.0;
    stats.target_std = 1.0;

    if (!stats.feature_mean || !stats.feature_std)
    {
        printf("Failed to allocate normalization stats\n");
        exit(1);
    }

    return stats;
}

void normalization_stats_free(NormalizationStats *stats)
{
    if (stats->feature_mean != NULL)
    {
        free(stats->feature_mean);
        stats->feature_mean = NULL;
    }

    if (stats->feature_std != NULL)
    {
        free(stats->feature_std);
        stats->feature_std = NULL;
    }
}

/*
------------------------------------
Fit normalization stats on dataset and apply
------------------------------------
*/

void dataset_fit_normalization(Dataset *ds,
                               NormalizationStats *stats)
{
    printf("Fitting and applying normalization...\n");

    for (int j = 0; j < ds->num_features; j++)
    {
        double mean = 0.0;
        double std = 0.0;

        for (int i = 0; i < ds->num_samples; i++)
        {
            mean += ds->X.data[i * ds->num_features + j];
        }

        mean /= ds->num_samples;

        for (int i = 0; i < ds->num_samples; i++)
        {
            double val = ds->X.data[i * ds->num_features + j];
            std += (val - mean) * (val - mean);
        }

        std = sqrt(std / ds->num_samples);

        if (std == 0.0)
            std = 1.0;

        stats->feature_mean[j] = mean;
        stats->feature_std[j] = std;

        for (int i = 0; i < ds->num_samples; i++)
        {
            int idx = i * ds->num_features + j;
            ds->X.data[idx] =
                (ds->X.data[idx] - mean) / std;
        }

        printf("Feature %d normalized (mean=%.3f std=%.3f)\n",
               j, mean, std);
    }

    double mean = 0.0;
    double std = 0.0;

    for (int i = 0; i < ds->num_samples; i++)
        mean += ds->y.data[i];

    mean /= ds->num_samples;

    for (int i = 0; i < ds->num_samples; i++)
    {
        double v = ds->y.data[i];
        std += (v - mean) * (v - mean);
    }

    std = sqrt(std / ds->num_samples);

    if (std == 0.0)
        std = 1.0;

    stats->target_mean = mean;
    stats->target_std = std;

    for (int i = 0; i < ds->num_samples; i++)
    {
        ds->y.data[i] =
            (ds->y.data[i] - mean) / std;
    }

    printf("Target normalized (mean=%.3f std=%.3f)\n",
           mean, std);

    printf("Dataset normalization complete\n");
}

/*
------------------------------------
Apply existing normalization stats
------------------------------------
*/

void dataset_apply_normalization(Dataset *ds,
                                 NormalizationStats *stats)
{
    printf("Applying existing normalization stats...\n");

    for (int j = 0; j < ds->num_features; j++)
    {
        double mean = stats->feature_mean[j];
        double std = stats->feature_std[j];

        if (std == 0.0)
            std = 1.0;

        for (int i = 0; i < ds->num_samples; i++)
        {
            int idx = i * ds->num_features + j;
            ds->X.data[idx] =
                (ds->X.data[idx] - mean) / std;
        }
    }

    for (int i = 0; i < ds->num_samples; i++)
    {
        ds->y.data[i] =
            (ds->y.data[i] - stats->target_mean) / stats->target_std;
    }
}

/*
------------------------------------
Backward-compatible helper
------------------------------------
*/

void dataset_normalize(Dataset *ds)
{
    NormalizationStats stats =
        normalization_stats_create(ds->num_features);

    dataset_fit_normalization(ds, &stats);
    normalization_stats_free(&stats);
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

/*
====================================
Dataset Split
====================================
*/

void dataset_split(Dataset *full,
                   Dataset *train,
                   Dataset *test,
                   double train_ratio)
{
    int train_samples =
        (int)(full->num_samples * train_ratio);

    int test_samples =
        full->num_samples - train_samples;

    *train = dataset_create(train_samples,
                            full->num_features);

    *test = dataset_create(test_samples,
                           full->num_features);

    for (int i = 0; i < train_samples; i++)
    {
        for (int j = 0; j < full->num_features; j++)
        {
            train->X.data[i * full->num_features + j] =
                full->X.data[i * full->num_features + j];
        }

        train->y.data[i] = full->y.data[i];
    }

    for (int i = 0; i < test_samples; i++)
    {
        for (int j = 0; j < full->num_features; j++)
        {
            test->X.data[i * full->num_features + j] =
                full->X.data[(train_samples + i) * full->num_features + j];
        }

        test->y.data[i] = full->y.data[train_samples + i];
    }

    printf("Dataset split complete\n");
    printf("Train samples: %d\n", train_samples);
    printf("Test samples: %d\n", test_samples);
}

/*
====================================
Input normalization / target denormalization
====================================
*/

void normalize_input(double *raw_input,
                     double *normalized_input,
                     int num_features,
                     NormalizationStats *stats)
{
    for (int i = 0; i < num_features; i++)
    {
        double std = stats->feature_std[i];
        if (std == 0.0)
            std = 1.0;

        normalized_input[i] =
            (raw_input[i] - stats->feature_mean[i]) / std;
    }
}

double denormalize_target(double normalized_value,
                          NormalizationStats *stats)
{
    return normalized_value * stats->target_std + stats->target_mean;
}

/*
====================================
Save / Load normalization stats
====================================
*/

void normalization_stats_save(NormalizationStats *stats,
                              const char *filename)
{
    FILE *fp = fopen(filename, "wb");

    if (fp == NULL)
    {
        printf("Error: could not open stats file for saving: %s\n", filename);
        exit(1);
    }

    fwrite(&stats->num_features, sizeof(int), 1, fp);
    fwrite(stats->feature_mean, sizeof(double), stats->num_features, fp);
    fwrite(stats->feature_std, sizeof(double), stats->num_features, fp);
    fwrite(&stats->target_mean, sizeof(double), 1, fp);
    fwrite(&stats->target_std, sizeof(double), 1, fp);

    fclose(fp);

    printf("Normalization stats saved to %s\n", filename);
}

NormalizationStats normalization_stats_load(const char *filename)
{
    FILE *fp = fopen(filename, "rb");

    if (fp == NULL)
    {
        printf("Error: could not open stats file for loading: %s\n", filename);
        exit(1);
    }

    int num_features = 0;
    fread(&num_features, sizeof(int), 1, fp);

    NormalizationStats stats =
        normalization_stats_create(num_features);

    fread(stats.feature_mean, sizeof(double), num_features, fp);
    fread(stats.feature_std, sizeof(double), num_features, fp);
    fread(&stats.target_mean, sizeof(double), 1, fp);
    fread(&stats.target_std, sizeof(double), 1, fp);

    fclose(fp);

    printf("Normalization stats loaded from %s\n", filename);

    return stats;
}

/*
====================================
Single-sample prediction
====================================
*/

double predict_sample(NeuralNetwork *net,
                      double *raw_features,
                      int num_features,
                      NormalizationStats *stats)
{
    double *normalized =
        (double *)malloc(sizeof(double) * num_features);

    if (normalized == NULL)
    {
        printf("Failed to allocate prediction input buffer\n");
        exit(1);
    }

    normalize_input(raw_features,
                    normalized,
                    num_features,
                    stats);

    Tensor input = tensor_create(1, num_features);

    for (int i = 0; i < num_features; i++)
    {
        input.data[i] = normalized[i];
    }

    Tensor pred = network_forward(net, &input);

    double value =
        denormalize_target(pred.data[0], stats);

    tensor_free(&pred);
    tensor_free(&input);
    free(normalized);

    return value;
}