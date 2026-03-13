#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../include/tinyml.h"

/*
------------------------------------
Helpers
------------------------------------
*/

static void trim(char *s)
{
    char *start = s;
    while (*start && isspace((unsigned char)*start))
        start++;

    if (start != s)
        memmove(s, start, strlen(start) + 1);

    int len = (int)strlen(s);
    while (len > 0 && isspace((unsigned char)s[len - 1]))
    {
        s[len - 1] = '\0';
        len--;
    }
}

static void copy_string(char *dst, const char *src, int max_len)
{
    strncpy(dst, src, max_len - 1);
    dst[max_len - 1] = '\0';
}

static void parse_hidden_layers(const char *value, ExperimentConfig *cfg)
{
    char buffer[512];
    copy_string(buffer, value, sizeof(buffer));

    cfg->num_hidden_layers = 0;

    char *token = strtok(buffer, ",");
    while (token != NULL && cfg->num_hidden_layers < MAX_LAYERS)
    {
        trim(token);
        cfg->hidden_layers[cfg->num_hidden_layers++] = atoi(token);
        token = strtok(NULL, ",");
    }

    if (cfg->num_hidden_layers == 0)
    {
        printf("Error: hidden_layers must contain at least one layer\n");
        exit(1);
    }
}

/*
------------------------------------
Default Config
------------------------------------
*/

void experiment_config_init(ExperimentConfig *cfg)
{
    copy_string(cfg->dataset_path, "data/housing.csv", MAX_PATH_LEN);
    cfg->num_samples = 20640;
    cfg->num_features = 8;
    cfg->train_ratio = 0.8;

    cfg->num_hidden_layers = 2;
    cfg->hidden_layers[0] = 32;
    cfg->hidden_layers[1] = 16;

    cfg->epochs = 20;
    cfg->batch_size = 32;
    cfg->learning_rate = 0.0005;
    cfg->l2_lambda = 1e-4;
    cfg->early_stopping_patience = 5;

    copy_string(cfg->checkpoint_path, "best_housing_model.bin", MAX_PATH_LEN);
    copy_string(cfg->history_path, "training_history.csv", MAX_PATH_LEN);
    copy_string(cfg->model_path, "housing_model.bin", MAX_PATH_LEN);
    copy_string(cfg->stats_path, "housing_stats.bin", MAX_PATH_LEN);
}

/*
------------------------------------
Load Config File
------------------------------------
*/

int experiment_config_load(const char *filename, ExperimentConfig *cfg)
{
    FILE *fp = fopen(filename, "r");

    if (fp == NULL)
    {
        printf("Error: could not open config file: %s\n", filename);
        return 0;
    }

    char line[1024];

    while (fgets(line, sizeof(line), fp))
    {
        trim(line);

        if (line[0] == '\0' || line[0] == '#')
            continue;

        char *equals = strchr(line, '=');
        if (equals == NULL)
            continue;

        *equals = '\0';

        char *key = line;
        char *value = equals + 1;

        trim(key);
        trim(value);

        if (strcmp(key, "dataset_path") == 0)
            copy_string(cfg->dataset_path, value, MAX_PATH_LEN);
        else if (strcmp(key, "num_samples") == 0)
            cfg->num_samples = atoi(value);
        else if (strcmp(key, "num_features") == 0)
            cfg->num_features = atoi(value);
        else if (strcmp(key, "train_ratio") == 0)
            cfg->train_ratio = atof(value);
        else if (strcmp(key, "hidden_layers") == 0)
            parse_hidden_layers(value, cfg);
        else if (strcmp(key, "epochs") == 0)
            cfg->epochs = atoi(value);
        else if (strcmp(key, "batch_size") == 0)
            cfg->batch_size = atoi(value);
        else if (strcmp(key, "learning_rate") == 0)
            cfg->learning_rate = atof(value);
        else if (strcmp(key, "l2_lambda") == 0)
            cfg->l2_lambda = atof(value);
        else if (strcmp(key, "early_stopping_patience") == 0)
            cfg->early_stopping_patience = atoi(value);
        else if (strcmp(key, "checkpoint_path") == 0)
            copy_string(cfg->checkpoint_path, value, MAX_PATH_LEN);
        else if (strcmp(key, "history_path") == 0)
            copy_string(cfg->history_path, value, MAX_PATH_LEN);
        else if (strcmp(key, "model_path") == 0)
            copy_string(cfg->model_path, value, MAX_PATH_LEN);
        else if (strcmp(key, "stats_path") == 0)
            copy_string(cfg->stats_path, value, MAX_PATH_LEN);
    }

    fclose(fp);
    return 1;
}