#include "../include/tinyml.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
    const char *config_path = "experiment.cfg";

    if (argc > 1)
    {
        config_path = argv[1];
    }

    ExperimentConfig exp_cfg;
    experiment_config_init(&exp_cfg);

    if (!experiment_config_load(config_path, &exp_cfg))
    {
        return 1;
    }

    printf("Using config: %s\n", config_path);
    printf("Dataset: %s\n", exp_cfg.dataset_path);
    printf("Hidden layers: ");
    for (int i = 0; i < exp_cfg.num_hidden_layers; i++)
    {
        printf("%d", exp_cfg.hidden_layers[i]);
        if (i < exp_cfg.num_hidden_layers - 1)
            printf(", ");
    }
    printf("\n");

    Dataset full =
        dataset_load_csv(exp_cfg.dataset_path,
                         exp_cfg.num_samples,
                         exp_cfg.num_features);

    Dataset train_ds;
    Dataset test_ds;

    dataset_split(&full,
                  &train_ds,
                  &test_ds,
                  exp_cfg.train_ratio);

    NormalizationStats stats =
        normalization_stats_create(train_ds.num_features);

    dataset_fit_normalization(&train_ds, &stats);
    dataset_apply_normalization(&test_ds, &stats);

    NeuralNetwork net;
    network_init(&net);

    int in_size = exp_cfg.num_features;

    for (int i = 0; i < exp_cfg.num_hidden_layers; i++)
    {
        network_add(&net,
            dense_create(in_size, exp_cfg.hidden_layers[i]));
        in_size = exp_cfg.hidden_layers[i];
    }

    network_add(&net, dense_create(in_size, 1));

    TrainingConfig train_cfg;
    train_cfg.epochs = exp_cfg.epochs;
    train_cfg.batch_size = exp_cfg.batch_size;
    train_cfg.learning_rate = exp_cfg.learning_rate;
    train_cfg.l2_lambda = exp_cfg.l2_lambda;
    train_cfg.early_stopping_patience = exp_cfg.early_stopping_patience;
    train_cfg.checkpoint_path = exp_cfg.checkpoint_path;
    train_cfg.history_path = exp_cfg.history_path;

    train(&net, &train_ds, &test_ds, train_cfg);

    printf("\nLoading best checkpoint...\n");
    network_load(&net, exp_cfg.checkpoint_path);

    printf("\nEvaluating best model...\n");

    double mse = evaluate_mse(&net, &test_ds);
    double rmse = evaluate_rmse(&net, &test_ds);

    printf("Best Test MSE: %.6f\n", mse);
    printf("Best Test RMSE: %.6f\n", rmse);

    network_save(&net, exp_cfg.model_path);
    normalization_stats_save(&stats, exp_cfg.stats_path);

    /* example single-sample prediction */
    double sample_house[8] = {
        -122.23, 37.88, 41.0, 880.0,
        129.0, 322.0, 126.0, 8.3252
    };

    double predicted_price =
        predict_sample(&net,
                       sample_house,
                       exp_cfg.num_features,
                       &stats);

    printf("\nSingle-sample prediction:\n");
    printf("Predicted house price: $%.2f\n", predicted_price);

    dataset_free(&full);
    dataset_free(&train_ds);
    dataset_free(&test_ds);
    normalization_stats_free(&stats);
    network_free(&net);

    return 0;
}