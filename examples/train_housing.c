#include "../include/tinyml.h"
#include <stdio.h>

int main()
{
    Dataset full =
        dataset_load_csv("data/housing.csv",
                         20640,
                         8);

    Dataset train_ds;
    Dataset test_ds;

    dataset_split(&full,
                  &train_ds,
                  &test_ds,
                  0.8);

    NormalizationStats stats =
        normalization_stats_create(train_ds.num_features);

    dataset_fit_normalization(&train_ds, &stats);
    dataset_apply_normalization(&test_ds, &stats);

    NeuralNetwork net;
    network_init(&net);

    network_add(&net,
        dense_create(8, 32));
    network_add(&net,
        dense_create(32, 16));
    network_add(&net,
        dense_create(16, 1));

    TrainingConfig config;
    config.epochs = 20;
    config.batch_size = 32;
    config.learning_rate = 0.0005;
    config.l2_lambda = 1e-4;
    config.early_stopping_patience = 5;
    config.checkpoint_path = "best_housing_model.bin";
    config.history_path = "training_history.csv";

    train(&net, &train_ds, &test_ds, config);

    printf("\nLoading best checkpoint...\n");
    network_load(&net, "best_housing_model.bin");

    printf("\nEvaluating best model...\n");

    double mse =
        evaluate_mse(&net, &test_ds);

    double rmse =
        evaluate_rmse(&net, &test_ds);

    printf("Best Test MSE: %.6f\n", mse);
    printf("Best Test RMSE: %.6f\n", rmse);

    network_save(&net, "housing_model.bin");
    normalization_stats_save(&stats, "housing_stats.bin");

    double sample_house[8] = {
        -122.23, 37.88, 41.0, 880.0,
        129.0, 322.0, 126.0, 8.3252
    };

    double predicted_price =
        predict_sample(&net,
                       sample_house,
                       8,
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