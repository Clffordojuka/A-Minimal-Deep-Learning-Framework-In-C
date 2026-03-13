# TinyML

A tiny deep learning framework built from scratch in pure C.

TinyML is a low-level machine learning project focused on understanding how neural networks work under the hood by implementing the core pieces directly in C. Instead of depending on high-level ML libraries, this project builds the training pipeline manually: tensors, dense layers, forward and backward propagation, optimization, normalization, checkpointing, experiment configuration, metric logging, and inference.

At its current milestone, TinyML can train a multi-layer neural network on the California Housing dataset, save and reload the best model, preserve normalization statistics, log training history to CSV, run experiments from a config file, and make single-sample predictions in real house-price units.

---

## Why TinyML exists

Modern machine learning libraries are powerful, but they also hide many of the details that matter if you want to understand how training actually works.

TinyML was built to make those details visible.

This project is designed to answer questions like:

* How are tensors represented in memory?
* What does a dense layer really compute?
* How does backpropagation work in practice?
* What changes when moving from sample-by-sample SGD to mini-batches?
* Why do optimizers like Adam help?
* How do you save a trained model and use it later for inference?
* How do you make experiment settings reproducible without rewriting source code each time?

The goal is not to replace PyTorch or TensorFlow. The goal is to learn by building.

---

## Current milestone

TinyML currently supports:

* tensor creation and memory management
* matrix multiplication and basic tensor operations
* dense fully connected layers
* ReLU hidden activations
* linear regression output
* manual forward propagation
* manual backward propagation
* mini-batch training
* Adam optimizer
* L2 regularization
* train/test split
* normalization fit on training data only
* validation monitoring
* early stopping
* best-checkpoint model saving and loading
* normalization statistics saving and loading
* single-sample prediction with denormalized output
* training history logging to CSV
* config-driven experiment execution

This means the framework now supports a full practical workflow:

```text
load data -> split -> normalize -> train -> validate -> checkpoint -> log -> save -> load -> predict
```

---

## Project structure

```text
TinyML/
├── data/
│   └── housing.csv
├── include/
│   └── tinyml.h
├── src/
│   ├── tensor.c
│   ├── layers.c
│   ├── network.c
│   ├── optimizers.c
│   ├── dataset.c
│   ├── train.c
│   └── config.c
├── examples/
│   ├── train_housing.c
│   └── run_experiment.c
├── build/
├── experiment.cfg
├── Makefile
└── README.md
```

The public API is exposed through a single header file, `tinyml.h`, while the implementation is split across `src/`. The `examples/` folder contains runnable entry points, including a config-driven experiment runner.

---

## Dataset

The current example uses the California Housing dataset.

Input features:

* longitude
* latitude
* housing_median_age
* total_rooms
* total_bedrooms
* population
* households
* median_income

Target:

* median_house_value

TinyML treats this as a regression problem and predicts a single continuous value.

---

## Development journey

TinyML did not start in its current form.

It began as a toy neural network trained on a tiny manually defined dataset to validate the basics of feedforward computation and gradient descent. From there, the framework evolved step by step.

### 1. Basic feedforward regression network

The first version implemented:

* manually initialized weights
* forward propagation
* mean squared error
* backpropagation

This was enough to prove the core math.

### 2. Real CSV dataset support

The next milestone was moving from tiny hardcoded arrays to a real dataset:

* CSV loading
* larger sample counts
* structured tabular input
* a real regression target

This shifted the project from educational sketch to practical experiment.

### 3. Stable preprocessing

As soon as real data was introduced, scale issues appeared. Feature normalization was added, and later improved so that normalization is fit on the training set only and then applied consistently to test data and future inference.

### 4. Safer framework structure

A lot of engineering work went into fixing:

* memory ownership bugs
* bad tensor lifetimes
* layer/output activation mistakes
* training crashes caused by low-level C issues

These fixes shaped the current architecture of the framework.

### 5. Mini-batch training

Training originally used sample-by-sample gradient updates. That worked, but it was noisy and slow. Mini-batch training was introduced to accumulate gradients and update less frequently, making the loop both cleaner and more realistic.

### 6. Adam optimizer

Once update logic was separated cleanly from backpropagation, Adam was added. This improved convergence and made the framework much more capable on the housing task.

### 7. L2 regularization

As the architecture deepened, overfitting became more visible. L2 regularization was added to improve generalization and stabilize deeper models.

### 8. Checkpointing and early stopping

The framework then grew to support:

* validation monitoring after each epoch
* automatic best-model checkpoint saving
* early stopping when validation no longer improves

This made experiments more reliable and much less wasteful.

### 9. Model save/load

Model serialization was added so trained weights and biases can be saved to disk and loaded into a matching architecture later.

### 10. Real-world inference

The next milestone completed the practical prediction pipeline:

* save normalization statistics
* load normalization statistics
* normalize a raw input sample
* run prediction
* denormalize the output back to a real house price

### 11. Training history logging

TinyML can now write epoch-level metrics to CSV, making it easier to analyze training behavior, compare runs, and plot curves externally.

### 12. Config-driven experiments

The latest upgrade removes the need to edit source code for every experiment. Hyperparameters, architecture choices, file paths, and output artifact names can now be defined in a config file and executed through a reusable experiment runner.

That is where TinyML currently stands.

---

## Current example architecture

A strong current housing example uses:

```text
8 -> 32 -> 16 -> 1
```

with:

* ReLU on hidden layers
* linear output layer
* Adam optimizer
* mini-batch training
* L2 regularization

This is still intentionally simple, but it is strong enough to produce meaningful regression results on the housing task.

---

## Example training flow

The framework now supports two main ways to run experiments.

### `examples/train_housing.c`

This example demonstrates the full workflow directly in code:

1. load the California Housing dataset
2. split it into train and test sets
3. fit normalization on training data
4. apply the same normalization to test data
5. construct the network
6. train using Adam and mini-batches
7. save the best checkpoint
8. reload the best checkpoint
9. evaluate the best model
10. save the trained model
11. save normalization statistics
12. predict the price of a single raw sample

### `examples/run_experiment.c`

This example reads experiment settings from a config file, then performs the same training and evaluation workflow without requiring source edits for each run.

This makes TinyML much more convenient for repeated experiments.

---

## Config-driven experiments

TinyML now supports a lightweight config-file workflow through `experiment.cfg`.

A typical config looks like this:

```ini
# Dataset
dataset_path = data/housing.csv
num_samples = 20640
num_features = 8
train_ratio = 0.8

# Architecture
hidden_layers = 32,16

# Training
epochs = 20
batch_size = 32
learning_rate = 0.0005
l2_lambda = 0.0001
early_stopping_patience = 5

# Output files
checkpoint_path = best_housing_model.bin
history_path = training_history.csv
model_path = housing_model.bin
stats_path = housing_stats.bin
```

This approach makes experiments easier to reproduce and compare.

---

## Training history logging

TinyML can now export training history to CSV during training.

A typical file looks like this:

```csv
epoch,train_loss,val_mse,val_rmse
0,0.804635,0.717934,0.847310
1,0.506493,0.649194,0.805726
2,0.424517,0.683988,0.827036
...
```

This makes it easy to:

* compare model variants
* inspect overfitting
* plot curves in Excel, Python, or Google Sheets
* keep a record of experiments over time

---

## Example output

A typical run looks like this:

```text
Epoch 0 | Train Loss 0.804635 | Val MSE 0.717934 | Val RMSE 0.847310
Epoch 1 | Train Loss 0.506493 | Val MSE 0.649194 | Val RMSE 0.805726
...
Early stopping triggered at epoch 6
Training history saved to training_history.csv

Loading best checkpoint...
Model loaded from best_housing_model.bin

Evaluating best model...
Best Test MSE: 0.649194
Best Test RMSE: 0.805726

Single-sample prediction:
Predicted house price: $204316.75
```

Exact numbers vary between runs because training is stochastic.

---

## Build

Compile with the Makefile:

```bash
make
```

To rebuild cleanly:

```bash
make clean
make
```

---

## Run

### Standard example

```bash
./build/train_housing
```

On Windows PowerShell:

```powershell
.\build\train_housing.exe
```

### Config-based experiment runner

```bash
./build/run_experiment
```

Or with an explicit config file:

```bash
./build/run_experiment experiment.cfg
```

On Windows PowerShell:

```powershell
.\build\run_experiment.exe
```

Make sure `data/housing.csv` is present.

---

## Generated artifacts

Depending on configuration, TinyML can generate:

* `best_housing_model.bin` — best validation checkpoint
* `housing_model.bin` — final exported model
* `housing_stats.bin` — saved normalization statistics
* `training_history.csv` — epoch-level training log
* additional experiment-specific output files from config-driven runs

---

## What this project demonstrates

TinyML is useful as both a learning project and a systems project.

It demonstrates practical understanding of:

* low-level tensor handling in C
* memory-aware ML implementation
* gradient-based learning from scratch
* regression model construction
* optimizer design
* normalization and inference consistency
* serialization of trained models
* experiment-oriented training workflows
* configuration-driven reproducibility

---

## Limitations

TinyML is still intentionally small and focused.

It does not yet include:

* convolutional layers
* recurrent models
* dropout
* batch normalization
* classification losses such as cross-entropy
* automatic differentiation
* GPU acceleration
* production-grade numerical optimization

The current goal is correctness, clarity, and a strong foundation for future milestones.

---

## What comes next

Natural next improvements include:

* richer experiment management
* command-line overrides on top of config files
* batch inference utilities
* additional activation functions
* classification support
* automatic differentiation
* more layer types
* better plotting and reporting workflows

TinyML is now at the point where new features can be added on top of a stable base rather than rebuilding the core every time.

---

## Philosophy

TinyML is built on a simple principle:

**understand first, abstract second.**

Every new feature in this project has been added only after the underlying mechanics were understood and implemented manually. That makes the project slower to grow than high-level experiments, but much stronger as a learning and research foundation.

---

## Status

TinyML has moved well beyond a toy example.

It is now a compact, understandable, extensible C-based deep learning framework capable of training, validating, checkpointing, saving, loading, logging training history, running config-based experiments, and making real predictions on structured data.

That is the milestone documented here.

---

## Author

Clifford Odiwuor Ojuka
