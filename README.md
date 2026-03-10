# TinyML

A tiny deep learning framework built from scratch in pure C.

TinyML is a low-level machine learning project focused on understanding how neural networks work under the hood by implementing the core pieces directly in C. Instead of depending on high-level ML libraries, this project builds the training pipeline manually: tensors, dense layers, forward and backward propagation, optimization, normalization, checkpointing, and inference.

At its current milestone, TinyML can train a multi-layer neural network on the California Housing dataset, save and reload the best model, preserve normalization statistics, and make single-sample predictions in real house-price units.

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

This means the framework now supports a full practical workflow:

```text id="m0mcmn"
load data -> split -> normalize -> train -> validate -> save -> load -> predict
```

---

## Project structure

```text id="0r6uid"
TinyML/
├── data/
│   └── housing.csv
├── include/
│   └── tinyml.h
├── src/
│   ├── tensor.c
│   ├── layer.c
│   ├── network.c
│   ├── optimizer.c
│   ├── dataset.c
│   └── train.c
├── examples/
│   └── train_housing.c
├── build/
├── Makefile
└── README.md
```

The public API is exposed through a single header file, `tinyml.h`, while the implementation is split across `src/`.

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

It began as a toy neural network trained on a tiny manually defined dataset to validate the basics of feedforward computation and gradient descent. From there, the framework evolved step by step:

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

The latest milestone completed the practical pipeline:

* save normalization statistics
* load normalization statistics
* normalize a raw input sample
* run prediction
* denormalize the output back to a real house price

That is where TinyML currently stands.

---

## Current example architecture

The strongest current housing example uses:

```text id="qjpw9l"
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

The `examples/train_housing.c` example currently demonstrates how to:

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

---

## Example output

A typical run looks like this:

```text id="ajab3j"
Epoch 0  | Train Loss 0.890418 | Val MSE 0.784094 | Val RMSE 0.885491
Epoch 1  | Train Loss 0.527127 | Val MSE 0.640162 | Val RMSE 0.800101
...
Epoch 18 | Train Loss 0.294092 | Val MSE 0.470942 | Val RMSE 0.686252

Loading best checkpoint...
Model loaded from best_housing_model.bin

Evaluating best model...
Best Test MSE: 0.470942
Best Test RMSE: 0.686252

Single-sample prediction:
Predicted house price: $286774.54
```

Exact numbers vary between runs because training is stochastic.

---

## Build

Compile with your existing Makefile:

```bash id="4x8h4y"
make
```

To rebuild cleanly:

```bash id="lgl9pr"
make clean
make
```

---

## Run

```bash id="lzpjdy"
./build/train_housing
```

On Windows PowerShell:

```powershell id="g7xjng"
.\build\train_housing.exe
```

Make sure `data/housing.csv` is present.

---

## Generated artifacts

Depending on configuration, TinyML can generate:

* `best_housing_model.bin` — best validation checkpoint
* `housing_model.bin` — final exported model
* `housing_stats.bin` — saved normalization statistics
* `training_history.csv` — epoch-level training log, if enabled

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
* config-driven architecture definitions
* production-grade numerical optimization

The current goal is correctness, clarity, and a strong foundation for future milestones.

---

## What comes next

Natural next improvements include:

* training history logging to CSV for plotting and experiment tracking
* command-line or config-based experiment control
* inference utilities for raw user input
* additional activation functions
* classification support
* automatic differentiation
* more layer types

TinyML is now at the point where new features can be added on top of a stable base rather than rebuilding the core every time.

---

## Philosophy

TinyML is built on a simple principle:

**understand first, abstract second.**

Every new feature in this project has been added only after the underlying mechanics were understood and implemented manually. That makes the project slower to grow than high-level experiments, but much stronger as a learning and research foundation.

---

## Status

TinyML has moved beyond a toy example.

It is now a compact, understandable, extensible C-based deep learning framework capable of training, validating, checkpointing, saving, loading, and making real predictions on structured data.

That is the milestone documented here.

---

## Author

Clifford Odiwuor Ojuka

---