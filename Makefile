# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Iinclude -g -Wall -Wextra

# Linker flags
LDFLAGS = -lm

# Directories
SRC_DIR = src
EXAMPLE_DIR = examples
BUILD_DIR = build

# Source files
SRC = $(SRC_DIR)/tensor.c $(SRC_DIR)/layers.c $(SRC_DIR)/network.c $(SRC_DIR)/optimizers.c $(SRC_DIR)/dataset.c $(SRC_DIR)/train.c $(SRC_DIR)/config.c

# Object files
OBJ = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/layers.o $(BUILD_DIR)/network.o $(BUILD_DIR)/optimizers.o $(BUILD_DIR)/dataset.o $(BUILD_DIR)/train.o $(BUILD_DIR)/config.o

# Static library
LIB = $(BUILD_DIR)/libtinyml.a

# Example executable
EXAMPLE = $(BUILD_DIR)/run_experiment

# Default target
all: $(LIB) example

# Create build directory (Windows safe)
$(BUILD_DIR):
	if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

# Compile tensor object
$(BUILD_DIR)/tensor.o: $(SRC_DIR)/tensor.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/tensor.c -o $(BUILD_DIR)/tensor.o

# Compile layers object
$(BUILD_DIR)/layers.o: $(SRC_DIR)/layers.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/layers.c -o $(BUILD_DIR)/layers.o

# Compile network object
$(BUILD_DIR)/network.o: $(SRC_DIR)/network.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/network.c -o $(BUILD_DIR)/network.o

# Compile optimizer object
$(BUILD_DIR)/optimizers.o: $(SRC_DIR)/optimizers.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/optimizers.c -o $(BUILD_DIR)/optimizers.o

# Compile dataset object
$(BUILD_DIR)/dataset.o: $(SRC_DIR)/dataset.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/dataset.c -o $(BUILD_DIR)/dataset.o

# Compile train object
$(BUILD_DIR)/train.o: $(SRC_DIR)/train.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/train.c -o $(BUILD_DIR)/train.o

# Compile config object
$(BUILD_DIR)/config.o: $(SRC_DIR)/config.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC_DIR)/config.c -o $(BUILD_DIR)/config.o

# Create static library
$(LIB): $(OBJ)
	ar rcs $(LIB) $(OBJ)

# Build example program
example: $(LIB)
	$(CC) $(CFLAGS) $(EXAMPLE_DIR)/run_experiment.c -L$(BUILD_DIR) -ltinyml $(LDFLAGS) -o $(EXAMPLE)

# Run example
run: example
	.\$(EXAMPLE)

# Clean build
clean:
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)