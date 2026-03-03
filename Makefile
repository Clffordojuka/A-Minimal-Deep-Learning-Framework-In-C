# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Iinclude -O2 -Wall -Wextra

# Linker flags
LDFLAGS = -lm

# Directories
SRC_DIR = src
EXAMPLE_DIR = examples
BUILD_DIR = build

# Source files
SRC = $(SRC_DIR)/tensor.c $(SRC_DIR)/layers.c

# Object files
OBJ = $(BUILD_DIR)/tensor.o $(BUILD_DIR)/layers.o

# Static library
LIB = $(BUILD_DIR)/libtinyml.a

# Example executable
EXAMPLE = $(BUILD_DIR)/test_layer

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

# Create static library
$(LIB): $(OBJ)
	ar rcs $(LIB) $(OBJ)

# Build example program
example: $(LIB)
	$(CC) $(CFLAGS) $(EXAMPLE_DIR)/test_layer.c -L$(BUILD_DIR) -ltinyml $(LDFLAGS) -o $(EXAMPLE)

# Run example
run: example
	.\$(EXAMPLE)

# Clean build
clean:
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)