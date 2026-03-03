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

# Files
SRC = $(SRC_DIR)/tensor.c
OBJ = $(BUILD_DIR)/tensor.o

LIB = $(BUILD_DIR)/libtinyml.a

EXAMPLE = $(BUILD_DIR)/test_tensor

# Default target
all: $(LIB) example

# Create build folder
$(BUILD_DIR):
	if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)

# Compile object files
$(BUILD_DIR)/tensor.o: $(SRC) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(SRC) -o $@

# Create static library
$(LIB): $(OBJ)
	ar rcs $(LIB) $(OBJ)

# Build example program
example: $(LIB)
	$(CC) $(CFLAGS) $(EXAMPLE_DIR)/test_tensor.c -L$(BUILD_DIR) -ltinyml $(LDFLAGS) -o $(EXAMPLE)

# Run example
run: example
	.\$(EXAMPLE)

# Clean build
clean:
	if exist $(BUILD_DIR) rmdir /s /q $(BUILD_DIR)