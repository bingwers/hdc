
SOURCE_DIR = src
INCLUDE_DIR = include
OUTPUT_DIR = obj
BIN_DIR = bin
CC = gcc
CFLAGS = -I$(INCLUDE_DIR) -O4 -fPIC

LIBS = -lpthread -lm

_DEPS = model.h mnist.h hypervector.h
DEPS =  $(patsubst %,$(INCLUDE_DIR)/%,$(_DEPS))

_OBJ = model.o mnist.o hypervector.o
OBJ = $(patsubst %,$(OUTPUT_DIR)/%,$(_OBJ))

all: $(BIN_DIR)/libmodel.so $(BIN_DIR)/imageManip

$(OUTPUT_DIR)/%.o : $(SOURCE_DIR)/%.c $(DEPS)
	mkdir -p $(OUTPUT_DIR) && $(CC) -c -o $@ $< $(CFLAGS)

$(BIN_DIR)/libmodel.so : $(OBJ)
	mkdir -p $(BIN_DIR) && $(CC) -shared -o $@ $^

$(BIN_DIR)/imageManip : $(OUTPUT_DIR)/imageManip.o $(OUTPUT_DIR)/mnist.o
	mkdir -p $(BIN_DIR) && $(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(OUTPUT_DIR)/* && rm -f $(BIN_DIR)/*

