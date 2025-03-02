#pragma once

#define LEARNING_RATE 1
#define EPOCHS 1000
#define MAX_NUM_SIZE 6
#define TRAINING_DATA_SIZE MAX_NUM_SIZE * MAX_NUM_SIZE

typedef enum {
  COLOR_RED,
  COLOR_GREEN,
  COLOR_YELLOW,
  COLOR_BLUE,
  COLOR_RESET
} Color;

typedef struct {
  double weights[2];
  double bias;
} Neuron;

typedef struct {
  Neuron hidden[2];
  Neuron output;
} NeuralNetwork;