#pragma once

#define INITIAL_LEARNING_RATE 0.8
// ESSA LEARNING_DECREASE EH UM TESTE
#define LEARNING_DECREASE 0.000
#define EPOCHS 1000000
#define MAX_NUM_SIZE 5
#define TRAINING_DATA_SIZE ((MAX_NUM_SIZE + 1) * (MAX_NUM_SIZE + 1))

double learning_rate = INITIAL_LEARNING_RATE;

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