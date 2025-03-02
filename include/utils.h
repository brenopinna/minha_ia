#pragma once

#include <math.h>
#include <stdio.h>
#include "ia.h"
#include "defines.h"

static const char *get_color_code(Color color) {
  switch (color) {
    case COLOR_RED:    return "\e[91m"; // Caso nao fique certo a cor, pesquise como funciona cor em ANSI pro seu OS
    case COLOR_GREEN:  return "\e[32m";
    case COLOR_YELLOW: return "\e[33m";
    case COLOR_BLUE:   return "\e[34m";
    case COLOR_RESET:
    default:           return "\e[0m";
  }
}

static void print_color(const char *text, Color color) {
  printf("%s%s\033[0m", get_color_code(color), text);
}

static double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_derivative(double x) {
  return x * (1 - x);
}

static double ReLU(double x) {
  if (x > 0) return x;
  return 0;
}

static double ReLU_derivative(double x) {
  if (x > 0) return 1;
  return 0;
}

static void print_model(NeuralNetwork *nn) {
  double w11 = nn->hidden[0].weights[0];
  double w12 = nn->hidden[0].weights[1];
  double b1 = nn->hidden[0].bias;
  double w21 = nn->hidden[1].weights[0];
  double w22 = nn->hidden[1].weights[1];
  double b2 = nn->hidden[1].bias;
  double w01 = nn->output.weights[0];
  double w02 = nn->output.weights[1];
  double b0 = nn->output.bias;
  print_color("Pesos e Biases do Modelo Salvo:\n", COLOR_YELLOW);
  printf("W11: %lg, ", w11);
  printf("W12: %lg, ", w12);
  printf("B1: %lg\n", b1);
  printf("W21: %lg, ", w21);
  printf("W22: %lg, ", w22);
  printf("B2: %lg\n", b2);
  printf("W01: %lg, ", w01);
  printf("W02: %lg, ", w02);
  printf("B0: %lg\n", b0);
  puts("-------------------------------");
}

static void load_training_data(double training_data[TRAINING_DATA_SIZE][3]) {
  int filled = 0;

  for (int x1 = 0; x1 <= MAX_NUM_SIZE; x1++) {
    for (int x2 = 0; x2 <= MAX_NUM_SIZE; x2++) {
      training_data[filled][0] = x1;
      training_data[filled][1] = x2;
      training_data[filled++][2] = x1 == x2 ? 1 : 0;
    }
  }
}

static void shuffle_training_data(double training_data[TRAINING_DATA_SIZE][3]) {
  for (int i = TRAINING_DATA_SIZE - 1; i > 0; i--) {
    int j = rand() % (i + 1);  // Gera um índice aleatório entre 0 e i

    // Troca cada uma das 3 colunas entre training_data[i] e training_data[j]
    for (int k = 0; k < 3; k++) {
      double temp = training_data[i][k];
      training_data[i][k] = training_data[j][k];
      training_data[j][k] = temp;
    }
  }
}

static double normalize(double x) {
  return x / MAX_NUM_SIZE;
}
