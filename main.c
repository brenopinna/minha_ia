#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "include/file_handle.h"
#include "include/ia.h"
#include "include/utils.h"
#include "include/defines.h"

#define TRAIN

int main() {
  srand(time(NULL));

  const char *model_name = "models/model4.bin";

  NeuralNetwork nn;

#ifdef TRAIN
  init_network(&nn);

  double training_data[TRAINING_DATA_SIZE][3];
  load_training_data(training_data);


  for (long long int epoch = 0; epoch < EPOCHS; epoch++) {
    shuffle_training_data(training_data);
    double error = 0;
    for (int batch = 0; batch < TRAINING_DATA_SIZE; batch++) {
      double x1 = normalize(training_data[batch][0]), x2 = normalize(training_data[batch][1]), target = training_data[batch][2];
      double output = backpropagation(&nn, x1, x2, target);
      error += fabs(output - target);
    }

    if (!(epoch % 10000)) {
      system("clear");
      print_color("ERRO MEDIO DA EPOCH:", COLOR_BLUE);
      printf(" %.2lf\n", (double)error / TRAINING_DATA_SIZE);
      print_color("PROGRESSO:", COLOR_GREEN);
      printf(" %.2lf%%\n", (double)100 * epoch / EPOCHS);
    }

    learning_rate = INITIAL_LEARNING_RATE / (1 + LEARNING_DECREASE * epoch);
  }
#else
  load_model(&nn, model_name);
  print_model(&nn);
#endif

  print_color("Resultados:\n", COLOR_GREEN);
  for (int i = 0; i <= MAX_NUM_SIZE; i++) {
    for (int j = 0; j <= MAX_NUM_SIZE; j++) {
      double z1, z2, h1, h2;
      double x1 = normalize(i), x2 = normalize(j);
      double output = forward(&nn, x1, x2, &h1, &h2, &z1, &z2);
      printf("%d e %d ==> %.2lf%%\n", i, j, output * 100);
    }
  }

#ifdef TRAIN
  save_model(&nn, model_name);
#endif
}