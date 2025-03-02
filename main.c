#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "include/file_handle.h"
#include "include/ia.h"
#include "include/utils.h"
#include "include/defines.h"

// #define TRAIN

int main() {
  srand(time(NULL));

  const char *model_name = "models/model2.bin";

  NeuralNetwork nn;

#ifdef TRAIN
  init_network(&nn);

  double training_data[TRAINING_DATA_SIZE][3];
  load_training_data(training_data);

  for (long long int i = 0; i < EPOCHS; i++) {
    for (int i = 0; i < TRAINING_DATA_SIZE; i++) {
      double x1 = training_data[i][0], x2 = training_data[i][1], target = training_data[i][2];
      backpropagation(&nn, x1, x2, target);
    }
  }
#else
  load_model(&nn, model_name);
  print_model(&nn);
#endif

  print_color("Resultados:\n", COLOR_GREEN);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 6; j++) {
      double h1, h2;
      double output = forward(&nn, i, j, &h1, &h2);
      printf("%d e %d ==> %.2lf%%\n", i, j, output * 100);
    }
  }

#ifdef TRAIN
  save_model(&nn, model_name);
#endif
}