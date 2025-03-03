#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "include/file_handle.h"
#include "include/ia.h"
#include "include/utils.h"
#include "include/defines.h"

// Se essa variavel estiver definida, seu modelo ser CRIADO ou SUBSTITUIDO por outro treinado.
// para apenas usar um modelo sem treinar, comente-a
// #define TRAIN

int main() {
  srand(time(NULL));

  const char *model_name = "models/99.96.bin";

  NeuralNetwork nn;

#ifdef TRAIN
  init_network(&nn);

  double training_data[TRAINING_DATA_SIZE][3];
  load_training_data(training_data);

  for (long long int epoch = 0; epoch < EPOCHS; epoch++) {
    shuffle_training_data(training_data);
    for (int batch = 0; batch < TRAINING_DATA_SIZE; batch += BATCH_SIZE) {
      double grad_w11 = 0, grad_w12 = 0, grad_b1 = 0,
        grad_w21 = 0, grad_22 = 0, grad_b2 = 0,
        grad_wo1 = 0, grad_wo2 = 0, grad_bo = 0;

      for (int mini_batch = batch; mini_batch < batch + BATCH_SIZE && mini_batch < TRAINING_DATA_SIZE; mini_batch++) {
        double x1 = normalize(training_data[mini_batch][0]), x2 = normalize(training_data[mini_batch][1]), target = training_data[mini_batch][2];
        double output = backpropagation(&nn, x1, x2, target,
                                        &grad_w11, &grad_w12, &grad_b1,
                                        &grad_w21, &grad_22, &grad_b2,
                                        &grad_wo1, &grad_wo2, &grad_bo);
      }

      nn.output.weights[0] -= learning_rate * (grad_wo1 / BATCH_SIZE);
      nn.output.weights[1] -= learning_rate * (grad_wo2 / BATCH_SIZE);
      nn.output.bias -= learning_rate * (grad_bo / BATCH_SIZE);

      nn.hidden[0].weights[0] -= learning_rate * (grad_w11 / BATCH_SIZE);
      nn.hidden[0].weights[1] -= learning_rate * (grad_w12 / BATCH_SIZE);
      nn.hidden[0].bias -= learning_rate * (grad_b1 / BATCH_SIZE);

      nn.hidden[1].weights[0] -= learning_rate * (grad_w21 / BATCH_SIZE);
      nn.hidden[1].weights[1] -= learning_rate * (grad_22 / BATCH_SIZE);
      nn.hidden[1].bias -= learning_rate * (grad_b2 / BATCH_SIZE);
    }

    if (!(epoch % 1000)) {
      system("clear");
      print_color("PROGRESSO:", COLOR_GREEN);
      printf(" %.2lf%%\n", (double)100 * epoch / EPOCHS);
    }

    learning_rate = INITIAL_LEARNING_RATE / (1 + LEARNING_DECREASE * epoch);
  }

#else
  load_model(&nn, model_name);
  print_model(&nn);
#endif

  FILE *depois = fopen("resultados.txt", "w");

  print_color("Resultados:\n", COLOR_GREEN);
  for (int i = 0; i <= MAX_NUM_SIZE; i++) {
    for (int j = 0; j <= MAX_NUM_SIZE; j++) {
      double z1, z2, h1, h2;
      double x1 = normalize(i), x2 = normalize(j);
      double output = forward(&nn, x1, x2, &h1, &h2, &z1, &z2);
      fprintf(depois, "%3d e %3d ==> %.2lf%%\n", i, j, output * 100);
    }
  }

  fclose(depois);

#ifdef TRAIN
  save_model(&nn, model_name);
#endif
}