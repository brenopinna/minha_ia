#pragma once

#include <math.h>
#include "utils.h"
#include "defines.h"

typedef enum {
  XAVIER,
  HE
} InitializeFunction;

static double he_init(int inputs) {
  double limit = sqrt(2.0 / inputs);
  return ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit;
}

static double xavier_init(int inputs) {
  double limit = sqrt(1.0 / inputs);
  return ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit;
}

static void init_neuron(Neuron *neuron, InitializeFunction I) {
  neuron->weights[0] = I == XAVIER ? xavier_init(2) : he_init(2);  // 2 entradas
  neuron->weights[1] = I == XAVIER ? xavier_init(2) : he_init(2);
  neuron->bias = 0;  // Bias pode começar como zero
}

static double neuron_output(Neuron *neuron, double x1, double x2) {
  double sum = neuron->weights[0] * x1 + neuron->weights[1] * x2 + neuron->bias;
  return (sum);
}

static void init_network(NeuralNetwork *nn) {
  init_neuron(&nn->hidden[0], HE);
  init_neuron(&nn->hidden[1], HE);
  init_neuron(&nn->output, XAVIER);
}

static double forward(NeuralNetwork *nn, double x1, double x2, double *h1, double *h2, double *z1, double *z2) {
  *z1 = neuron_output(&nn->hidden[0], x1, x2);
  *z2 = neuron_output(&nn->hidden[1], x1, x2);
  *h1 = ReLU(*z1);
  *h2 = ReLU(*z2);
  return sigmoid(neuron_output(&nn->output, *h1, *h2));
}

static double backpropagation(NeuralNetwork *nn, double x1, double x2, double target,
                              double *grad_w11, double *grad_w12, double *grad_b1,
                              double *grad_w21, double *grad_22, double *grad_b2,
                              double *grad_wo1, double *grad_wo2, double *grad_bo
) {
  double z1, z2, h1, h2, output;

  output = forward(nn, x1, x2, &h1, &h2, &z1, &z2);

  double weight = (target == 1) ? 2.0 : 1.0;

  double delta_output = weight * (output - target) * sigmoid_derivative(output);
  double delta_h1 = delta_output * nn->output.weights[0] * ReLU_derivative(z1);
  double delta_h2 = delta_output * nn->output.weights[1] * ReLU_derivative(z2);

  *grad_w11 += delta_h1 * x1; *grad_w12 += delta_h1 * x2; *grad_b1 += delta_h1;
  *grad_w21 += delta_h2 * x1; *grad_22 += delta_h2 * x2; *grad_b2 += delta_h2;
  *grad_wo1 += delta_output * h1; *grad_wo2 += delta_output * h2; *grad_bo += delta_output;

  return output;
}