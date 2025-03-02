#pragma once

#include "utils.h"
#include "defines.h"

static void init_neuron(Neuron *neuron) {
  neuron->weights[0] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  neuron->weights[1] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  neuron->bias = (double)rand() / RAND_MAX * 2.0 - 1.0;
}

static double neuron_output(Neuron *neuron, double x1, double x2) {
  double sum = neuron->weights[0] * x1 + neuron->weights[1] * x2 + neuron->bias;
  return sigmoid(sum);
}

static void init_network(NeuralNetwork *nn) {
  init_neuron(&nn->hidden[0]);
  init_neuron(&nn->hidden[1]);
  init_neuron(&nn->output);
}

static double forward(NeuralNetwork *nn, double x1, double x2, double *h1, double *h2) {
  *h1 = neuron_output(&nn->hidden[0], x1, x2);
  *h2 = neuron_output(&nn->hidden[1], x1, x2);
  return neuron_output(&nn->output, *h1, *h2);
}

static void backpropagation(NeuralNetwork *nn, double x1, double x2, double target) {
  double h1, h2, output;

  output = forward(nn, x1, x2, &h1, &h2);

  double delta_output = (output - target) * sigmoid_derivative(output);
  double delta_h1 = delta_output * nn->output.weights[0] * sigmoid_derivative(h1);
  double delta_h2 = delta_output * nn->output.weights[1] * sigmoid_derivative(h2);

  nn->output.weights[0] -= LEARNING_RATE * delta_output * h1;
  nn->output.weights[1] -= LEARNING_RATE * delta_output * h2;
  nn->output.bias -= LEARNING_RATE * delta_output;

  nn->hidden[0].weights[0] -= LEARNING_RATE * delta_h1 * x1;
  nn->hidden[0].weights[1] -= LEARNING_RATE * delta_h1 * x2;
  nn->hidden[0].bias -= LEARNING_RATE * delta_h1;

  nn->hidden[1].weights[0] -= LEARNING_RATE * delta_h2 * x1;
  nn->hidden[1].weights[1] -= LEARNING_RATE * delta_h2 * x2;
  nn->hidden[1].bias -= LEARNING_RATE * delta_h2;
}