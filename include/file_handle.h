#pragma once

#include <stdio.h>
#include <string.h>
#include "ia.h"
#include "defines.h"

static void save_model(NeuralNetwork *nn, const char *file_name) {
  FILE *bin = fopen(file_name, "wb");
  if (!bin) {
    print_color("Erro", COLOR_RED);
    printf(" ao abrir o arquivo [%s]. Verifique se ele existe e tente novamente.\n", file_name);
    exit(1);
  }
  fwrite(nn, sizeof(NeuralNetwork), 1, bin);
  fclose(bin);
}

static void load_model(NeuralNetwork *nn, const char *file_name) {
  FILE *bin = fopen(file_name, "rb");
  if (!bin) {
    print_color("Erro", COLOR_RED);
    printf(" ao abrir o arquivo ");
    print_color(file_name, COLOR_GREEN);
    puts(". Verifique se ele existe e tente novamente.");
    exit(1);
  }
  fread(nn, sizeof(NeuralNetwork), 1, bin);
  fclose(bin);
}
