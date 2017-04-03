#include <stdarg.h>
#include <iostream>
#include "cognosco.hpp"
#include "matrix/matrix_header.hpp"
#include "neuron.hpp"

cognosco::Network::Network() {}

cognosco::Network::Network(int layer_count, ...) {
  n_layer = layer_count;
  va_list lnc;
  va_start(lnc, layer_count);
  int last_neurons = va_arg(lnc, int);
  n_input = last_neurons;
  for (int i = 1; i < layer_count; i++) {
    int neurons = va_arg(lnc, int);
    matrix<double> bias_mat(neurons, 1);
    bias_mat.fillRand(0, 1);
    bias_matrix.push_back(bias_mat);
    matrix<double> weight_mat(neurons, last_neurons);
    weight_mat.fillRand(0, (1 / sqrt(last_neurons)));
    weight_matrix.push_back(weight_mat);
    last_neurons = neurons;
  }
  n_output = last_neurons;
  va_end(lnc);
}

cognosco::Network::Network(const Network& copy_net) {
  n_layer = copy_net.n_layer;
  weight_matrix = copy_net.weight_matrix;
  bias_matrix = copy_net.bias_matrix;
}

cognosco::Network::~Network() {
  n_layer = 0;
  weight_matrix.clear();
  bias_matrix.clear();
}

std::vector<double> cognosco::Network::ForwardProp(std::vector<double> input) {
  matrix<double> value_mat(n_input, 1, input);
  for (int i = 1; i < n_layer; i++) {
    value_mat =
        Sigmoid(dot(weight_matrix[i - 1], value_mat) + bias_matrix[i - 1]);
  }
  return (value_mat.getVector());
}
