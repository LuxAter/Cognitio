#include <stdarg.h>
#include <vector>
#include "../matrix/matrix_headers.hpp"
#include "network.hpp"

cognosco::Network::Network() {
  n_layer = 0;
  n_input = 0;
  n_output = 0;
}

cognosco::Network::Network(int layer_count, ...) {
  n_layer = layer_count;
  va_list lnc;
  va_start(lnc, layer_count);
  int last_neurons = va_arg(lnc, int);
  n_input = last_neurons;
  for (int i = 1; i < n_layer; i++) {
    int neurons = va_arg(lnc, int);
    Matrix<double> bias_mat(neurons, 1);
    bias_mat.FillRand(0.0, 1.0);
    bias_matrix.push_back(bias_mat);
    Matrix<double> weight_mat(neurons, last_neurons);
    weight_mat.FillRand(0.0, (1.0 / sqrt(last_neurons)));
    weight_matrix.push_back(weight_mat);
    last_neurons = neurons;
  }
  n_output = last_neurons;
  va_end(lnc);
}

cognosco::Network::Network(std::vector<int> layers) {
  n_layer = layers.size();
  if (layers.size() > 0) {
    n_input = layers[0];
    n_output = layers[layers.size() - 1];
    for (int i = 1; i < n_layer; i++) {
      int neurons = layers[i];
      Matrix<double> bias_mat(neurons, 1);
      bias_mat.FillRand(0.0, 1.0);
      bias_matrix.push_back(bias_mat);
      Matrix<double> weight_mat(neurons, layers[i - 1]);
      weight_mat.FillRand(0.0, (1.0 / sqrt(layers[i - 1])));
      weight_matrix.push_back(weight_mat);
    }
  }
}

cognosco::Network::Network(const Network& copy_net) {
  n_layer = copy_net.n_layer;
  n_input = copy_net.n_input;
  n_output = copy_net.n_output;
  weight_matrix = copy_net.weight_matrix;
  bias_matrix = copy_net.bias_matrix;
}

cognosco::Network::~Network() {
  n_layer = 0;
  n_input = 0;
  n_output = 0;
  weight_matrix.clear();
  bias_matrix.clear();
}

std::vector<double> cognosco::Network::ForwardProp(std::vector<double> input) {
  std::vector<double> output;
  if (input.size() != n_input) {
  } else {
    Matrix<double> value_mat(n_input, 1, input);
    for (int i = 1; i < n_layer; i++) {
    }
    output = value_mat.GetVector();
  }
  return (output);
}
