#include <stdarg.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "../matrix/matrix_headers.hpp"
#include "network.hpp"
#include "neuron.hpp"

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
  layer_layout.push_back(last_neurons);
  bias_matrix.push_back(Matrix<double>());
  weight_matrix.push_back(Matrix<double>());
  for (int i = 1; i < n_layer; i++) {
    int neurons = va_arg(lnc, int);
    layer_layout.push_back(neurons);
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
  layer_layout = layers;
  n_layer = layers.size();
  if (layers.size() > 0) {
    n_input = layers[0];
    n_output = layers[layers.size() - 1];
    bias_matrix.push_back(Matrix<double>());
    weight_matrix.push_back(Matrix<double>());
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
    pessum::Log(
        pessum::WARNING,
        "Number of input values %i does not equal number of input neurons %i",
        "ForwardProp", input.size(), n_input);
  } else {
    Matrix<double> value_mat(n_input, 1, input);
    layer_z.clear();
    layer_a.clear();
    layer_z.push_back(value_mat);
    Sigmoid(value_mat);
    layer_a.push_back(value_mat);
    for (int i = 2; i <= n_layer; i++) {
      value_mat = Dot(weight_matrix[i - 1], value_mat) + bias_matrix[i - 1];
      layer_z.push_back(value_mat);
      Sigmoid(value_mat);
      layer_a.push_back(value_mat);
    }
    output = value_mat.GetVector();
  }
  return (output);
}

std::pair<std::vector<cognosco::Matrix<double>>,
          std::vector<cognosco::Matrix<double>>>
cognosco::Network::BackwardProp(std::vector<double> input,
                                std::vector<double> expected) {
  std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>> partials;
  Matrix<double> output(n_output, 1, ForwardProp(input));
  Matrix<double> expected_out(n_output, 1, expected);
  Matrix<double> delta(n_output, 1);
  delta = (output - expected_out) * SigmoidPrime(layer_z[layer_z.size() - 1]);
  partials.first.push_back(delta);
  partials.second.push_back(
      Transpose(Dot(layer_a[layer_a.size() - 2], Transpose(delta))));
  // printf("%ix%i\n", partials.second[0].GetShape().first,
  //        partials.second[0].GetShape().second);
  for (int i = n_layer - 2; i > 0; i--) {
    // printf("δ^%i = ((W^%i)^T * δ^%i) ʘ σ\'(z^%i)\n", i, i + 1, i + 1, i);
    // printf("%i:  w+1:%ix%i d:%ix%i z:%ix%i, a-1:%ix%i\n", i,
    //        weight_matrix[i + 1].GetShape().first,
    //        weight_matrix[i + 1].GetShape().second, delta.GetShape().first,
    //        delta.GetShape().second, layer_z[i].GetShape().first,
    //        layer_z[i].GetShape().second, layer_a[i - 1].GetShape().first,
    //        layer_a[i - 1].GetShape().second);
    delta = (Dot(Transpose(weight_matrix[i + 1]), delta) *
             SigmoidPrime(layer_z[i]));
    partials.first.insert(partials.first.begin(), delta);
    partials.second.insert(partials.second.begin(),
                           Transpose(Dot(layer_a[i - 1], Transpose(delta))));
  }
  partials.first.insert(partials.first.begin(), Matrix<double>());
  partials.second.insert(partials.second.begin(), Matrix<double>());
  for (int i = 0; i < partials.second.size(); i++) {
    // printf("%i:%ix%i\n", i, partials.second[i].GetShape().first,
    //        partials.second[i].GetShape().second);
  }
  return (partials);
}

void cognosco::Network::StochasticGradientDescent(
    std::vector<std::pair<std::vector<double>, std::vector<double>>>
        training_data,
    int epochs, int batch_size) {
  for (int i = 0; i < epochs; i++) {
    epoch++;
    std::random_shuffle(training_data.begin(), training_data.end());
    std::vector<std::pair<std::vector<double>, std::vector<double>>>
        training_batch;
    for (int j = 0; j < training_data.size(); j++) {
      training_batch.push_back(training_data[j]);
      if (training_batch.size() == batch_size) {
        std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>>
            delta =
                BackwardProp(training_batch[0].first, training_batch[0].second);
        for (int k = 1; k < training_batch.size(); k++) {
          std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>>
              delta_delta = BackwardProp(training_batch[k].first,
                                         training_batch[k].second);
          for (int l = 0; l < delta.first.size(); l++) {
            delta.first[l] = delta.first[l] + delta_delta.first[l];
          }
          for (int l = 0; l < delta.second.size(); l++) {
            delta.second[l] = delta.second[l] + delta_delta.second[l];
          }
        }
        for (int j = 0; j < delta.first.size(); j++) {
          delta.first[j] = (1.0f / (double)batch_size) * delta.first[j];
          // printf("bias matrix %i: %ix%i ->", j,
          // bias_matrix[j].GetShape().first,
          //        bias_matrix[j].GetShape().second);
          bias_matrix[j] = bias_matrix[j] + delta.first[j];
          // printf(" %ix%i\n", bias_matrix[j].GetShape().first,
          //        bias_matrix[j].GetShape().second);
        }
        for (int j = 0; j < delta.second.size(); j++) {
          delta.second[j] = (1.0f / (double)batch_size) * delta.second[j];
          // printf("weight matrix %i: %ix%i ->", j,
          //        weight_matrix[j].GetShape().first,
          //        weight_matrix[j].GetShape().second);
          weight_matrix[j] = weight_matrix[j] + delta.second[j];
          // printf(" %ix%i\n", weight_matrix[j].GetShape().first,
          //        weight_matrix[j].GetShape().second);
        }
        training_batch.clear();
        // printf("------------------------------------------------------\n");
      }
    }
  }
}

std::string cognosco::Network::GetString() {
  std::string str = "";
  int length = std::to_string(n_layer).size();
  for (int i = 0; i < n_layer; i++) {
    std::string layer_str = "";
    for (int j = 0; j < layer_layout[i]; j++) {
      layer_str += " X";
    }
    ssize_t buff_size =
        snprintf(NULL, 0, "%*i|%s\n", length, i, layer_str.c_str());
    char* formated_string = new char[buff_size];
    sprintf(formated_string, "%*i|%s\n", n_layer, i, layer_str.c_str());
    str += std::string(formated_string);
  }
  return (str);
}
