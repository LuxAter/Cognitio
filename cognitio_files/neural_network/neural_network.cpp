#include <pessum.h>
#include <stdarg.h>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include "activation_functions.hpp"
#include "calculation.hpp"
#include "neural_network.hpp"

std::vector<double> cognitio::CreateRandBias(int size, double mean,
                                             double std_dev) {
  std::vector<double> bias;
  std::default_random_engine generator(rand());
  std::normal_distribution<double> rand_gen(mean, std_dev);
  for (int i = 0; i < size; i++) {
    bias.push_back(rand_gen(generator));
  }
  return (bias);
}

std::vector<std::vector<double>> cognitio::CreateRandWeight(int size,
                                                            int prev_size,
                                                            double mean,
                                                            double std_dev) {
  std::vector<std::vector<double>> weight;
  for (int i = 0; i < size; i++) {
    weight.push_back(CreateRandBias(prev_size, mean, std_dev));
  }
  return (weight);
}

cognitio::NeuralNetwork::NeuralNetwork() {
  n_epoch = 0;
  n_layers = 0;
  n_input = 0;
  n_output = 0;
}

cognitio::NeuralNetwork::NeuralNetwork(int layer_count, ...) {
  n_epoch = 0;
  n_layers = layer_count;
  va_list layer_neuron_count;
  va_start(layer_neuron_count, layer_count);
  int last_neuron_count = va_arg(layer_neuron_count, int);
  n_input = last_neuron_count;
  network_layout.push_back(last_neuron_count);
  for (int i = 1; i < layer_count; i++) {
    int neuron_count = va_arg(layer_neuron_count, int);
    network_layout.push_back(neuron_count);
    biases.push_back(CreateRandBias(neuron_count, 0.0, 1.0));
    weights.push_back(CreateRandWeight(neuron_count, last_neuron_count, 0.0,
                                       1.0 / sqrt(last_neuron_count)));
    last_neuron_count = neuron_count;
  }
  n_output = last_neuron_count;
  va_end(layer_neuron_count);
}

cognitio::NeuralNetwork::NeuralNetwork(std::vector<int> layout) {
  n_epoch = 0;
  n_layers = layout.size();
  n_input = layout.front();
  n_output = layout.back();
  network_layout = layout;
  for (int i = 1; i < n_layers; i++) {
    biases.push_back(CreateRandBias(layout[i], 0.0, 1.0));
    weights.push_back(CreateRandWeight(layout[i], layout[i - 1], 0.0,
                                       1.0 / sqrt(layout[i - 1])));
  }
}

cognitio::NeuralNetwork::NeuralNetwork(const NeuralNetwork& copy) {
  n_epoch = 0;
  n_layers = copy.n_layers;
  n_input = copy.n_input;
  n_output = copy.n_output;
  network_layout = copy.network_layout;
  biases = copy.biases;
  weights = copy.weights;
  z_values = copy.z_values;
  activation_values = copy.activation_values;
}

cognitio::NeuralNetwork::~NeuralNetwork() {
  network_layout.clear();
  biases.clear();
  weights.clear();
  z_values.clear();
  activation_values.clear();
  n_layers = 0;
  n_input = 0;
  n_output = 0;
  n_epoch = 0;
}

void cognitio::NeuralNetwork::EnablePessumLogging() {
  pessum_logging = !pessum_logging;
}

void cognitio::NeuralNetwork::GradientDecent(
    std::vector<std::pair<std::vector<double>, std::vector<double>>>
        training_data,
    int epochs, int batch_size, double learning_rate,
    std::vector<std::pair<std::vector<double>, std::vector<double>>>
        test_data) {
  for (int i = 0; i < epochs; i++) {
    std::random_shuffle(training_data.begin(), training_data.end());
    for (int j = 0; j < training_data.size(); j += batch_size) {
      UpdateBatch(
          std::vector<std::pair<std::vector<double>, std::vector<double>>>(
              training_data.begin() + j,
              training_data.begin() + j + batch_size),
          learning_rate);
    }
    if (test_data.size() != 0) {
      double correct = Evaluate(test_data);
      if (pessum_logging == true) {
        double perc = correct / (double)test_data.size();
        pessum::Log(pessum::DATA, "Compleated epoch %i: %f/%i %f",
                    "cognitio::NeuralNetwork::GradientDecent", n_epoch + i,
                    correct, test_data.size(), perc * 100.0);
      }
    } else {
      if (pessum_logging == true) {
        pessum::Log(pessum::DATA, "Compleated epoch %i",
                    "cognitio::NeuralNetwork::GradientDecent", n_epoch + i);
      }
    }
  }
  n_epoch += epochs;
}

void cognitio::NeuralNetwork::UpdateBatch(
    std::vector<std::pair<std::vector<double>, std::vector<double>>> batch,
    double learning_rate) {
  std::vector<std::vector<double>> nabla_bias = biases;
  std::vector<std::vector<std::vector<double>>> nabla_weight = weights;
  for (int i = 0; i < nabla_bias.size(); i++) {
    for (int j = 0; j < nabla_bias[i].size(); j++) {
      nabla_bias[i][j] = 0.0;
      for (int k = 0; k < nabla_weight[i][j].size(); k++) {
        nabla_weight[i][j][k] = 0.0;
      }
    }
  }

  for (int i = 0; i < batch.size(); i++) {
    std::pair<std::vector<std::vector<double>>,
              std::vector<std::vector<std::vector<double>>>>
        delta_nabla = BackwardProp(batch[i].first, batch[i].second);
    for (int j = 0; j < nabla_bias.size(); j++) {
      for (int k = 0; k < nabla_bias[j].size(); k++) {
        nabla_bias[j][k] += delta_nabla.first[j][k];
        for (int l = 0; l < nabla_weight[j][k].size(); l++) {
          nabla_weight[j][k][l] += delta_nabla.second[j][k][l];
        }
      }
    }
  }

  learning_rate /= batch.size();
  for (int i = 0; i < nabla_bias.size(); i++) {
    for (int j = 0; j < nabla_bias[i].size(); j++) {
      biases[i][j] -= learning_rate * nabla_bias[i][j];
      for (int k = 0; k < nabla_weight[i][j].size(); k++) {
        weights[i][j][k] -= learning_rate * nabla_weight[i][j][k];
      }
    }
  }
}

double cognitio::NeuralNetwork::Evaluate(
    std::vector<std::pair<std::vector<double>, std::vector<double>>>
        test_batch) {
  double count = 0;
  for (int i = 0; i < test_batch.size(); i++) {
    std::vector<double> output = ForwardProp(test_batch[i].first);
    double diff = 0.0;
    for (int j = 0; j < output.size() && j < test_batch[i].second.size(); j++) {
      if (test_batch[i].second[j] != 0.0) {
        diff += (1.0 - (fabs(test_batch[i].second[j] - output[j]) /
                        test_batch[i].second[j]));
      } else {
        diff += (1.0 - fabs(output[j]));
      }
    }
    diff /= output.size();
    count += diff;
  }
  return (count);
}

std::vector<double> cognitio::NeuralNetwork::ForwardProp(
    std::vector<double> input) {
  std::vector<double> output;
  if (input.size() != n_input) {
    pessum::Log(pessum::WARNING,
                "Number of inputs provided does not match number of inputs "
                "required for neural network, %i!=%i",
                "cognitio::NeuralNetwork::ForwardProp", input.size(), n_input);
  } else {
    activation_values.push_back(input);
    for (int i = 1; i < n_layers; i++) {
      z_values.push_back(
          CalculateZ(weights[i - 1], biases[i - 1], activation_values[i - 1]));
      activation_values.push_back(Sigmoid(z_values.back()));
      std::string act_str = "";
      for (int i = 0; i < activation_values.back().size(); i++) {
        act_str += std::to_string(activation_values.back()[i]);
        if (i != activation_values.back().size() - 1) {
          act_str += ",";
        }
      }
      pessum::Log(pessum::DEBUG, "%s", "", act_str.c_str());
    }
    if (activation_values.back().size() != n_output) {
      pessum::Log(pessum::WARNING,
                  "Number of outputs produced does not match number of outputs "
                  "expected for neural network, %i!=%i",
                  "cognitio::NeuralNetwork::ForwardProp",
                  activation_values.back().size(), n_output);
    }
    output = activation_values.back();
  }
  return (output);
}

std::pair<std::vector<std::vector<double>>,
          std::vector<std::vector<std::vector<double>>>>
cognitio::NeuralNetwork::BackwardProp(std::vector<double> input,
                                      std::vector<double> expected_output) {
  std::pair<std::vector<std::vector<double>>,
            std::vector<std::vector<std::vector<double>>>>
      nabla;
  nabla.first = biases;
  nabla.second = weights;
  for (int i = 0; i < nabla.first.size(); i++) {
    for (int j = 0; j < nabla.first[i].size(); j++) {
      nabla.first[i][j] = 0;
      for (int k = 0; k < nabla.second[i][j].size(); k++) {
        nabla.second[i][j][k] = 0;
      }
    }
  }

  ForwardProp(input);
  std::vector<double> delta =
      CostDerivative(activation_values.back(), expected_output);
  nabla.first.back() = delta;
  for (int i = 0; i < delta.size(); i++) {
    for (int j = 0; j < activation_values[activation_values.size() - 2].size();
         j++) {
      nabla.second.back()[i][j] =
          delta[i] * activation_values[activation_values.size() - 2][j];
    }
  }

  for (int l = n_layers - 2; l > 0; l--) {
    std::vector<double> temp_delta(z_values[l].size(), 0.0);
    std::vector<double> sp = z_values[l];
    for (int i = 0; i < temp_delta.size(); i++) {
      for (int j = 0; j < delta.size(); j++) {
        temp_delta[i] += delta[j] * weights[l][j][i];
      }
      temp_delta[i] *= sp[i];
    }
    delta = temp_delta;
    nabla.first[l] = delta;
    for (int i = 0; i < delta.size(); i++) {
      for (int j = 0; j < activation_values[l - 1].size(); j++) {
        nabla.second[l][i][j] = delta[i] * activation_values[l - 1][j];
      }
    }
  }
  return (nabla);
}

std::string cognitio::NeuralNetwork::GetVis(bool vertical) {
  std::string outstr;
  if (vertical == true) {
  } else if (vertical == false) {
    int size = std::to_string(n_layers).size() + 1;
    int max = 0;
    for (int i = 0; i < n_layers; i++) {
      if (network_layout[i] > max) {
        max = network_layout[i];
      }
      outstr +=
          std::string(" ", size - std::to_string(i).size()) + std::to_string(i);
    }
    outstr += "\n";
    for (int i = 0; i < max; i++) {
      for (int j = 0; j < n_layers; j++) {
        outstr += std::string(" ", size - 1);
        if (network_layout[j] > i) {
          outstr += "#";
        } else {
          outstr += " ";
        }
      }
      outstr += "\n";
    }
  }
  return (outstr);
}

std::string cognitio::NeuralNetwork::PrintData() {
  std::string output;
  output += "Biases:\n=======\n";
  for (int i = 0; i < biases.size(); i++) {
    output += std::to_string(i + 1) + ": ";
    for (int j = 0; j < biases[i].size(); j++) {
      output += std::to_string(biases[i][j]);
      if (j != biases[i].size() - 1) {
        output += ",";
      }
    }
    output += "\n";
  }
  output += "\nWeights:\n========\n";
  for (int i = 0; i < weights.size(); i++) {
    output += std::to_string(i + 1) + ":\n";
    for (int j = 0; j < weights[i].size(); j++) {
      output += "  " + std::to_string(j) + ": ";
      for (int k = 0; k < weights[i][j].size(); k++) {
        output += std::to_string(weights[i][j][k]);
        if (k != weights[i][j].size() - 1) {
          output += ",";
        }
      }
      output += "\n";
    }
    output += "\n";
  }
  return (output);
}

std::vector<double> cognitio::NeuralNetwork::CostDerivative(
    std::vector<double> output_activation,
    std::vector<double> expected_output) {
  for (int i = 0; i < output_activation.size() && i < expected_output.size();
       i++) {
    output_activation[i] -= expected_output[i];
  }
  return (output_activation);
}
