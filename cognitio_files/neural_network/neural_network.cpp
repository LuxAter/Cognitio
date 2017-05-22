#include "neural_network.hpp"
#include <vector>
#include <stdarg.h>
#include <random>
#include <string>

std::vector<double> cognitio::CreateRandBias(int size, double mean, double std_dev){
  std::vector<double> bias;
  std::default_random_engine generator(rand());
  std::normal_distribution<double> rand_gen(mean, std_dev);
  for(int i = 0; i < size; i++){
    bias.push_back(rand_gen(generator));
  }
  return(bias);
}

std::vector<std::vector<double>> cognitio::CreateRandWeight(int size, int prev_size, double mean, double std_dev){
  std::vector<std::vector<double>> weight;
  for(int i = 0; i < size; i++){
    weight.push_back(CreateRandBias(prev_size, mean, std_dev));
  }
  return(weight);
}

cognitio::NeuralNetwork::NeuralNetwork(){
  n_layers = 0;
  n_input = 0;
  n_output = 0;
}

cognitio::NeuralNetwork::NeuralNetwork(int layer_count, ...){
  n_layers = layer_count;
  va_list layer_neuron_count;
  va_start(layer_neuron_count, layer_count);
  int last_neuron_count = va_arg(layer_neuron_count, int);
  n_input = last_neuron_count;
  network_layout.push_back(last_neuron_count);
  for(int i = 1; i < layer_count; i++){
    int neuron_count = va_arg(layer_neuron_count, int);
    network_layout.push_back(neuron_count);
    biases.push_back(CreateRandBias(neuron_count, 0.0, 1.0));
    weights.push_back(CreateRandWeight(neuron_count, last_neuron_count, 0.0, 1.0 / sqrt(last_neuron_count)));
    last_neuron_count = neuron_count;
  }
  n_output = last_neuron_count;
  va_end(layer_neuron_count);
}

cognitio::NeuralNetwork::NeuralNetwork(std::vector<int> layout){
  n_layers = layout.size();
  n_input = layout.front();
  n_output = layout.back();
  network_layout = layout;
  for(int i = 1; i < n_layers; i++){
    biases.push_back(CreateRandBias(layout[i], 0.0, 1.0));
    weights.push_back(CreateRandWeight(layout[i], layout[i-1], 0.0, 1.0 / sqrt(layout[i-1])));
  }
}

cognitio::NeuralNetwork::NeuralNetwork(const NeuralNetwork& copy){
  n_layers = copy.n_layers;
  n_input = copy.n_input;
  n_output = copy.n_output;
  network_layout = copy.network_layout;
  biases = copy.biases;
  weights = copy.weights;
  z_values = copy.z_values;
  activation_values = copy.activation_values;
}

cognitio::NeuralNetwork::~NeuralNetwork(){
  network_layout.clear();
  biases.clear();
  weights.clear();
  z_values.clear();
  activation_values.clear();
  n_layers = 0;
  n_input = 0;
  n_output = 0;
}

std::string cognitio::NeuralNetwork::GetVis(bool vertical){
  std::string outstr;
  if(vertical == true){

  } else if (vertical == false){
    int size = std::to_string(n_layers).size() + 1;
    int max = 0;
    for(int i = 0; i < n_layers; i++){
      if(network_layout[i] > max){
        max = network_layout[i];
      }
      outstr += std::string(" ", size - std::to_string(i).size()) + std::to_string(i);
    }
    outstr += "\n";
    for(int i = 0; i < max; i++){
      for(int j = 0; j < n_layers; j++){
        outstr += std::string(" ", size - 1);
        if(network_layout[j] > i){
          outstr += "#";
        }else{
          outstr += " ";
        }
      }
      outstr += "\n";
    }
  }
  return(outstr);
}
