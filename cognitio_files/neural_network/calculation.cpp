#include <vector>
#include <math.h>
#include "calculation.hpp"
#include <pessum.h>

std::vector<double> cognitio::CalculateZ(std::vector<std::vector<double>> weights, std::vector<double> biases, std::vector<double> activations){
  int neuron_count = biases.size();
  std::vector<double> z_values(biases.size(), 0.0);
  if(weights.front().size() != activations.size()){
    pessum::Log(pessum::WARNING, "Number of weights for neuron does not match number of previous layer activation values %i!=%i", "cognitio::CalculateZ", weights.front().size(), activations.size()); 
  }
  for(int i = 0; i < neuron_count; i++){
    for(int j = 0; j < activations.size(); j++){
      z_values[i] += weights[i][j] * activations[j];
    }
    z_values[i] += biases[i];
  }
  return(z_values);
}
