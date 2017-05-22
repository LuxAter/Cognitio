#ifndef COGNITIO_NEURAL_NETWORK_CALCULATION_HPP
#define COGNITIO_NEURAL_NETWORK_CALCULATION_HPP
#include <vector>
namespace cognitio{
  std::vector<double> CalculateZ(std::vector<std::vector<double>> weights, std::vector<double> biases, std::vector<double> activations);
}
#endif
