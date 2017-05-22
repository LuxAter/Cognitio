#ifndef COGNITIO_NEURAL_NETWORK_HPP
#define COGNITIO_NEURAL_NETWORK_HPP
#include <vector>
#include <string>
namespace cognitio{

  std::vector<double> CreateRandBias(int size, double mean, double std_dev);
  std::vector<std::vector<double>> CreateRandWeight(int size, int prev_size, double mean, double std_dev);

  class NeuralNetwork{
    public:
      NeuralNetwork();
      NeuralNetwork(int layer_count, ...);
      NeuralNetwork(std::vector<int> layout);
      NeuralNetwork(const NeuralNetwork& copy);
      ~NeuralNetwork();
      std::string GetVis(bool vertical);
    private:
      int n_layers, n_input, n_output;
      std::vector<int> network_layout;
      std::vector<std::vector<double>> biases;
      std::vector<std::vector<std::vector<double>>> weights;
      std::vector<std::vector<double>> z_values, activation_values;
  };
}
#endif
