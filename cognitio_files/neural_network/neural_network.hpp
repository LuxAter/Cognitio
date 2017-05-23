#ifndef COGNITIO_NEURAL_NETWORK_HPP
#define COGNITIO_NEURAL_NETWORK_HPP
#include <string>
#include <vector>

namespace cognitio {

  std::vector<double> CreateRandBias(int size, double mean, double std_dev);
  std::vector<std::vector<double>> CreateRandWeight(int size, int prev_size,
                                                    double mean,
                                                    double std_dev);

  class NeuralNetwork {
   public:
    NeuralNetwork();
    NeuralNetwork(int layer_count, ...);
    NeuralNetwork(std::vector<int> layout);
    NeuralNetwork(const NeuralNetwork& copy);
    ~NeuralNetwork();

    void EnablePessumLogging();

    void GradientDecent(
        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            training_data,
        int epochs, int batch_size, double learning_rate,
        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            test_data = {});

    void UpdateBatch(
        std::vector<std::pair<std::vector<double>, std::vector<double>>> batch,
        double learning_rate);

    double Evaluate(
        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            test_batch);

    std::vector<double> ForwardProp(std::vector<double> input);
    std::pair<std::vector<std::vector<double>>,
              std::vector<std::vector<std::vector<double>>>>
    BackwardProp(std::vector<double> input,
                 std::vector<double> expected_output);

    std::string GetVis(bool vertical);
    std::string PrintData();

   private:
    std::vector<double> CostDerivative(std::vector<double> output_activation,
                                       std::vector<double> expected_output);

    bool pessum_logging = false;
    int n_layers, n_input, n_output, n_epoch;
    std::vector<int> network_layout;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> z_values, activation_values;
  };
}
#endif
