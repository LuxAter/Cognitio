#ifndef COGNOSCO_NETWORK_HPP
#define COGNOSCO_NETWORK_HPP
#include <vector>
#include "../matrix/matrix_headers.hpp"
namespace cognosco {
  class Network {
   public:
    Network();
    Network(int layer_count, ...);
    Network(std::vector<int> layers);
    Network(const Network& copy_net);
    ~Network();

    std::vector<double> ForwardProp(std::vector<double> input);
    std::pair<std::vector<Matrix<double>>, std::vector<Matrix<double>>>
    BackwardProp(std::vector<double> input, std::vector<double> expected);

    void StochasticGradientDescent(
        std::vector<std::pair<std::vector<double>, std::vector<double>>>
            training_data,
        int epochs, int batch_size);

    std::string GetString();

   private:
    int epoch = 0;
    double learning_rate = 3.0;
    int n_layer, n_input, n_output;
    std::vector<Matrix<double>> weight_matrix, bias_matrix;
    std::vector<int> layer_layout;
    std::vector<Matrix<double>> layer_z, layer_a;
  };
}
#endif
