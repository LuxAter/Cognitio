#ifndef COGNOSCO_NETWORK_HPP
#define COGNOSCO_NETWORK_HPP
#include <vector>
#include "../matrix/matrix.hpp"
namespace cognosco {
  class Network {
   public:
    Network();
    Network(int layer_count, ...);
    Network(std::vector<int> layers);
    Network(const Network& copy_net);
    ~Network();

   private:
    int n_layers, n_input, n_output;
    std::vector<Matrix<double>> weight_matrix, bias_matrix;
  };
}
#endif
