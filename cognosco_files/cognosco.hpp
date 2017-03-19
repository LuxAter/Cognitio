#ifndef COGNOSCO_COGNOSCO_HPP
#define COGNOSCO_COGNOSCO_HPP
#include <stdarg.h>
#include <vector>
#include "matrix/matrix_header.hpp"
namespace cognosco {
  class Network {
   public:
    Network();
    Network(int layer_count, ...);
    Network(const Network& copy_net);
    ~Network();
    std::vector<double> ForwardProp(std::vector<double> input);

   private:
    int n_layer = 0, n_input = 0, n_output = 0;
    std::vector<matrix<double>> weight_matrix;
    std::vector<matrix<double>> bias_matrix;
  };
}
#endif
