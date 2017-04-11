#ifndef COGNOSCO_NEURON_HPP
#define COGNOSCO_NEURON_HPP
#include "../matrix/matrix_header.hpp"
namespace cognosco {
  matrix<double> Softmax(matrix<double> input);
  matrix<double> Sigmoid(matrix<double> input);
}
#endif
