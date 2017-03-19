#ifndef COGNOSCO_SOFTMAX_HPP
#define COGNOSCO_SOFTMAX_HPP
#include "matrix/matrix_header.hpp"
namespace cognosco {
  matrix<double> Softmax(matrix<double> input);
  matrix<double> SoftmaxPrime(matrix<double> input);
}
#endif
