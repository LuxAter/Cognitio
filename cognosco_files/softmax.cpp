#include <math.h>
#include "matrix/matrix_header.hpp"
#include "softmax.hpp"

cognosco::matrix<double> cognosco::Softmax(cognosco::matrix<double> input) {
  double total = 0;
  input.elementOperation(exp);
  total = sum(input);
  input = (1.0 / total) * input;
  return (input);
}
