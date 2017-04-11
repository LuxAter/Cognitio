#include <math.h>
#include "../matrix/matrix_header.hpp"
#include "neuron.hpp"

cognosco::matrix<double> cognosco::Softmax(cognosco::matrix<double> input) {
  double total = 0;
  input.elementOperation(exp);
  total = sum(input);
  input = (1.0 / total) * input;
  return (input);
}

cognosco::matrix<double> cognosco::Sigmoid(cognosco::matrix<double> input){
  for(int i = 0; i < input.n_row; i++){
    input.matrix_data[i][0] = 1.0 / (1 + exp(-input.matrix_data[i][0]));
  }
  return(input);
}
