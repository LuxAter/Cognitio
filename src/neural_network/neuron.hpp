#ifndef COGNOSCO_NEURON_HPP
#define COGNOSCO_NEURON_HPP
#include "../matrix/matrix_headers.hpp"
namespace cognosco {
  template <class _T>
  Matrix<_T> Sigmoid(Matrix<_T> input) {
    for (int i = 0; i < input.n_row; i++) {
      input.terms[i][0] = 1.0 / (1 + exp(-input.terms[i][0]));
    }
    return (input);
  }

  template <class _T>
  Matrix<_T> SigmoidPrime(Matrix<_T> input) {
    for (int i = 0; i < input.n_row; i++) {
      input.terms[i][0] =
          exp(input.terms[i][0]) / pow(exp(input.terms[i][0]) + 1, 2);
    }
    return (input);
  }

  template <class _T>
  Matrix<_T> Softmax(Matrix<_T> input) {
    _T total = _T();
    input.ElementOperation(exp);
    total = Sum(input);
    input = (input / total);
    return (input);
  }
}
#endif
