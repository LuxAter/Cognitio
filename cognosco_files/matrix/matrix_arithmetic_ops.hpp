#ifndef COGNOSCO_MATRIX_ARITHMETIC_OPS_HPP
#define COGNOSCO_MATRIX_ARITHMETIC_OPS_HPP
#include <algorithm>
#include <limits>
#include "matrix.hpp"
namespace cognosco {
  template <class _TYPE>
  matrix<_TYPE> operator+(const matrix<_TYPE>& mat_a,
                          const matrix<_TYPE>& mat_b) {
    matrix<_TYPE> sum(std::max(mat_a.n_row, mat_b.n_row),
                      std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < sum.n_row; i++) {
      for (int j = 0; j < sum.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col) {
          sum.matrix_data[i][j] += mat_a.matrix_data[i][j];
        }
        if (i < mat_b.n_row && j < mat_b.n_col) {
          sum.matrix_data[i][j] += mat_b.matrix_data[i][j];
        }
      }
    }
    return (sum);
  }

  template <class _TYPE>
  matrix<_TYPE> operator-(const matrix<_TYPE>& mat_a,
                          const matrix<_TYPE>& mat_b) {
    matrix<_TYPE> diff(std::max(mat_a.n_row, mat_b.n_row),
                       std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < diff.n_row; i++) {
      for (int j = 0; j < diff.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col) {
          diff.matrix_data[i][j] += mat_a.matrix_data[i][j];
        }
        if (i < mat_b.n_row && j < mat_b.n_col) {
          diff.matrix_data[i][j] -= mat_b.matrix_data[i][j];
        }
      }
    }
    return (diff);
  }

  template <class _TYPE>
  matrix<_TYPE> operator*(const matrix<_TYPE>& mat_a,
                          const matrix<_TYPE>& mat_b) {
    matrix<_TYPE> prod(std::max(mat_a.n_row, mat_b.n_row),
                       std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < prod.n_row; i++) {
      for (int j = 0; j < prod.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col && i < mat_b.n_row &&
            j < mat_b.n_col) {
          prod.matrix_data[i][j] =
              mat_a.matrix_data[i][j] * mat_b.matrix_data[i][j];
        }
      }
    }
    return (prod);
  }

  template <class _TYPE1, class _TYPE2>
  matrix<_TYPE2> operator*(const _TYPE1& scalar, matrix<_TYPE2>& mat) {
    matrix<_TYPE2> prod(mat.n_row, mat.n_col);
    for (int i = 0; i < prod.n_row; i++) {
      for (int j = 0; j < prod.n_col; j++) {
        prod.matrix_data[i][j] = scalar * mat.matrix_data[i][j];
      }
    }
    return (prod);
  }

  template <class _TYPE>
  matrix<_TYPE> operator/(const matrix<_TYPE>& mat_a,
                          const matrix<_TYPE>& mat_b) {
    matrix<_TYPE> quot(std::max(mat_a.n_row, mat_b.n_row),
                       std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < quot.n_row; i++) {
      for (int j = 0; j < quot.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col && i < mat_b.n_row &&
            j < mat_b.n_col && mat_b.matrix_data[i][j] != _TYPE()) {
          quot.matrix_data[i][j] =
              mat_a.matrix_data[i][j] / mat_b.matrix_data[i][j];
        }
      }
    }
    return (quot);
  }

  template <class _TYPE>
  matrix<_TYPE>& operator++(const matrix<_TYPE>& mat) {
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        mat.matrix_data[i][j]++;
      }
    }
    return (mat);
  }

  template <class _TYPE>
  matrix<_TYPE>& operator--(const matrix<_TYPE>& mat) {
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        mat.matrix_data[i][j]--;
      }
    }
    return (mat);
  }
}
#endif
