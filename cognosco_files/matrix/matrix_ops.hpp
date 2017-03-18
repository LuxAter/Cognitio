#ifndef COGNOSCO_MATRIX_OPS_HPP
#define COGNOSCO_MATRIX_OPS_HPP
#include <iostream>
#include "matrix.hpp"
namespace cognosco {
  template <class _T>
  _T sum(matrix<_T> mat) {
    _T sum_value = _T();
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        sum_value += mat.matrix_data[i][j];
      }
    }
    return (sum_value);
  }

  template <class _T>
  matrix<_T> sum(matrix<_T> mat, int axis) {
    matrix<_T> sum_mat;
    if (axis == 0) {
      sum_mat = matrix<_T>(1, mat.n_col);
      for (int i = 0; i < mat.n_row; i++) {
        for (int j = 0; j < mat.n_col; j++) {
          sum_mat.matrix_data[0][j] += mat.matrix_data[i][j];
        }
      }
    } else if (axis == 1) {
      sum_mat = matrix<_T>(mat.n_row, 1);
      for (int i = 0; i < mat.n_row; i++) {
        for (int j = 0; j < mat.n_col; j++) {
          sum_mat.matrix_data[i][0] += mat.matrix_data[i][j];
        }
      }
    }
    return (sum_mat);
  }

  template <class _T>
  matrix<_T> transpose(matrix<_T> mat) {
    matrix<_T> transpose_mat(mat.n_col, mat.n_row);
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        transpose_mat.matrix_data[j][i] = mat.matrix_data[i][j];
      }
    }
    return (transpose_mat);
  }

  template <class _T>
  _T det(matrix<_T> mat) {
    _T det_value = _T();
    if (mat.n_row == mat.n_col) {
      if (mat.n_row == 2) {
        det_value = (mat.matrix_data[0][0] * mat.matrix_data[1][1]) -
                    (mat.matrix_data[1][0] * mat.matrix_data[0][1]);
      } else {
        for (int i = 0; i < mat.n_col; i++) {
          matrix<_T> sub_mat(mat.n_row - 1, mat.n_col - 1);
          for (int j = 1; j < mat.n_row; j++) {
            int sub_k = 0;
            for (int k = 0; k < mat.n_col; k++) {
              if (k != i) {
                sub_mat.matrix_data[j - 1][sub_k] = mat.matrix_data[j][k];
                sub_k++;
              }
            }
          }
          det_value += (pow(-1, i) * mat.matrix_data[0][i] * det(sub_mat));
        }
      }
    }
    return (det_value);
  }

  template <class _T>
  _T trace(matrix<_T> mat) {
    _T trace_value = _T();
    for (int i = 0; i < mat.n_row && i < mat.n_col; i++) {
      trace_value += mat.matrix_data[i][i];
    }
    return (trace_value);
  }

  template <class _T>
  matrix<_T> inverse(matrix<_T> mat) {
    matrix<_T> mat_inverse(mat.n_row, mat.n_col);
    if (mat.n_row == mat.n_col && det(mat) != _T()) {
      for (int r = 0; r < mat_inverse.n_row; r++) {
        for (int c = 0; c < mat_inverse.n_col; c++) {
          matrix<_T> sub_mat(mat.n_row - 1, mat.n_col - 1);
          int sub_r = 0;
          for (int cr = 0; cr < mat.n_row; cr++) {
            int sub_c = 0;
            if (cr != r) {
              for (int cc = 0; cc < mat.n_col; cc++) {
                if (cc != c) {
                  sub_mat.matrix_data[sub_r][sub_c] = mat.matrix_data[cr][cc];
                  sub_c++;
                }
              }
              sub_r++;
            }
          }
          mat_inverse.matrix_data[r][c] =
              (pow(-1, r) * pow(-1, c) * det(sub_mat));
        }
      }
      mat_inverse = transpose(mat_inverse);
      mat_inverse = (1 / det(mat)) * mat_inverse;
    }
    return (mat_inverse);
  }

  template <class _T>
  matrix<_T> dot(matrix<_T> mat_a, matrix<_T> mat_b) {
    matrix<_T> mat_dot(mat_a.n_row, mat_b.n_col);
    if (mat_a.n_col == mat_b.n_row) {
      for (int i = 0; i < mat_a.n_row; i++) {
        for (int j = 0; j < mat_b.n_col; j++) {
          for (int k = 0; k < mat_a.n_col; k++) {
            mat_dot.matrix_data[i][j] +=
                mat_a.matrix_data[i][k] * mat_b.matrix_data[k][j];
          }
        }
      }
    }
    return (mat_dot);
  }
}
#endif
