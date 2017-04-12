#ifndef COGNOSCO_MATRIX_MATH_HPP
#define COGNOSCO_MATRIX_MATH_HPP
#include "matrix.hpp"
namespace cognosco {
  template <class _T>
  _T Sum(Matrix<_T> mat) {
    _T sum_value = _T();
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        sum_value += mat.terms[i][j];
      }
    }
    return (sum_value);
  }

  template <class _T>
  Matrix<_T> Sum(Matrix<_T> mat, int axis) {
    Matrix<_T> sum_mat;
    if (axis == 0) {
      sum_mat = Matrix<_T>(1, mat.n_col);
      for (int i = 0; i < mat.n_row; i++) {
        for (int j = 0; j < mat.n_col; j++) {
          sum_mat.terms[0][j] += mat.terms[i][j];
        }
      }
    } else if (axis == 1) {
      sum_mat = Matrix<_T>(mat.n_row, 1);
      for (int i = 0; i < mat.n_row; i++) {
        for (int j = 0; j < mat.n_col; j++) {
          sum_mat.terms[i][0] += mat.terms[i][j];
        }
      }
    }
    return (sum_mat);
  }

  template <class _T>
  Matrix<_T> Transpose(Matrix<_T> mat) {
    Matrix<_T> transpose_mat(mat.n_col, mat.n_row);
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        transpose_mat.terms[j][i] = mat.terms[i][j];
      }
    }
    return (transpose_mat);
  }

  template <class _T>
  _T Det(Matrix<_T> mat) {
    _T det_value = _T();
    if (mat.n_row == mat.n_col) {
      if (mat.n_row == 2) {
        det_value = (mat.terms[0][0] * mat.terms[1][1]) -
                    (mat.terms[1][0] * mat.terms[0][1]);
      } else {
        for (int i = 0; i < mat.n_col; i++) {
          Matrix<_T> sub_mat(mat.n_row - 1, mat.n_col - 1);
          for (int j = 1; j < mat.n_row; j++) {
            int sub_k = 0;
            for (int k = 0; k < mat.n_col; k++) {
              if (k != i) {
                sub_mat.terms[j - 1][sub_k] = mat.terms[j][k];
                sub_k++;
              }
            }
          }
          det_value += (pow(-1, i) * mat.terms[0][i] * Det(sub_mat));
        }
      }
    } else {
      pessum::Log(pessum::WARNING,
                  "Matrix must be square not %ix%i for determinant", "Det",
                  mat.n_row, mat.n_col);
    }
    return (det_value);
  }

  template <class _T>
  _T Trace(Matrix<_T> mat) {
    _T trace_value = _T();
    for (int i = 0; i < mat.n_row && i < mat.n_col; i++) {
      trace_value += mat.terms[i][i];
    }
    return (trace_value);
  }

  template <class _T>
  Matrix<_T> Inverse(Matrix<_T> mat) {
    Matrix<_T> mat_inverse(mat.n_row, mat.n_col);
    if (mat.n_row == mat.n_col && Det(mat) != _T()) {
      for (int r = 0; r < mat_inverse.n_row; r++) {
        for (int c = 0; c < mat_inverse.n_col; c++) {
          Matrix<_T> sub_mat(mat.n_row - 1, mat.n_col - 1);
          int sub_r = 0;
          for (int cr = 0; cr < mat.n_row; cr++) {
            int sub_c = 0;
            if (cr != r) {
              for (int cc = 0; cc < mat.n_col; cc++) {
                if (cc != c) {
                  sub_mat.terms[sub_r][sub_c] = mat.terms[cr][cc];
                  sub_c++;
                }
              }
              sub_r++;
            }
          }
          mat_inverse.terms[r][c] = (pow(-1, r) * pow(-1, c) * det(sub_mat));
        }
      }
      mat_inverse = Transpose(mat_inverse);
      mat_inverse = (1 / Det(mat)) * mat_inverse;
    } else {
      pessum::Log(pessum::WARNING,
                  "Matrix must be square not %ix%i and must have a determinant "
                  "not equal to %i not %i",
                  "Inverse", mat.n_row, mat.n_col, _T(), Det(mat));
    }
    return (mat_inverse);
  }

  template <class _T>
  Matrix<_T> Dot(Matrix<_T> mat_a, Matrix<_T> mat_b) {
    Matrix<_T> mat_dot(mat_a.n_row, mat_b.n_col);
    if (mat_a.n_col == mat_b.n_row) {
      for (int i = 0; i < mat_a.n_row; i++) {
        for (int j = 0; j < mat_b.n_col; j++) {
          for (int k = 0; k < mat_a.n_col; k++) {
            mat_dot.terms[i][j] += mat_a.terms[i][k] * mat_b.terms[k][j];
          }
        }
      }
    } else {
      pessum::Log(
          pessum::WARNING,
          "Columns of matrix A must match the rows of matrix B not %i and %i",
          "Dot", mat_a.n_col, mat_b.n_row);
    }
    return (mat_dot);
  }
  template <class _T>
  Matrix<_T> operator+(const Matrix<_T>& mat_a, const Matrix<_T>& mat_b) {
    Matrix<_T> sum(std::max(mat_a.n_row, mat_b.n_row),
                   std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < sum.n_row; i++) {
      for (int j = 0; j < sum.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col) {
          sum.terms[i][j] += mat_a.terms[i][j];
        }
        if (i < mat_b.n_row && j < mat_b.n_col) {
          sum.terms[i][j] += mat_b.terms[i][j];
        }
      }
    }
    return (sum);
  }

  template <class _T>
  Matrix<_T> operator-(const Matrix<_T>& mat_a, const Matrix<_T>& mat_b) {
    Matrix<_T> diff(std::max(mat_a.n_row, mat_b.n_row),
                    std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < diff.n_row; i++) {
      for (int j = 0; j < diff.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col) {
          diff.terms[i][j] += mat_a.terms[i][j];
        }
        if (i < mat_b.n_row && j < mat_b.n_col) {
          diff.terms[i][j] -= mat_b.terms[i][j];
        }
      }
    }
    return (diff);
  }

  template <class _T>
  Matrix<_T> operator*(const Matrix<_T>& mat_a, const Matrix<_T>& mat_b) {
    Matrix<_T> prod(std::max(mat_a.n_row, mat_b.n_row),
                    std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < prod.n_row; i++) {
      for (int j = 0; j < prod.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col && i < mat_b.n_row &&
            j < mat_b.n_col) {
          prod.terms[i][j] = mat_a.terms[i][j] * mat_b.terms[i][j];
        }
      }
    }
    return (prod);
  }

  template <class _T1, class _T2>
  Matrix<_T2> operator*(const _T1& scalar, Matrix<_T2>& mat) {
    Matrix<_T2> prod(mat.n_row, mat.n_col);
    for (int i = 0; i < prod.n_row; i++) {
      for (int j = 0; j < prod.n_col; j++) {
        prod.terms[i][j] = scalar * mat.terms[i][j];
      }
    }
    return (prod);
  }

  template <class _T>
  Matrix<_T> operator/(const Matrix<_T>& mat_a, const Matrix<_T>& mat_b) {
    Matrix<_T> quot(std::max(mat_a.n_row, mat_b.n_row),
                    std::max(mat_a.n_col, mat_b.n_col));
    for (int i = 0; i < quot.n_row; i++) {
      for (int j = 0; j < quot.n_row; j++) {
        if (i < mat_a.n_row && j < mat_a.n_col && i < mat_b.n_row &&
            j < mat_b.n_col && mat_b.terms[i][j] != _T()) {
          quot.terms[i][j] = mat_a.terms[i][j] / mat_b.terms[i][j];
        }
      }
    }
    return (quot);
  }

  template <class _T>
  Matrix<_T>& operator++(const Matrix<_T>& mat) {
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        mat.terms[i][j]++;
      }
    }
    return (mat);
  }

  template <class _T>
  Matrix<_T>& operator--(const Matrix<_T>& mat) {
    for (int i = 0; i < mat.n_row; i++) {
      for (int j = 0; j < mat.n_col; j++) {
        mat.terms[i][j]--;
      }
    }
    return (mat);
  }
}
#endif
