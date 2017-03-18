#ifndef COGNOSCO_MATRIX_RELATIONAL_OPS_HPP
#define COGNOSCO_MATRIX_RELATIONAL_OPS_HPP
namespace cognosco {
  template <class _TYPE>
  bool operator==(const matrix<_TYPE>& mat_a, const matrix<_TYPE>& mat_b) {
    if (mat_a.n_row != mat_b.n_row || mat_a.n_col != mat_b.n_col) {
      return (false);
    } else {
      for (int i = 0; i < mat_a.n_row; i++) {
        for (int j = 0; j < mat_a.n_col; j++) {
          if (mat_a.matrix_data[i][j] != mat_b.matrix_data[i][j]) {
            return (false);
          }
        }
      }
      return (true);
    }
  }

  template <class _TYPE>
  bool operator!=(const matrix<_TYPE>& mat_a, const matrix<_TYPE>& mat_b) {
    return (!(mat_a == mat_b));
  }
}
#endif
