#ifndef COGNOSCO_MATRIX_HPP
#define COGNOSCO_MATRIX_HPP
#include <string>
#include <vector>
namespace cognosco {
template <class _TYPE> class matrix {
public:
  int n_col = 0, n_row = 0;
  std::vector<std::vector<_TYPE>> matrix_data;
  matrix() {
    n_col = 0;
    n_row = 0;
  }
  matrix(int rows, int cols) {
    matrix_data.clear();
    matrix_data = std::vector<std::vector<_TYPE>>(
        rows, std::vector<_TYPE>(cols, _TYPE()));
    n_row = rows;
    n_col = cols;
  }
  matrix(int rows, int cols, _TYPE val) {
    matrix_data.clear();
    matrix_data =
        std::vector<std::vector<_TYPE>>(rows, std::vector<_TYPE>(cols, val));
    n_row = rows;
    n_col = cols;
  }
  ~matrix() {
    n_col = 0;
    n_row = 0;
    matrix_data.clear();
  }
  std::string get_string() {
    std::string line = "";
    line += "[";
    for (int i = 0; i < n_row; i++) {
      line += "[";
      for (int j = 0; j < n_col; j++) {
        line += to_string(matrix_data[i][j]);
        if (j != n_col - 1) {
          line += ",";
        }
      }
      line += "]";
      if (i != n_row - 1) {
        line += ",";
      }
    }
    line += "]";
    return (line);
  }
};
}
#endif
