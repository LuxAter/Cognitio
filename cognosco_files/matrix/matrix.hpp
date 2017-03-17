#ifndef COGNOSCO_MATRIX_HPP
#define COGNOSCO_MATRIX_HPP
#include <string>
#include <vector>
#include "../strings/string.hpp"
namespace cognosco {
  template <class _TYPE>
  class matrix {
   public:
    // Constructors
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
    // Destructor
    ~matrix() {
      n_col = 0;
      n_row = 0;
      matrix_data.clear();
    }
    // Data Manipulation
    void setElement(int i, int j, _TYPE value) { matrix_data[i][j] = value; }

    void fill(_TYPE value) {
      for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
          matrix_data[i][j] = value;
        }
      }
    }

    void fillDiagonal(_TYPE value) {
      for (int i = 0; i < n_row && i < n_col; i++) {
        matrix_data[i][i] = value;
      }
    }
    // Property Retrieval
    int Dim() { return (2); }

    std::vector<int> Shape() { return (std::vector<int>{n_row, n_col}); }

    // Visualization
    std::string getString() {
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

   private:
    int n_col = 0, n_row = 0;
    std::vector<std::vector<_TYPE>> matrix_data;
  };
}
#endif
