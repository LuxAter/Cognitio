#ifndef COGNOSCO_MATRIX_HPP
#define COGNOSCO_MATRIX_HPP
#include <stdarg.h>
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

    matrix(int rows, int cols, std::vector<std::vector<_TYPE>> elements) {
      n_row = rows;
      n_col = cols;
      matrix_data = std::vector<std::vector<_TYPE>>(
          rows, std::vector<_TYPE>(cols, _TYPE()));
      setElements(elements);
    }

    matrix(const matrix<_TYPE>& copy_mat) {
      n_row = copy_mat.n_row;
      n_col = copy_mat.n_col;
      matrix_data = copy_mat.matrix_data;
    }

    // Destructor
    ~matrix() {
      n_col = 0;
      n_row = 0;
      matrix_data.clear();
    }
    // Data Manipulation

    void setElements(std::vector<_TYPE> elements) {
      if (elements.size() == n_row * n_col) {
        int current_row = 0, current_col = 0;
        for (int i = 0; i < elements.size(); i++) {
          matrix_data[current_row][current_col] = elements[i];
          current_col++;
          if (current_col == n_col) {
            current_col = 0;
            current_row++;
          }
        }
      }
    }

    void setElements(std::vector<std::vector<_TYPE>> elements) {
      if (elements.size() == n_row && elements[0].size() == n_col) {
        matrix_data = elements;
      }
    }

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

    void reShape(int rows, int cols) {
      std::vector<_TYPE> element_vector;
      for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
          element_vector.push_back(matrix_data[i][j]);
        }
      }
      matrix_data.clear();
      n_row = rows;
      n_col = cols;
      matrix_data = std::vector<std::vector<_TYPE>>(
          rows, std::vector<_TYPE>(cols, _TYPE()));
      setElements(element_vector);
    }
    // Property Retrieval
    int getDim() { return (2); }

    std::pair<int, int> getShape() { return (std::make_pair(n_row, n_col)); }

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
    // Operators
    _TYPE operator[](int i) {
      if (i > -1 && i < n_row * n_col) {
        return (matrix_data[i / n_col][i % n_col]);
      } else {
        return (_TYPE());
      }
    }

    _TYPE operator[](std::pair<int, int> index) {
      if (index.first > -1 && index.second > -1 && index.first < n_row &&
          index.second < n_col) {
        return (matrix_data[index.first][index.second]);
      } else {
        return (_TYPE());
      }
    }

    _TYPE operator()(int i, int j) {
      if (i > -1 && j > -1 && i < n_row && j < n_col) {
        return (matrix_data[i][j]);
      } else {
        return (_TYPE());
      }
    }

    int n_col = 0, n_row = 0;
    std::vector<std::vector<_TYPE>> matrix_data;

   private:
  };
}
#endif
