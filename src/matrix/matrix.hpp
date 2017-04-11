#ifndef COGNOSCO_MATRIX_HPP
#define COGNOSCO_MATRIX_HPP
#include <pessum.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>
namespace cognosco {
  template <class _T>
  class Matrix {
   public:
    Matrix() {
      n_row = 0;
      n_col = 0;
    }

    Matrix(int rows, int cols) {
      n_row = rows;
      n_col = cols;
      terms = std::vector<std::vector<_T>>(n_row, std::vector<_T>(n_col, _T()));
    }

    Matrix(int rows, int cols, _T base) {
      n_row = rows;
      n_col = cols;
      terms = std::vector<std::vector<_T>>(n_row, std::vector<_T>(n_col, base));
    }

    Matrix(int rows, int cols, std::vector<_T> elements) {
      terms = std::vector<std::vector<_T>>(n_row, std::vector<_T>(n_col, _T()));
      int current_row = 0, current_col = 0;
      for (int i = 0; i < elements.size(); i++) {
        terms[current_row][current_col] = elements[i];
        current_col++;
        if (current_col == n_col) {
          current_col = 0;
          current_row++;
        }
      }
    }

    Matrix(std::vector<std::vector<_T>> elements) {
      n_row = elements.size();
      if (n_row > 0) {
        n_col = elements[0].size();
      } else {
        n_col = 0;
      }
      terms = elements;
    }

    Matrix(const Matrix& copy_mat) {
      n_row = copy_mat.n_row;
      n_col = copy_mat.n_col;
      terms = copy_mat.terms;
    }

    ~Matrix() {
      terms.clear();
      n_row = 0;
      n_col = 0;
    }

    void SetElements(std::vector<_T> elements) {
      if (elements.size() == n_row * n_col) {
        int current_row = 0, current_col = 0;
        for (int i = 0; i < elements.size(); i++) {
          terms[current_row][current_col] = elements[i];
          current_col++;
          if (current_col == n_col) {
            current_col = 0;
            current_row++;
          }
        }
      }
    }

    void SetElements(std::vector<std::vector<_T>> elements) {
      if (elements.size() == n_row && elements[0].size() == n_col) {
        terms = elements;
      }
    }

    void Set(int i, int j, _T val) {
      if (i > 0 && i < n_row && j > 0 && j < n_col) {
        terms[i][j] = val;
      }
    }

    void Fill(_T val) {
      for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
          terms[i][j] = val;
        }
      }
    }

    void FillDiagonal(_T val) {
      for (int i = 0; i < n_row && i < n_col; i++) {
        terms[i][i] = val;
      }
    }

    void FillRand(_T mean, _T std_dev) {
      std::default_random_engine generator(rand());
      std::normal_distribution<_T> rand_gen(mean, std_dev);
      for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
          terms[i][j] = rand_gen(generator);
        }
      }
    }

    void ReShape(int rows, int cols) {
      std::vector<_T> element_vector;
      for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
          element_vector.push_back(terms[i][j]);
        }
      }
      terms.clear();
      n_row = rows;
      n_col = cols;
      terms = std::vector<std::vector<_T>>(rows, std::vector<_T>(cols, _T()));
      int current_row = 0, current_col = 0;
      for (int i = 0; i < element_vector.size(); i++) {
        terms[current_row][current_col] = element_vector[i];
        current_col++;
        if (current_col == n_col) {
          current_col = 0;
          current_row++;
        }
      }
    }

    void ElementOperation(_T (*func)(_T)) {
      for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
          terms[i][j] = func(terms[i][j]);
        }
      }
    }

    std::pair<int, int> GetShape() { return (std::make_pair(n_row, n_col)); }

    std::string GetString() {
      std::string str = "[";
      for (int i = 0; i < n_row; i++) {
        str += "[";
        for (int j = 0; j < n_col; j++) {
          std::stringstream ss;
          ss << terms[i][j];
          std::string str_sub;
          ss >> str_sub;
          str += str_sub;
          if (j != n_col - 1) {
            str += ",";
          }
        }
        str += "]";
        if (i != n_row - 1) {
          str += ",";
        }
      }
      str += "]";
      return (str);
    }

    _T operator()(int i, int j) {
      if (i > -1 && j > -1 && i < n_row && j < n_col) {
        return (terms[i][j]);
      } else {
        return (_T());
      }
    }

   private:
    int n_row, n_col;
    std::vector<std::vector<_T>> terms;
  };
}
#endif
