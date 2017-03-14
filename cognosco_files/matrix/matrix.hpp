#ifndef COGNOSCO_MATRIX_HPP
#define COGNOSCO_MAtriX_HPP
#include <vector>
#include <string>
namespace cognosco{
  template<class _TYPE>
  class matrix{
  public:
    matrix(){
      n_col = 0;
      n_row = 0;
    }
    matrix(int rows, int cols){
      matrix_data.clear();
      matrix_data = std::vector<std::vector<_TYPE>>(cols, std::vector<_TYPE>(rows, _TYPE()));
      n_row = rows;
      n_col = cols;
    }
    matrix(int n_row, int n_col, _TYPE val){
      matrix_data.clear();
      matrix_data = std::vector<std::vector<_TYPE>(cols, std::vector<_TYPE>(rows, val));
      n_row = rows;
      n_col = cols;
    }
    ~matrix(){
      n_col = 0;
      n_row = 0;
      matrix_data.clear();
    }
    std::string get_string(){
      std::string line = "";
      line += "[";
      for(
    }
    int n_col = 0, n_row = 0;
    std::vector<std::vector<_TYPE>> matrix_data;
  };
}
#endif
