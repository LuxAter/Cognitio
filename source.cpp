#include <iostream>
#include "cognosco_files/matrix/matrix.hpp"

using namespace cognosco;

int main() {
  matrix<int> mat(3, 3);
  mat.fillDiagonal(1);
  std::cout << mat.getString() << '\n';
  return (0);
}
