#include <iostream>
#include "src/cognosco_headers.hpp"

using namespace cognosco;

int main() {
  srand(time(NULL));
  Matrix<double> mat(3, 3);
  mat.FillRand(0.0, 0.5);
  std::cout << mat.GetString() << "\n";
  return (0);
}
