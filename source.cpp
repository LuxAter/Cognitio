#include <iostream>
#include "src/cognosco_headers.hpp"

using namespace cognosco;

int main() {
  srand(time(NULL));
  Matrix<double> mat(3, 3);
  std::cout << mat.GetString() << "\n";
  return (0);
}
