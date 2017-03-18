#include <iostream>
#include "cognosco_files/matrix/matrix_header.hpp"

using namespace cognosco;

int main() {
  // matrix<int> a(3, 3, {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
  matrix<float> a(3, 3, {{-2, 2, -3}, {-1, 1, 3}, {2, 0, -1}});
  // std::cout << "A = " << a.getString() << "\n";
  std::cout << a.getString() << "\n";
  std::cout << "dim = " << a.getShape().first << "x" << a.getShape().second
            << "\n";
  std::cout << "det = " << det(a) << "\n";
  std::cout << "trace = " << trace(a) << "\n";
  std::cout << "inverse = " << inverse(a).getString() << "\n";
  return (0);
}
