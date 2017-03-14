#include "cognosco_files/matrix/matrix.hpp"
#include <iostream>
#include <pessum.h>

using namespace cognosco;

int main() {
  pessum::InitializePessum(true, true);
  matrix<double> mat(3, 3, 3.1415);
  std::cout << mat.get_string() << '\n';
  pessum::TerminatePessum();
  return (0);
}
