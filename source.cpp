#include <iostream>
#include "cognosco_files/cognosco.hpp"
#include "cognosco_files/matrix/matrix_header.hpp"
#include "cognosco_files/softmax.hpp"

using namespace cognosco;

int main() {
  srand(time(NULL));
  Network net(2, 3, 3);
  std::vector<double> in;
  for (int i = 0; i < 3; i++) {
    in.push_back(rand() % 10);
  }
  std::cout << "<";
  for (int i = 0; i < in.size(); i++) {
    std::cout << in[i] << ",";
  }
  std::cout << ">\n";
  in = net.ForwardProp(in);

  std::cout << "<";
  for (int i = 0; i < in.size(); i++) {
    std::cout << in[i] << ",";
  }
  std::cout << ">\n";
  return (0);
}
