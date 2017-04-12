#include <iostream>
#include "src/cognosco_headers.hpp"

using namespace cognosco;

void Handle(std::pair<int, std::string> entry) {
  if (entry.first == pessum::ERROR) {
    system("setterm -fore red");
  } else if (entry.first == pessum::WARNING) {
    system("setterm -fore yellow");
  } else if (entry.first == pessum::TRACE) {
    system("setterm -fore cyan");
  }
  std::cout << entry.second << "\n";
  system("setterm -fore white");
}

std::string PrintVec(std::vector<double> vec) {
  std::string str = "<";
  for (int i = 0; i < vec.size(); i++) {
    std::stringstream ss;
    ss << vec[i];
    std::string val;
    ss >> val;
    str += val;
    if (i != vec.size() - 1) {
      str += ",";
    }
  }
  str += ">";
  return (str);
}

int main() {
  pessum::SetLogHandle(Handle);
  srand(time(NULL));
  Network net(3, 3, 5, 2);
  std::cout << net.GetString() << "\n";
  std::vector<double> in = {2, 5, 7};
  std::cout << PrintVec(in) << "->\n";
  // PrintVec(net.ForwardProp(in));
  return (0);
}
