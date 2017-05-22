#include <pessum.h>
#include <iostream>
#include "cognitio_files/cognosco_headers.hpp"

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

using namespace cognitio;

int main(int argc, char const* argv[]) {
  pessum::SetLogHandle(Handle);
  NeuralNetwork net(std::vector<int>{3, 5, 5, 5, 2});
std::cout << net.GetVis(false) << "\n";
pessum::SaveLog("out.log");
  return 0;
}
