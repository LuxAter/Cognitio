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

void Print(std::vector<double> vec){
  std::cout << "<";
  for(int i = 0; i < vec.size(); i++){
    std::cout << vec[i];
    if(i != vec.size() - 1){
      std::cout << ",";
    }
  }
  std::cout << ">\n";
}

using namespace cognitio;

int main(int argc, char const* argv[]) {
  pessum::SetLogHandle(Handle);
  NeuralNetwork net(std::vector<int>{3, 2});
  std::cout << net.GetVis(false) << "\n";
  std::cout << net.PrintData() << "\n";
  std::vector<double> input{2.0,5.0,3.0};
  Print(input);
  Print(net.ForwardProp(input));
  pessum::SaveLog("out.log");
  return 0;
}
