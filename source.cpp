#include <pessum.h>
#include <iostream>
#include "cognitio_files/cognosco_headers.hpp"

using namespace cognitio;

void Handle(std::pair<int, std::string> entry) {
  if (entry.first == pessum::ERROR) {
    system("setterm -fore red");
  } else if (entry.first == pessum::WARNING) {
    system("setterm -fore yellow");
  } else if (entry.first == pessum::TRACE) {
    system("setterm -fore blue");
  } else if (entry.first == pessum::DATA) {
    system("setterm -fore magenta");
  } else if (entry.first == pessum::DEBUG) {
    system("setterm -fore cyan");
  } else if (entry.first == pessum::SUCCESS) {
    system("setterm -fore green");
  }
  std::cout << entry.second << "\n";
  system("setterm -fore white");
}

std::string Print(std::vector<double> vec) {
  std::string out = "<";
  for (int i = 0; i < vec.size(); i++) {
    out += std::to_string(vec[i]);
    if (i != vec.size() - 1) {
      out += ",";
    }
  }
  out += ">";
  return(out);
}

std::vector<std::pair<std::vector<double>, std::vector<double>>> GenData(
    int size) {
  std::vector<std::pair<std::vector<double>, std::vector<double>>> set;
  for (int i = 0; i < size; i++) {
    std::pair<std::vector<double>, std::vector<double>> entry;
    entry.first = {(double)(rand() % 10), (double)(rand() % 10),
                   (double)(rand() % 10)};
    //int ones = 0, tens = 0,
    int sum = entry.first[0] + entry.first[1] + entry.first[2];
    //tens = sum % 10;
    //sum /= 10;
    //ones = sum % 10;
    //entry.second = {(double)ones, (double)tens};
    if(sum > 10){
      entry.second = {0.0, 1.0};
    }else{
      entry.second = {1.0, 0.0};
    }
    set.push_back(entry);
  }
  return (set);
}

int main(int argc, char const* argv[]) {
  std::vector<std::pair<std::vector<double>, std::vector<double>>>
      trainingdata = GenData(100);
  std::vector<std::pair<std::vector<double>, std::vector<double>>> testdata =
      GenData(100);
  pessum::SetLogHandle(Handle);
  NeuralNetwork net(std::vector<int>{3, 5, 2});
  net.EnablePessumLogging();
  std::vector<double> input{7.0, 5.0, 3.0};
  std::vector<double> input_2{7.0, 5.0, 3.0};
  std::cout << net.GetVis(false) << "\n";
  std::cout << Print(input) << " ==> " << Print(net.ForwardProp(input)) << "\n";
  net.GradientDecent(trainingdata, 1, 10, 3.0, testdata);
  std::cout << "==================================================\n";
  for(int i = 0; i < testdata.size(); i++){
    std::cout << Print(testdata[i].first) << " ==> " << Print(net.ForwardProp(testdata[i].first)) << " == " << Print(testdata[i].second) << "\n";
  }
  std::cout << "==================================================\n";
  std::cout << Print(input) << " ==> " << Print(net.ForwardProp(input)) << "\n";
  //std::cout << net.GetVis(false) << "\n";
  // std::cout << net.PrintData() << "\n";
  //Print(input);
  //Print(net.ForwardProp(input));
  pessum::SaveLog("out.log");
  return 0;
}
