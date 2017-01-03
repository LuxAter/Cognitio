#include <aequus_headers.h>
#include <pessum_headers.h>
#include "cognosco_files/neural_network/neural.h"

int main(int argc, char *argv[]) {
  pessum::InitializePessumComponents();
  aequus::framework::SdlStartUp();
  cognosco::neural::NeuralNetwork net;
  net.CreateNeuralNetwork({5, 3, 2});
  std::vector<cognosco::neural::Item> data;
  for (int i = 0; i < 100; i++) {
    cognosco::neural::Item newitem;
    newitem.inputdata = {1, 2, 3, 4, 5};
    newitem.expectedresult = {1, 0};
    data.push_back(newitem);
  }
  net.SetLearingRate(1);
  net.StandardGradientDecent(data, 100, 1);
  double percentage = net.Evaluate(data);
  pessum::logging::Log(pessum::logging::LOG_DATA,
                       std::to_string(percentage) + "\% accuracy");
  aequus::framework::TerminateSdl();
  pessum::TerminatePessumComponents();
  return (1);
}
