#include <aequus_headers.h>
#include <pessum_headers.h>
#include "cognosco_files/neural_network/neural.h"

int main(int argc, char *argv[]) {
  pessum::InitializePessumComponents();
  aequus::framework::SdlStartUp();
  cognosco::neural::NeuralNetwork net;
  net.CreateNeuralNetwork({2, 2, 2});
  pessum::math::DisplayVector(net.FeedForward({1, 1}));
  std::vector<cognosco::neural::Item> data;
  cognosco::neural::Item newitem;
  newitem.inputdata = {1, 1};
  newitem.expectedresult = {1, 0};
  data.push_back(newitem);
  net.SetLearingRate(1);
  net.StandardGradientDecent(data, 1, 1);
  aequus::framework::TerminateSdl();
  pessum::TerminatePessumComponents();
  return (1);
}
