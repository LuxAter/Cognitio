#include <aequus_headers.h>
#include <pessum_headers.h>
#include "cognosco_files/neural_network/neural.h"

int main(int argc, char *argv[]) {
  pessum::InitializePessumComponents();
  aequus::framework::SdlStartUp();
  cognosco::neural::NeuralNetwork net;
  net.CreateNeuralNetwork({3, 5, 10});
  std::vector<cognosco::neural::Item> data;
  for (int i = 0; i < 100; i++) {
    cognosco::neural::Item newitem;
    for (int j = 0; j < 3; j++) {
      newitem.inputdata.push_back((double)(rand() % 3));
    }
    int index = pessum::math::Total(newitem.inputdata);
    for (int j = 0; j < 10; j++) {
      if (j == index) {
        newitem.expectedresult.push_back(1);
      } else {
        newitem.expectedresult.push_back(0);
      }
    }
    data.push_back(newitem);
  }
  net.SetLearingRate(0.5);
  double percentage = net.Evaluate(data);
  pessum::logging::Log(pessum::logging::LOG_DATA,
                       std::to_string(percentage * 100) + "\% accuracy");
  net.StochasticGradientDescent(data, 10, 10, true);
  percentage = net.Evaluate(data);
  pessum::logging::Log(pessum::logging::LOG_DATA,
                       std::to_string(percentage * 100) + "\% accuracy");
  aequus::framework::TerminateSdl();
  pessum::TerminatePessumComponents();
  return (1);
}
