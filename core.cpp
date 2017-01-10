#include <pessum.h>
#include "cognosco_files/neural_network/neural.h"

int main(int argc, char *argv[]) {
  pessum::InitializePessumComponents();
  cognosco::neural::NeuralNetwork net;

  net.CreateNeuralNetwork({10, 5, 2});
  std::vector<cognosco::neural::Item> data;
  for (int i = 0; i < 100; i++) {
    cognosco::neural::Item newitem;
    newitem.inputdata = {(double)rand(), (double)rand(), (double)rand(), (double)rand(), (double)rand(),
                         (double)rand(), (double)rand(), (double)rand(), (double)rand(), (double)rand()};
    newitem.expectedresult = {0, 1};
    data.push_back(newitem);
  }
  net.SetLearingRate(0.2);
  double percentage = net.Evaluate(data);
  pessum::logging::Log(pessum::logging::DATA,
                       std::to_string(percentage * 100) + "\% accuracy");
  net.StochasticGradientDescent(data, 10, 10, true);
  percentage = net.Evaluate(data);
  pessum::logging::Log(pessum::logging::DATA,
                       std::to_string(percentage * 100) + "\% accuracy");
  pessum::TerminatePessumComponents();
  return (1);
}
