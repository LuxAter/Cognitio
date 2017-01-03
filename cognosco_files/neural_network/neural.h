#ifndef COGNOSCO_FILES_NEURAL_NETWORK_NEURAL_H_
#define COGNOSCO_FILES_NEURAL_NETWORK_NEURAL_H_
#include <string>
#include <vector>
namespace cognosco {
namespace neural {
struct Item {
  std::vector<double> inputdata;
  std::vector<double> expectedresult;
};
class NeuralNetwork {
 public:
  void CreateNeuralNetwork(std::vector<int> neurons, std::string name = "NULL");
  std::vector<double> ForwardPropogation(std::vector<double> inputdata);
  void BackwardPropogation(std::vector<double> inputdata,
                           std::vector<double> expectedoutput);
  void StandardGradientDecent(std::vector<Item> inputdata, int epochs,
                              int batchsize);
  double Evaluate(std::vector<Item> evaluationdata);
  void SetLearingRate(double rate);

 private:
  std::vector<std::vector<double>> activations;
  std::vector<std::vector<std::vector<double>>> weights;
  double learningrate;
  int logloc;
  double Sigmoid(double z);
  std::vector<double> Sigmoid(std::vector<double> z);
  std::vector<std::vector<double>> Sigmoid(std::vector<std::vector<double>> z);
  double SigmoidPrime(double z);
  std::vector<double> SigmoidPrime(std::vector<double> z);
};
}
}
#endif
