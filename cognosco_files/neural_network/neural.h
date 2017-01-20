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
  void CreateNeuralNetwork(std::vector<int> neurons, std::string name = "",
                           int loc = -1, bool logout = true);
  void SaveNetworkToFile(std::string file);
  void LoadNetworkFromFile(std::string file);
  void StochasticGradientDescent(std::vector<Item> inputdata, int epochs,
                                 int batchsize, bool display = false);
  void UpdateBatch(std::vector<Item> items);
  double Evaluate(std::vector<Item> evaluationdata);
  void SetLearingRate(double rate);
  std::vector<double> ForwardPropogation(std::vector<double> inputdata);
  std::vector<std::vector<std::vector<double>>>
  BackwardPropogation(std::vector<double> inputdata,
                      std::vector<double> expectedoutput);
  std::vector<double> GetVector();
  void InterpretVector(std::vector<double> vec);

private:
  std::vector<std::vector<double>> activations;
  std::vector<std::vector<std::vector<double>>> weights;
  double learningrate;
  int logloc, globalepoch;
  double Sigmoid(double z);
  std::vector<double> Sigmoid(std::vector<double> z);
  std::vector<std::vector<double>> Sigmoid(std::vector<std::vector<double>> z);
  double SigmoidPrime(double z);
  std::vector<double> SigmoidPrime(std::vector<double> z);
};
}
}
#endif
