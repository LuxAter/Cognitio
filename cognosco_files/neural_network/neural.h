#ifndef COGNOSCO_FILES_NEURAL_NETWORK_NEURAL_H_
#define COGNOSCO_FILES_NEURAL_NETWORK_NEURAL_H_
#include <string>
#include <vector>
namespace cognosco {
namespace neural {
struct Neuron {
  double value, bias;
  std::vector<double> weights;
};
struct Item {
  std::vector<double> inputdata;
  std::vector<double> expectedresult;
};
class NeuralNetwork {
 public:
  void CreateNeuralNetwork(std::vector<int> neurons, std::string name = "NULL");
  std::vector<double> FeedForward(std::vector<double> inputdata);
  void StandardGradientDecent(std::vector<Item> inputdata, int epochs,
                              int batchsize);
  double Evaluate(std::vector<Item> evaluationdata);
  void SetLearingRate(double rate);

 private:
  std::vector<std::vector<Neuron>> network;
  std::vector<std::vector<std::vector<double>>> weightnetwork;
  std::vector<std::vector<double>> biasnetwork;
  std::vector<std::vector<double>> activationnetwork;
  double learningrate;
  int logloc;
  double Sigmoid(double z);
  std::vector<double> VectorSigmoid(std::vector<double> z);
  double SigmoidPrime(double z);
  std::vector<double> VectorSigmoidPrime(std::vector<double> z);
  void UpdateBatch(std::vector<Item> batch);
  void BackProp(Item item,
                std::vector<std::vector<std::vector<double>>>& deltanablab,
                std::vector<std::vector<std::vector<double>>>& deltanablaw);
};
}
}
#endif
