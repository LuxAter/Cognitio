#include <math.h>
#include <pessum_headers.h>
#include <algorithm>
#include <vector>
#include "neural.h"

double E = 2.7182818284590452353602874713526624977572470936999;

double drand() { return ((double)rand() / RAND_MAX); }

std::string GenName() {
  std::string name;
  std::vector<std::string> names =
      pessum::luxreader::LoadLuxListFile("resources/luxfiles/names.lux");
  name = names[rand() % names.size()];
  return (name);
}

void cognosco::neural::NeuralNetwork::CreateNeuralNetwork(
    std::vector<int> neurons, std::string name) {
  if (name == "NULL") {
    name = GenName();
  }
  logloc = pessum::logging::AddLogLocation(
      "cognosco_files/neural_network/neural/[" + name + "]/");
  for (int i = 0; i < neurons.size(); i++) {
    std::vector<double> activationlayer;
    std::vector<std::vector<double>> weightlayer;
    for (int j = 0; j < neurons[i]; j++) {
      std::vector<double> weightneuron;
      if (i > 0) {
        for (int k = 0; k < neurons[i - 1]; k++) {
          weightneuron.push_back(drand() + drand() - 1);
        }
      }
      weightlayer.push_back(weightneuron);
      activationlayer.push_back(0);
    }
    activations.push_back(activationlayer);
    weights.push_back(weightlayer);
  }
  pessum::logging::LogLoc(pessum::logging::LOG_SUCCESS,
                          "Created neural network \"" + name + "\"", logloc,
                          "CreateNeuralNetwork");
}

void cognosco::neural::NeuralNetwork::StandardGradientDecent(
    std::vector<Item> inputdata, int epochs, int batchsize) {
  std::string line = "";
  for (int i = 0; i < weights.size(); i++) {
    line += "[";
    for (int j = 0; j < weights[i].size(); j++) {
      line += "[";
      for (int k = 0; k < weights[i][j].size(); k++) {
        line += pessum::math::ReduceDouble(weights[i][j][k]) + " ";
      }
      line += "]";
    }
    line += "]";
  }
  pessum::logging::Log(pessum::logging::LOG_INFORMATION, line);
  for (int i = 0; i < epochs; i++) {
    std::random_shuffle(inputdata.begin(), inputdata.end());
    std::vector<std::vector<Item>> batches;
    std::vector<Item> batch;
    for (int j = 0; j < inputdata.size(); j++) {
      batch.push_back(inputdata[j]);
      if (batch.size() == batchsize) {
        batches.push_back(batch);
        batch.clear();
      }
    }
    for (int j = 0; j < batches.size(); j++) {
      for (int k = 0; k < batches[j].size(); k++) {
        ForwardPropogation(batches[j][k].inputdata);
        BackwardPropogation(batches[j][k].inputdata,
                            batches[j][k].expectedresult);
      }
    }
    double percentage = Evaluate(inputdata);
    // pessum::logging::Log(pessum::logging::LOG_DATA,
    //                     std::to_string(percentage) + "\% accuracy");
  }

  line = "";
  for (int i = 0; i < weights.size(); i++) {
    line += "[";
    for (int j = 0; j < weights[i].size(); j++) {
      line += "[";
      for (int k = 0; k < weights[i][j].size(); k++) {
        line += pessum::math::ReduceDouble(weights[i][j][k]) + " ";
      }
      line += "]";
    }
    line += "]";
  }
  pessum::logging::Log(pessum::logging::LOG_INFORMATION, line);
}

double cognosco::neural::NeuralNetwork::Evaluate(
    std::vector<Item> evaluationdata) {
  double sum = 0;
  for (int i = 0; i < evaluationdata.size(); i++) {
    std::vector<double> output =
        ForwardPropogation(evaluationdata[i].inputdata);
    int goalindex = 0, recevedindex = 0;
    for (int j = 0; j < evaluationdata[i].expectedresult.size(); j++) {
      if (evaluationdata[i].expectedresult[j] >=
          evaluationdata[i].expectedresult[goalindex]) {
        goalindex++;
      }
      if (output[j] >= output[recevedindex]) {
        recevedindex++;
      }
    }
    if (goalindex == recevedindex) {
      sum++;
    }
  }
  sum /= (double)evaluationdata.size();
  return (sum);
}

void cognosco::neural::NeuralNetwork::SetLearingRate(double rate) {
  learningrate = rate;
}

std::vector<double> cognosco::neural::NeuralNetwork::ForwardPropogation(
    std::vector<double> inputdata) {
  if (inputdata.size() != activations[0].size()) {
    pessum::logging::LogLoc(
        pessum::logging::LOG_WARNING,
        "Input data size must match size of first layer of neurons", logloc,
        "ForwardPropogation");
    return (inputdata);
  }
  activations[0] = inputdata;
  for (int i = 1; i < activations.size(); i++) {
    std::vector<std::vector<double>> activationmatrix;
    activationmatrix.push_back(activations[i - 1]);
    activationmatrix = Sigmoid(pessum::math::DotMatrix(
        pessum::math::ProductMatrix(activationmatrix, weights[i])));
    activations[i] = activationmatrix[0];
  }
  return (activations[activations.size() - 1]);
}

void cognosco::neural::NeuralNetwork::BackwardPropogation(
    std::vector<double> inputdata, std::vector<double> expectedouput) {
  std::vector<double> outputerror =
      pessum::math::Diff(expectedouput, activations[activations.size() - 1]);
  std::vector<double> deltaweightsum = pessum::math::Product(
      SigmoidPrime(pessum::math::DotMatrix(weights[weights.size() - 1])[0]),
      outputerror);
  std::vector<std::vector<double>> activationmatrix;
  for (int i = 0; i < deltaweightsum.size(); i++) {
    activationmatrix.push_back(activations[activations.size() - 2]);
  }
  std::vector<std::vector<double>> deltaweights = pessum::math::DivMatrix(
      pessum::math::TransposeMatrix({deltaweightsum}), activationmatrix);
  deltaweights = pessum::math::ScalarMultiplyMatrix(learningrate, deltaweights);
  weights[weights.size() - 1] =
      pessum::math::SumMatrix(deltaweights, weights[weights.size() - 1]);
  std::vector<std::vector<double>> deltahiddensum =
      pessum::math::TransposeMatrix({deltaweightsum});
  for (int i = activations.size() - 2; i > 0; i--) {
    std::vector<double> hiddensum = pessum::math::DotMatrix(weights[i])[0];
    deltahiddensum = pessum::math::ProductMatrix(
        pessum::math::DivMatrix(deltahiddensum, weights[i + 1]),
        {SigmoidPrime(hiddensum)});
    activationmatrix.clear();
    activationmatrix = {activations[i - 1]};
    deltaweights.clear();
    for (int j = 0; j < deltahiddensum.size(); j++) {
      for (int k = 0; k < deltahiddensum[j].size(); k++) {
        std::vector<double> col;
        for (int l = 0; l < activationmatrix[0].size(); l++) {
          if (j == 0) {
            col.push_back(deltahiddensum[j][k] / activationmatrix[0][l]);
          } else {
            deltaweights[j][k] +=
                (deltahiddensum[j][k] / activationmatrix[0][l]);
          }
        }
        if (j == 0) {
          deltaweights.push_back(col);
        }
      }
    }
    deltaweights =
        pessum::math::ScalarMultiplyMatrix(learningrate, deltaweights);
    weights[i] = pessum::math::SumMatrix(deltaweights, weights[i]);
  }
}

double cognosco::neural::NeuralNetwork::Sigmoid(double z) {
  return (1.0 / (1.0 + (pow(E, -z))));
}

std::vector<double> cognosco::neural::NeuralNetwork::Sigmoid(
    std::vector<double> z) {
  for (int i = 0; i < z.size(); i++) {
    z[i] = Sigmoid(z[i]);
  }
  return (z);
}
std::vector<std::vector<double>> cognosco::neural::NeuralNetwork::Sigmoid(
    std::vector<std::vector<double>> z) {
  for (int i = 0; i < z.size(); i++) {
    z[i] = Sigmoid(z[i]);
  }
  return (z);
}

double cognosco::neural::NeuralNetwork::SigmoidPrime(double z) {
  return (pow(E, z) / (pow(1 + pow(E, z), 2)));
}

std::vector<double> cognosco::neural::NeuralNetwork::SigmoidPrime(
    std::vector<double> z) {
  for (int i = 0; i < z.size(); i++) {
    z[i] = SigmoidPrime(z[i]);
  }
  return (z);
}
