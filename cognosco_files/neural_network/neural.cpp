#include <math.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <pessum.h>
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
  globalepoch = 0;
  pessum::logging::LogLoc(pessum::logging::SUCCESS,
                          "Created neural network \"" + name + "\"", logloc,
                          "CreateNeuralNetwork");
}

void cognosco::neural::NeuralNetwork::StochasticGradientDescent(
    std::vector<Item> inputdata, int epochs, int batchsize, bool display) {
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
      UpdateBatch(batches[j]);
    }
    globalepoch++;
    if (display == true) {
      double percentage = Evaluate(inputdata);
      pessum::logging::LogLoc(pessum::logging::INFORMATION,
                              "Epoch: " + std::to_string(globalepoch) + "[\%" +
                                  pessum::math::ReduceDouble(percentage * 100) +
                                  "]",
                              logloc, "StochasticGradientDescent");
    }
  }
}

void cognosco::neural::NeuralNetwork::UpdateBatch(std::vector<Item> items) {
  std::vector<std::vector<std::vector<double>>> deltaweights, newdeltaweights;
  deltaweights = weights;
  for (int i = 0; i < deltaweights.size(); i++) {
    for (int j = 0; j < deltaweights[i].size(); j++) {
      for (int k = 0; k < deltaweights[i][j].size(); k++) {
        deltaweights[i][j][k] = 0;
      }
    }
  }
  for (int i = 0; i < items.size(); i++) {
    ForwardPropogation(items[i].inputdata);
    newdeltaweights =
        BackwardPropogation(items[i].inputdata, items[i].expectedresult);
    for (int j = deltaweights.size() - 1; j > 0; j--) {
      deltaweights[j] =
          pessum::math::SumMatrix(deltaweights[j], newdeltaweights[j - 1]);
    }
  }
  for (int i = 0; i < deltaweights.size(); i++) {
    deltaweights[i] = pessum::math::ScalarMultiplyMatrix(
        (learningrate / (double)items.size()), deltaweights[i]);
    weights[i] = pessum::math::SumMatrix(weights[i], deltaweights[i]);
  }
}

double cognosco::neural::NeuralNetwork::Evaluate(
    std::vector<Item> evaluationdata) {
  double sum = 0;
  for (int i = 0; i < evaluationdata.size(); i++) {
    std::vector<double> outputerror =
        pessum::math::Diff(evaluationdata[i].expectedresult,
                           ForwardPropogation(evaluationdata[i].inputdata));
    for (int j = 0; j < outputerror.size(); j++) {
      outputerror[j] = fabs(outputerror[j]);
    }
    double maximum = outputerror.size();
    sum += ((maximum - pessum::math::Total(outputerror)) / maximum);
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
        pessum::logging::WARNING,
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

std::vector<std::vector<std::vector<double>>>
cognosco::neural::NeuralNetwork::BackwardPropogation(
    std::vector<double> inputdata, std::vector<double> expectedouput) {
  std::vector<std::vector<std::vector<double>>> output;
  std::vector<double> outputerror;
  outputerror =
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
  output.push_back(deltaweights);
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
            if (activationmatrix[0][l] == 0) {
              col.push_back(0);
            } else {
              col.push_back(deltahiddensum[j][k] / activationmatrix[0][l]);
            }
          } else {
            if (activationmatrix[0][l] == 0) {
              deltaweights[k][l] += 0;
            } else {
              deltaweights[k][l] +=
                  (deltahiddensum[j][k] / activationmatrix[0][l]);
            }
          }
        }
        if (j == 0) {
          deltaweights.push_back(col);
        }
      }
    }
    output.insert(output.begin(), deltaweights);
  }
  return (output);
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
