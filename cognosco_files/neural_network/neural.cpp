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
    std::vector<Neuron> layer;
    std::vector<std::vector<double>> weightlayer;
    std::vector<double> biaslayer;
    std::vector<double> activationlayer;
    for (int j = 0; j < neurons[i]; j++) {
      Neuron newneuron;
      newneuron.bias = drand();
      newneuron.value = 0;
      if (i > 0) {
        for (int k = 0; k < neurons[i - 1]; k++) {
          newneuron.weights.push_back(drand());
        }
      }
      weightlayer.push_back(newneuron.weights);
      biaslayer.push_back(newneuron.bias);
      activationlayer.push_back(0);
      layer.push_back(newneuron);
    }
    network.push_back(layer);
    weightnetwork.push_back(weightlayer);
    biasnetwork.push_back(biaslayer);
    activationnetwork.push_back(activationlayer);
  }
  pessum::logging::LogLoc(pessum::logging::LOG_SUCCESS,
                          "Created neural network \"" + name + "\"", logloc,
                          "CreateNeuralNetwork");
}

void cognosco::neural::NeuralNetwork::StandardGradientDecent(
    std::vector<Item> inputdata, int epochs, int batchsize) {
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
  }
}

double cognosco::neural::NeuralNetwork::Evaluate(
    std::vector<Item> evaluationdata) {
  double sum = 0;
  for (int i = 0; i < evaluationdata.size(); i++) {
    std::vector<double> output = FeedForward(evaluationdata[i].inputdata);
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
  sum /= evaluationdata.size();
  return (sum);
}

void cognosco::neural::NeuralNetwork::SetLearingRate(double rate) {
  learningrate = rate;
}

std::vector<double> cognosco::neural::NeuralNetwork::FeedForward(
    std::vector<double> inputdata) {
  activationnetwork[0] = inputdata;
  for (int i = 1; i < network.size(); i++) {
    std::vector<std::vector<double>> activationmatrix;
    std::vector<std::vector<double>> biasmatrix;
    std::vector<std::vector<double>> weightmatrix = weightnetwork[i];
    activationmatrix.push_back(activationnetwork[i - 1]);
    biasmatrix.push_back(biasnetwork[i]);
    activationmatrix = pessum::math::SumMatrix(
        pessum::math::DotMatrix(
            pessum::math::ProductMatrix(weightmatrix, activationmatrix)),
        biasmatrix);
    activationnetwork[i] = activationmatrix[0];
    activationnetwork[i] = VectorSigmoid(activationnetwork[i]);
  }
  return (activationnetwork[activationnetwork.size() - 1]);
}

double cognosco::neural::NeuralNetwork::Sigmoid(double z) {
  return (1.0 / (1.0 + (pow(E, -z))));
}

std::vector<double> cognosco::neural::NeuralNetwork::VectorSigmoid(
    std::vector<double> z) {
  for (int i = 0; i < z.size(); i++) {
    z[i] = Sigmoid(z[i]);
  }
  return (z);
}

double cognosco::neural::NeuralNetwork::SigmoidPrime(double z) {
  return (Sigmoid(z) * (1 - Sigmoid(z)));
}

std::vector<double> cognosco::neural::NeuralNetwork::VectorSigmoidPrime(
    std::vector<double> z) {
  for (int i = 0; i < z.size(); i++) {
    z[i] = SigmoidPrime(z[i]);
  }
  return (z);
}

void cognosco::neural::NeuralNetwork::UpdateBatch(std::vector<Item> batch) {
  std::vector<std::vector<double>> nablab, deltanablab;
  std::vector<std::vector<std::vector<double>>> nablaw, deltanablaw;
  for (int i = 0; i < biasnetwork.size(); i++) {
    std::vector<double> nablabcol;
    std::vector<std::vector<double>> nablawcol;
    for (int j = 0; j < biasnetwork[j].size(); j++) {
      nablabcol.push_back(0);
      std::vector<double> nablawneuron;
      for (int k = 0; k < weightnetwork.size(); k++) {
        nablawneuron.push_back(0);
      }
      nablawcol.push_back(nablawneuron);
    }
    nablab.push_back(nablabcol);
    nablaw.push_back(nablawcol);
  }
  for (int i = 0; i < batch.size(); i++) {
    deltanablab.clear();
    deltanablaw.clear();
    BackProp(batch[i], deltanablab, deltanablaw);
    for (int layer = 0; layer < nablab.size(); layer++) {
      nablab[layer] = pessum::math::Sum(nablab[layer], deltanablab[layer]);
      for (int neuron = 0; neuron < nablaw[layer].size(); neuron++) {
        nablaw[layer][neuron] = pessum::math::Sum(nablaw[layer][neuron],
                                                  deltanablaw[layer][neuron]);
      }
    }
  }
  double multi = learningrate / (double)batch.size();
  for (int layer = 0; layer < biasnetwork.size(); layer++) {
    biasnetwork[layer] = pessum::math::Diff(
        biasnetwork[layer], pessum::math::ScalerMultiply(multi, nablab[layer]));
    for (int neuron = 0; neuron < weightnetwork[layer].size(); neuron++) {
      weightnetwork[layer][neuron] = pessum::math::Diff(
          weightnetwork[layer][neuron],
          pessum::math::ScalerMultiply(multi, nablaw[layer][neuron]));
    }
  }
}

void cognosco::neural::NeuralNetwork::BackProp(
    Item item, std::vector<std::vector<double>>& deltanablab,
    std::vector<std::vector<std::vector<double>>>& deltanablaw) {
  for (int i = 0; i < biasnetwork.size(); i++) {
    std::vector<double> nablabcol;
    std::vector<std::vector<double>> nablawcol;
    for (int j = 0; j < biasnetwork[j].size(); j++) {
      nablabcol.push_back(0);
      std::vector<double> nablawneuron;
      for (int k = 0; k < weightnetwork.size(); k++) {
        nablawneuron.push_back(0);
      }
      nablawcol.push_back(nablawneuron);
    }
    deltanablab.push_back(nablabcol);
    deltanablaw.push_back(nablawcol);
  }
  std::vector<std::vector<std::vector<double>>> activations, zs;
  std::vector<std::vector<double>> activation, z;
  activation.push_back(item.inputdata);
  activation = pessum::math::TransposeMatrix(activation);
  activations.push_back(activation);
  for (int layer = 1; layer < network.size(); layer++) {
    std::vector<std::vector<double>> b;
    b.push_back(biasnetwork[layer]);
    std::vector<std::vector<double>> w = weightnetwork[layer];
    if (activations.size() == 1) {
      z = pessum::math::ProductMatrix(w, activation);
      z = pessum::math::TransposeMatrix(z);
      z = pessum::math::DotMatrix(z);
      while (z.size() < b[0].size()) {
        z.push_back(z[0]);
      }
      z = pessum::math::TransposeMatrix(z);
    } else {
      z = pessum::math::MatrixMultiply(w, activation);
    }
    z = pessum::math::SumMatrix(z, b);
    zs.push_back(z);
    activation.clear();
    for (int i = 0; i < z.size(); i++) {
      activation.push_back(VectorSigmoid(z[i]));
    }
    activations.push_back(activation);
  }
  std::vector<std::vector<double>> y, sigprime;
  for (int i = 0; i < zs[zs.size() - 1].size(); i++) {
    sigprime.push_back(VectorSigmoidPrime(zs[zs.size() - 1][i]));
  }
  y.push_back(item.expectedresult);
  y = pessum::math::TransposeMatrix(y);
  std::vector<std::vector<double>> delta =
      pessum::math::DiffMatrix(activations[activations.size() - 1], y);
  delta = pessum::math::ProductMatrix(delta, sigprime);
  deltanablab[deltanablab.size() - 1] =
}
