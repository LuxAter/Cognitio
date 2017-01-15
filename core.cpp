#include <pessum.h>
#include "cognosco_files/neural_network/neural.h"
#include "cognosco_files/genetic_algorithm/genetic.h"

double Fit(cognosco::neural::NeuralNetwork network){
  std::vector<double> input;
  for(int i = 0; i < 10; i++){
    input.push_back(rand() % 10);
  }
  input = network.ForwardPropogation(input);
  double val = input[0];
  double goal = pessum::math::Total(input);
  double fitness = 1.0 / (double)(abs(goal - val));
  return(fitness);
}

int main(int argc, char *argv[]) {
  pessum::InitializePessumComponents();
  cognosco::genetic::GeneticAlgorithm gen;
  gen.CreateGeneticAlgorithm({10, 5, 1});
  gen.SetFitnessFunction(Fit);
  gen.RunGeneticAlgorithm();
  pessum::TerminatePessumComponents();
  return (1);
}
