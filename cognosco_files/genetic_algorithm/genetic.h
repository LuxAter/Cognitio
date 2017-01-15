#ifndef COGNOSCO_FILES_GENETIC_ALGORITHM_GENETIC_CORE_H_
#define COGNOSCO_FILES_GENETIC_ALGORITHM_GENETIC_CORE_H_
#include <vector>
#include "../neural_network/neural.h"
namespace cognosco {
namespace genetic {
struct Individual {
  neural::NeuralNetwork network;
  double fitness, cumulativefitness, bottemup;
};
class GeneticAlgorithm {
 public:
  void CreateGeneticAlgorithm(std::vector<int> neurons = {1, 1},
                              int population = 100, double mutation = 0.05,
                              std::string name = "");
  void RunGeneticAlgorithm(int genorations = 1);
  void SetFitnessFunction(double (*function)(neural::NeuralNetwork));

 private:
  int logloc;
  std::vector<Individual> population;
  std::vector<int> neuralpattern{10, 5, 2};
  int populationsize = 100, globalgenorations = 0;
  double mutationrate = 0.3, totalfitness = 0, totalbottemup = 0;
  double (*FitnessFunction)(neural::NeuralNetwork);

  void GenoratePopulation();
  void CalculateFitness();
  void SumFitness();
  void Sort();
  void CumulateFitness();
  void Killoff();
  void Reproduce();
  void Mutate();
  int SelectHigh();
  int SelectLow();
};
bool SortCheck(const Individual a, const Individual b);
}
}
#endif
