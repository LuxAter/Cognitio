#include "genetic.h"
#include <pessum.h>
#include "../cognosco.h"
#include "../neural_network/neural.h"
#include <vector>
#include <algorithm>
#include <cmath>


void cognosco::genetic::GeneticAlgorithm::CreateGeneticAlgorithm(std::vector<int> neurons, int population, double mutation, std::string name){
  neuralpattern = neurons;
  populationsize = population;
  mutationrate = mutation;
  if(name == ""){
    name = GenName();
  }
  logloc = pessum::logging::AddLogLocation("cognosco_files/genetic_algorithm/genetic/[" + name + "]");
  GenoratePopulation();
  pessum::logging::LogLoc(pessum::logging::SUCCESS,
                          "Created genetic algorithm \"" + name + "\"", logloc,
                          "CreateGeneticAlgorithm");
}

void cognosco::genetic::GeneticAlgorithm::RunGeneticAlgorithm(int genorations){
  for(int i = 0; i < genorations; i++){
    CalculateFitness();
    SumFitness();
    Sort();
    CumulateFitness();
    Killoff();
    Reproduce();
    Mutate();
  }
  pessum::logging::Log(pessum::logging::DATA, std::to_string(genorations));
  pessum::logging::Log(pessum::logging::DATA, "Best Accuracy:" + std::to_string(population[0].fitness));
  globalgenorations += genorations;
}

void cognosco::genetic::GeneticAlgorithm::SetFitnessFunction(double (*Fitness)(neural::NeuralNetwork)){
  FitnessFunction = Fitness;
}

void cognosco::genetic::GeneticAlgorithm::GenoratePopulation(){
  for(int i = 0; i < populationsize; i++){
    Individual newind;
    newind.network.CreateNeuralNetwork(neuralpattern, "", logloc);
    newind.fitness = 0;
    population.push_back(newind);
  }
}


void cognosco::genetic::GeneticAlgorithm::CalculateFitness(){
  for(int i = 0; i < populationsize; i++){
    population[i].fitness = FitnessFunction(population[i].network);
  }
}

void cognosco::genetic::GeneticAlgorithm::SumFitness(){
  totalfitness = 0;
  for(int i = 0; i < populationsize; i++){
    totalfitness += population[i].fitness;
  }
}

void cognosco::genetic::GeneticAlgorithm::Sort(){
    std::sort(population.begin(), population.end(), SortCheck);
}

void cognosco::genetic::GeneticAlgorithm::CumulateFitness(){
  double fitness = 0;
  for (int i = populationsize - 1; i >= 0; i--) {
    fitness += population[i].fitness;
    population[i].cumulativefitness = fitness;
  }
  fitness = 0;
  for (int i = 0; i < populationsize; i++) {
    fitness += population[i].fitness;
    population[i].bottemup = fitness;
  }
  totalbottemup = fitness;
}

void cognosco::genetic::GeneticAlgorithm::Killoff(){
  while (population.size() > ceil(populationsize / (double)2)) {
    int index = SelectLow();
    //population[index].network.TerminateNeural
    population.erase(population.begin() + index);
  }
}

void cognosco::genetic::GeneticAlgorithm::Reproduce(){
  std::vector<Individual> newpop;
  std::vector<Individual> parents = population;
  while (newpop.size() + parents.size() < populationsize) {
    Individual newind;
    int parentone = SelectHigh();
    int parenttwo = SelectHigh();
    if (parenttwo == parentone) {
      if (parenttwo == 0) {
        parenttwo++;
      } else {
        parenttwo = 0;
      }
    }
    newind.network.CreateNeuralNetwork(neuralpattern, "", logloc);
    newind.fitness = 0;
    std::vector<double> weightvector, aweights, bweights;
    aweights =population[parentone].network.GetVector();
    weightvector = aweights;
    bweights =population[parenttwo].network.GetVector();
    for(int i = 0;i < aweights.size();i++){
      if(drand() > 0.5){
        weightvector[i] = bweights[i];
      }
    }
    newind.network.InterpretVector(weightvector);

    newpop.push_back(newind);
  }
  population.clear();
  population = parents;
  population.insert(population.end(), newpop.begin(), newpop.end());
}

void cognosco::genetic::GeneticAlgorithm::Mutate(){
  for (int i = ceil(populationsize / (double)2); i < population.size(); i++) {
    std::vector<double> weightvector = population[i].network.GetVector();
    for(int j = 0; j < weightvector.size();j++){
      if(drand() <= mutationrate){
        weightvector[j] = drand();
      }
    }
    population[i].network.InterpretVector(weightvector);
  }
}

int cognosco::genetic::GeneticAlgorithm::SelectHigh() {
  double point = drand() * totalfitness;
  for (int i = population.size() - 1; i >= 0; i--) {
    if (population[i].cumulativefitness > point) {
      return (i);
    }
  }
  return (0);
}

int cognosco::genetic::GeneticAlgorithm::SelectLow() {
  double point = drand() * totalbottemup;
  for (int i = population.size() - 1; i >= 0; i--) {
    if (point < population[i].bottemup) {
      return (i);
    }
  }
  return (0);
}


bool cognosco::genetic::SortCheck(const Individual a, const Individual b){
  return(a.fitness > b.fitness);
}
