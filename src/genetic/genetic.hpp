#ifndef COGNOSCO_GENETIC_HPP
#define COGNOSCO_GENETIC_HPP
#define drand() (fabs((double)rand() / (RAND_MAX + 1)))
#include <math.h>
#include <pessum.h>
#include <algorithm>
#include <iostream>
#include <vector>
namespace cognosco {
  template <class _T>
  class Genetic {
   public:
    struct Individual {
      Individual() {}
      Individual(_T val) { chromosome = val; }
      _T chromosome = _T();
      double fitness = 0.0;
    };
    void GenPop(int pop_count) {
      pop_size = pop_count;
      if (generate != NULL) {
        while (population.size() < pop_count) {
          population.push_back(Individual(generate()));
        }
      } else {
        pessum::Log(pessum::WARNING, "No member genoration function supplied",
                    "cognosco/Genetic/GenPop");
      }
    }
    void FindFitness() {
      if (fitness != NULL) {
        for (int i = 0; i < population.size(); i++) {
          population[i].fitness = fitness(population[i].chromosome);
        }
      }
    }
    int SelectLow() {
      double point = fabs((double)rand() / (RAND_MAX + 1)) * total_fitness;
      double fit_count = total_fitness;
      for (int i = population.size() - 1; i >= 0; i--) {
        fit_count -= population[i].fitness;
        if (point < fit_count) {
          return (i);
        }
      }
      return (population.size() - 1);
    }
    int SelectHigh() {
      double point = drand() * total_fitness;
      double fit_count = 0;
      for (int i = 0; i < population.size(); i++) {
        fit_count += population[i].fitness;
        if (point < fit_count) {
          return (i);
        }
      }
      return (0);
    }
    void Reproduce() {
      std::vector<Individual> newpop;
      std::vector<Individual> parents = population;
      while (newpop.size() + parents.size() < pop_size) {
        Individual new_ind;
        int parent_one = SelectHigh();
        int parent_two = SelectHigh();
        if (parent_two == parent_one) {
          if (parent_two == 0) {
            parent_two++;
          } else {
            parent_two = 0;
          }
        }
        new_ind.chromosome =
            crossover(parents[parent_one].chromosome,
                      parents[parent_two].chromosome, cross_over_rate);
        newpop.push_back(new_ind);
      }
      population.clear();
      population = parents;
      population.insert(population.end(), newpop.begin(), newpop.end());
    }
    void Mutate() {
      for (int i = 1; i < population.size(); i++) {
        if (drand() < mutation_rate) {
          if (mutate != NULL) {
            population[i].chromosome =
                mutate(population[i].chromosome, mutation_rate);
          }
        }
      }
    }
    void KillOff(double perc = 0.5) {
      int cutoff = ceil((double)population.size() * perc);
      while (population.size() > cutoff) {
        population.erase(population.begin() + SelectLow());
      }
    }
    void SumFitness() {
      total_fitness = 0.0;
      for (int i = 0; i < population.size(); i++) {
        total_fitness += population[i].fitness;
      }
    }
    void Sort() {
      for (int i = 0; i < population.size(); i++) {
        int min = i;
        for (int j = i; j < population.size(); j++) {
          if (population[j].fitness > population[min].fitness) {
            min = j;
          }
        }
        if (min != i) {
          iter_swap(population.begin() + i, population.begin() + min);
        }
      }
    }

    void SetFunction(_T (*func)()) { generate = func; }
    void SetFunction(double (*func)(_T)) { fitness = func; }
    void SetFunction(_T (*func)(_T, _T, double)) { crossover = func; }
    void SetFunction(_T (*func)(_T, double)) { mutate = func; }

    void Run(int n_gen = 1) {
      for (int i = 0; i < n_gen; i++) {
        FindFitness();
        SumFitness();
        Sort();
        KillOff();
        Reproduce();
        Mutate();
        generation++;
      }
    }

    void Print() {
      for (int i = 0; i < population.size(); i++) {
        printf("[%i] %25s | %f\n", i, population[i].chromosome.c_str(),
               population[i].fitness);
      }
    }

    _T operator[](int index) {
      while (index > population.size()) {
        index -= population.size();
      }
      while (index < 0) {
        index = population.size() - index;
      }
      return (population[index].chromosome);
    }

   private:
    int pop_size, generation;
    double mutation_rate = 1, cross_over_rate = 0.5, total_fitness;
    _T (*generate)() = NULL;
    double (*fitness)(_T) = NULL;
    _T (*crossover)(_T, _T, double) = NULL;
    _T (*mutate)(_T, double) = NULL;
    std::vector<Individual> population;
  };
}
#endif
