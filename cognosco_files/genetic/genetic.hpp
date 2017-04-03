#ifndef COGNOSCO_GENETIC_HPP
#define COGNOSCO_GENETIC_HPP
#include <vector>
namespace cognosco{
  template<class _T>
  class Genetic{
  public:
    struct Individual{
      _T chromosome;
      double fitness;
    };
    Genetic();
    Genetic(int pop_count);
    void GenPop(int pop_count);
    void FindFitness();
    void Selection();
    void CrossOver();
    void Mutate();
  private:
    int pop_size, generation;
    double mutation_rate, cross_over_rate;
    _T (*generate)();
    double (*fitness)(_T);
    _T (*crossOver)(_T, _T, double);
    _T (*mutate)(_T, double);
    std::vector<Individual> population;
  };
}
#endif
