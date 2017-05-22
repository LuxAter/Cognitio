#include "activation_functions.hpp"
#include <vector>
#include <math.h>

std::vector<double> cognitio::Sigmoid(std::vector<double> z){
  for(int i = 0; i < z.size(); i++){
    z[i] = 1.0 / (1.0 + exp(-z[i]));
  }
  return(z);
}
