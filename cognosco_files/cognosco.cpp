#include "cognosco.h"
#include <pessum.h>
#include <vector>
#include <string>

std::string cognosco::GenName() {
  std::string name;
  std::vector<std::string> names =
      pessum::luxreader::LoadLuxListFile("resources/luxfiles/names.lux");
  name = names[rand() % names.size()];
  return (name);
}


double cognosco::drand() { return ((double)rand() / RAND_MAX); }
