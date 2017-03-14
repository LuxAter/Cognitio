#include <pessum.h>
#include "cognosco_files/matrix/matrix.hpp"

int main(){
  pessum::InitializePessum(true, true);
  pessum::TerminatePessum();
  return(0);
}
