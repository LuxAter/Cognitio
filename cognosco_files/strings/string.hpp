#ifndef COGNOSCO_STRING
#define COGNOSCO_STRING
#include <string>
namespace cognosco {
  std::string to_string(int value);
  std::string to_string(long value);
  std::string to_string(long long value);
  std::string to_string(unsigned value);
  std::string to_string(unsigned long value);
  std::string to_string(unsigned long long value);
  std::string to_string(float value);
  std::string to_string(double value);
  std::string to_string(long double value);
  std::string to_string(char value);
  std::string to_string(const char* value);
}
#endif
