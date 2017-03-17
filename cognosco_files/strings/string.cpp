#include <string>
#include "string.hpp"

std::string cognosco::to_string(int value) { return (std::to_string(value)); }
std::string cognosco::to_string(long value) { return (std::to_string(value)); }

std::string cognosco::to_string(long long value) {
  return (std::to_string(value));
}

std::string cognosco::to_string(unsigned value) {
  return (std::to_string(value));
}

std::string cognosco::to_string(unsigned long value) {
  return (std::to_string(value));
}

std::string cognosco::to_string(unsigned long long value) {
  return (std::to_string(value));
}

std::string cognosco::to_string(float value) { return (std::to_string(value)); }

std::string cognosco::to_string(double value) {
  return (std::to_string(value));
}

std::string cognosco::to_string(long double value) {
  return (std::to_string(value));
}

std::string cognosco::to_string(char value) { return (std::string(1, value)); }

std::string cognosco::to_string(const char* value) {
  return (std::string(value));
}
