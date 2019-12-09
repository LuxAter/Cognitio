#include "logger.hpp"

#include <cstdarg>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>

namespace cognitio {
  namespace logger {
    std::function<void(const std::string&, const unsigned&)> log_callback_ =
        console;
    FILE* log_file_;
  }  // namespace logger
}  // namespace cognitio

void cognitio::logger::init_file(const std::string& file_path) {
  log_file_ = fopen(file_path.c_str(), "w");
}
void cognitio::logger::init_file_data() {}
void cognitio::logger::init_file_datatime() {}
void cognitio::logger::init_file_time() {}
void cognitio::logger::term_file() {
  if (log_file_ != NULL) {
    fclose(log_file_);
  }
}

void cognitio::logger::console(const std::string& msg, const unsigned& level) {
  switch (level) {
    case 0:
      std::cout << "\033[1;31m" << msg << "\033[0m" << std::endl;
      break;
    case 1:
      std::cout << "\033[1;33m" << msg << "\033[0m" << std::endl;
      break;
    case 2:
      std::cout << "\033[1;36m" << msg << "\033[0m" << std::endl;
      break;
    case 3:
      std::cout << "\033[1;35m" << msg << "\033[0m" << std::endl;
      break;
  }
}
void cognitio::logger::file(const std::string& msg, const unsigned& level) {
  fprintf(log_file_, "%s", msg.c_str());
  if (level <= 1) fflush(log_file_);
}

void cognitio::logger::log(const LogLevel& level, const std::string& msg,
                           const std::string& file, const std::string func,
                           unsigned long line, ...) {
  va_list args;
  va_start(args, line);
  char buff[255];
  vsnprintf(buff, 255, msg.c_str(), args);
  va_end(args);
  char log_msg[512];
  time_t current_time = time(NULL);
  tm* local_tm = localtime(&current_time);
  std::string level_str = "";
  switch (level) {
    case ERROR:
      level_str = "ERROR";
      break;
    case WARNING:
      level_str = "WARNING";
      break;
    case INFO:
      level_str = "INFO";
      break;
    case DEBUG:
      level_str = "DEBUG";
      break;
  }
  snprintf(log_msg, 512, "%04d-%02d-%02d %02d:%02d:%02d | %s:%s:%lu | %s | %s",
           local_tm->tm_year + 1900, local_tm->tm_mon + 1, local_tm->tm_mday,
           local_tm->tm_hour, local_tm->tm_min, local_tm->tm_sec, file.c_str(),
           func.c_str(), line, level_str.c_str(), buff);
  if (log_callback_ != NULL) {
    log_callback_(std::string(log_msg), level);
  }
}
