#ifndef COGNITIO_LOGGER_HPP_
#define COGNITIO_LOGGER_HPP_

#include <cstdio>
#include <functional>
#include <string>

#define error(msg, ...) \
  logger::log(cognitio::logger::ERROR, msg, __FILE__, __func__, __LINE__, ##__VA_ARGS__)
#define warn(msg, ...) \
  logger::log(cognitio::logger::WARN, msg, __FILE__, __func__, __LINE__, ##__VA_ARGS__)
#define info(msg, ...) \
  logger::log(cognitio::logger::INFO, msg, __FILE__, __func__, __LINE__, ##__VA_ARGS__)
#define debug(msg, ...) \
  logger::log(cognitio::logger::DEBUG, msg, __FILE__, __func__, __LINE__, ##__VA_ARGS__)

namespace cognitio {
  namespace logger {
    enum LogLevel { ERROR, WARNING, INFO, DEBUG };
    void init_file(const std::string& file_path);
    void init_file_data();
    void init_file_datatime();
    void init_file_time();
    void term_file();

    void console(const std::string& msg, const unsigned& level);
    void file(const std::string& msg, const unsigned& level);

    void log(const LogLevel& level, const std::string& msg,
             const std::string& file, const std::string func,
             unsigned long line, ...);

    extern std::function<void(const std::string&, const unsigned&)>
        log_callback_;
    extern FILE* log_file_;
  }  // namespace logger
}  // namespace cognitio

#endif  // COGNITIO_LOGGER_HPP_
