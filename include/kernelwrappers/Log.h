#ifndef KERNELWRAPPERS_LOG_H
#define KERNELWRAPPERS_LOG_H

#include <functional>
#include <string_view>

namespace kernelwrappers {

/*
 * Minimal, dependency-free logging facility.
 *
 * The library emits log messages through an injectable callback. By default no
 * callback is set and all messages are discarded, so consumers can plug in
 * their own logger (spdlog, std::cout, ...) without kernelwrappers depending
 * on any particular logging framework.
 */
class Log {
public:
  enum class Level { Trace, Debug, Info, Warn, Error, Off };

  using Callback = std::function<void(Level, std::string_view)>;

  // Set the callback that receives all log messages.
  static void setCallback(Callback callback);

  // Set the minimum level that is forwarded to the callback.
  static void setLevel(Level level);

  static Level getLevel();

  static void trace(std::string_view message);
  static void debug(std::string_view message);
  static void info(std::string_view message);
  static void warn(std::string_view message);
  static void error(std::string_view message);

private:
  static void log(Level level, std::string_view message);

  static Level level_;
  static Callback callback_;
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_LOG_H
