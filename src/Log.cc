#include <kernelwrappers/Log.h>

namespace kernelwrappers {

Log::Level Log::level_ = Log::Level::Info;
Log::Callback Log::callback_ = nullptr;

void Log::setCallback(Callback callback) { callback_ = std::move(callback); }

void Log::setLevel(Level level) { level_ = level; }

Log::Level Log::getLevel() { return level_; }

void Log::trace(std::string_view message) { log(Level::Trace, message); }

void Log::debug(std::string_view message) { log(Level::Debug, message); }

void Log::info(std::string_view message) { log(Level::Info, message); }

void Log::warn(std::string_view message) { log(Level::Warn, message); }

void Log::error(std::string_view message) { log(Level::Error, message); }

void Log::log(Level level, std::string_view message) {
  if (!callback_ || level < level_) {
    return;
  }
  callback_(level, message);
}

} // namespace kernelwrappers
