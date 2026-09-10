#include <cerrno>
#include <cxxopts.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef KERNELWRAPPERS_TUNE_APPLICATION
#error "KERNELWRAPPERS_TUNE_APPLICATION must be configured by CMake"
#endif

namespace {

int waitForChild(pid_t pid) {
  int status = 0;
  while (waitpid(pid, &status, 0) == -1) {
    if (errno != EINTR) {
      std::perror("waitpid");
      return 1;
    }
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}

int captureKernels(const std::string &patterns, const std::filesystem::path &dir,
                   bool force) {
  const pid_t pid = fork();
  if (pid == -1) {
    std::perror("fork");
    return 1;
  }
  if (pid == 0) {
    setenv(force ? "KERNEL_LAUNCHER_CAPTURE_FORCE"
                 : "KERNEL_LAUNCHER_CAPTURE",
           patterns.c_str(), 1);
    setenv("KERNEL_LAUNCHER_CAPTURE_DIR", dir.c_str(), 1);
    std::cout << "Calling tune application: " << KERNELWRAPPERS_TUNE_APPLICATION << std::endl;
    execl(KERNELWRAPPERS_TUNE_APPLICATION, KERNELWRAPPERS_TUNE_APPLICATION,
          static_cast<char *>(nullptr));
    std::perror("exec application");
    _exit(127);
  }
  return waitForChild(pid);
}

int tuneCaptures(const std::vector<std::filesystem::path> &captures,
                 bool force, const std::string &time) {
  std::vector<std::string> args = {KERNELWRAPPERS_TUNE_PYTHON,
                                   KERNELWRAPPERS_TUNE_SCRIPT,
                                   "--strategy",
                                   "random",
                                   "--output",
                                   KERNELWRAPPERS_TUNE_WISDOM_DIRECTORY};
  if (force) {
    args.emplace_back("--force");
  }
  if (!time.empty()) {
    args.emplace_back("--time");
    args.emplace_back(time);
  }
  for (const auto &capture : captures) {
    args.emplace_back(capture.string());
  }

  std::vector<char *> argv;
  argv.reserve(args.size() + 1);
  for (std::string &arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid == -1) {
    std::perror("fork");
    return 1;
  }
  if (pid == 0) {
    setenv("KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON",
           KERNELWRAPPERS_TUNE_KERNEL_LAUNCHER_PYTHON, 1);
    execv(argv[0], argv.data());
    std::perror("exec Kernel Launcher tuner");
    _exit(127);
  }
  return waitForChild(pid);
}

std::set<std::string>
captureKeys(const std::vector<std::filesystem::path> &captures) {
  const std::regex keyPattern(R"regex("key"\s*:\s*"([^"]+)")regex");
  std::set<std::string> keys;
  for (const auto &capture : captures) {
    std::ifstream input(capture);
    std::string contents((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    std::smatch match;
    if (std::regex_search(contents, match, keyPattern)) {
      keys.insert(match[1].str());
    }
  }
  return keys;
}

} // namespace

int main(int argc, char **argv) {
  cxxopts::Options options("tune", "Capture and tune KernelWrappers kernels");
  options.add_options()("a,all", "Tune every kernel launched by the application")(
      "f,force", "Overwrite matching wisdom entries")(
      "time", "Maximum tuning time per kernel", cxxopts::value<std::string>())(
      "h,help", "Print usage")(
      "kernels", "Kernel tuning keys", cxxopts::value<std::vector<std::string>>());
  options.parse_positional({"kernels"});

  cxxopts::ParseResult arguments;
  try {
    arguments = options.parse(argc, argv);
  } catch (const cxxopts::exceptions::exception &error) {
    std::cerr << error.what() << "\n\n" << options.help();
    return 2;
  }

  if (arguments.count("help")) {
    std::cout << options.help();
    return 0;
  }

  const bool force = arguments.count("force") != 0;
  const bool all = arguments.count("all") != 0;
  const std::string time =
      arguments.count("time") ? arguments["time"].as<std::string>() : "";
  const std::vector<std::string> kernels =
      arguments.count("kernels")
          ? arguments["kernels"].as<std::vector<std::string>>()
          : std::vector<std::string>();

  if (all == !kernels.empty()) {
    std::cerr << "Specify one or more kernels, or use --all.\n\n"
              << options.help();
    return 2;
  }

  const std::string patterns = all ? "*" : [&]() {
    std::string result;
    for (const std::string &kernel : kernels) {
      if (!result.empty()) {
        result += ',';
      }
      result += kernel;
    }
    return result;
  }();

  const auto captureDir = std::filesystem::temp_directory_path() /
                          ("kernelwrappers-tune-" + std::to_string(getpid()));
  const std::filesystem::path wisdomDir(KERNELWRAPPERS_TUNE_WISDOM_DIRECTORY);
  std::filesystem::create_directories(captureDir);
  std::filesystem::create_directories(wisdomDir);

  std::cout << "Capturing " << patterns << "...\n";
  if (const int result = captureKernels(patterns, captureDir, force); result != 0) {
    std::cerr << "The application exited with status " << result
              << "; captures retained at " << captureDir << '\n';
    return result;
  }

  std::vector<std::filesystem::path> captures;
  for (const auto &entry : std::filesystem::directory_iterator(captureDir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".json") {
      captures.emplace_back(entry.path());
    }
  }
  if (captures.empty()) {
    std::cerr << "No captures were created. A matching wisdom entry may already "
                 "exist; use --force to retune it. Otherwise, check that the "
                 "requested tuning keys are launched by the application.\n";
    return 1;
  }

  const std::set<std::string> keys = captureKeys(captures);
  if (keys.empty()) {
    std::cerr << "The captures do not contain tuning keys.\n";
    return 1;
  }

  std::cout << "Tuning " << captures.size() << " captured kernel(s)...\n";
  const int result = tuneCaptures(captures, force, time);
  if (result != 0) {
    std::cerr << "Tuning failed; captures retained at " << captureDir << '\n';
    return result;
  }

  for (const std::string &key : keys) {
    if (!std::filesystem::exists(wisdomDir / (key + ".wisdom"))) {
      std::cerr << "The tuner did not produce wisdom for " << key
                << "; captures retained at " << captureDir << '\n';
      return 1;
    }
  }

  std::filesystem::remove_all(captureDir);
  std::cout << "Wisdom written to " << wisdomDir << '\n';
  return 0;
}
