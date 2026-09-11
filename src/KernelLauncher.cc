#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelLauncher.h>
#include <kernelwrappers/Log.h>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace kernelwrappers {
void KernelLauncher::launch(JITKernel &kernel) {
  // Ensure that a compiled kernel exists before launch.
  // If a compiled kernel is already available, for instance when the user
  // called compile() explicitly, this will be a no-op.
  kernel.compile();

#ifdef KERNELWRAPPERS_TUNER
  if (kernel.itsOnlineTuning) {
    if (kernel.captureLaunch()) {
      kernel.itsStream.launchKernel(
          *kernel.itsFunction, kernel.itsGridDims.x, kernel.itsGridDims.y,
          kernel.itsGridDims.z, kernel.itsBlockDims.x, kernel.itsBlockDims.y,
          kernel.itsBlockDims.z, 0, kernel.itsKernelArgs);
    }
    return;
  }
#endif

  kernel.itsStream.launchKernel(*kernel.itsFunction, kernel.itsGridDims.x,
                                kernel.itsGridDims.y, kernel.itsGridDims.z,
                                kernel.itsBlockDims.x, kernel.itsBlockDims.y,
                                kernel.itsBlockDims.z, 0, kernel.itsKernelArgs);
}
}

#ifdef KERNELWRAPPERS_TUNER
namespace kernelwrappers {

const char *KernelLauncher::defaultWisdomDir() {
#ifdef KERNELWRAPPERS_WISDOM_PATH
  return KERNELWRAPPERS_WISDOM_PATH;
#else
  return nullptr;
#endif
}

std::vector<std::string> &KernelLauncher::wisdomDirsStorage() {
  static std::vector<std::string> dirs = {"."};
  return dirs;
}

void KernelLauncher::appendWisdomDirsFromEnv() {
  const char *value = std::getenv("KERNEL_LAUNCHER_WISDOM");
  if (value == nullptr) {
    return;
  }

  std::string dir;
  for (const char ch : std::string(value) + ':') {
    if (ch == ':' || ch == ';' || ch == ',') {
      if (!dir.empty()) {
        wisdomDirsStorage().push_back(std::move(dir));
        dir.clear();
      }
    } else {
      dir += ch;
    }
  }
}

std::string &KernelLauncher::captureDirStorage() {
  static std::string dir;
  return dir;
}


const char *KernelLauncher::onlineWisdomDir() { 
    if (const char *value = std::getenv("KERNEL_LAUNCHER_WISDOM")) {
    return value;
  }
  return KERNELWRAPPERS_TUNE_WISDOM_DIRECTORY;
}

bool KernelLauncher::runPythonTuner(const std::filesystem::path &capture) {
  const pid_t pid = fork();
  if (pid == -1) {
    return false;
  }
  if (pid == 0) {
    setenv("KERNELWRAPPERS_KERNEL_LAUNCHER_PYTHON",
           KERNELWRAPPERS_TUNE_KERNEL_LAUNCHER_PYTHON, 1);
    execl(KERNELWRAPPERS_TUNE_PYTHON, KERNELWRAPPERS_TUNE_PYTHON,
          KERNELWRAPPERS_TUNE_SCRIPT, "--strategy", "random", "--force",
          "--output", onlineWisdomDir(), capture.c_str(),
          static_cast<char *>(nullptr));
    _exit(127);
  }

  int status = 0;
  return waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
         WEXITSTATUS(status) == 0;
}

bool KernelLauncher::restoreArrayInputs(
    const std::vector<kernel_launcher::KernelArg> &args,
    const std::vector<std::vector<uint8_t>> &inputs) {
    size_t input = 0;
  for (const kernel_launcher::KernelArg &arg : args) {
    if (!arg.is_array()) {
      continue;
    }

    void *ptr = nullptr;
    std::memcpy(&ptr, arg.as_void_ptr(), sizeof(ptr));
    const std::vector<uint8_t> &bytes = inputs.at(input++);
    if (cuMemcpy(reinterpret_cast<CUdeviceptr>(ptr),
                 reinterpret_cast<CUdeviceptr>(bytes.data()),
                 bytes.size()) != CUDA_SUCCESS) {
      return false;
    }
  }
  return true;
}

void KernelLauncher::configureTuning(JITKernel &kernel) {
  kernel.configureTuning();
}


void KernelLauncher::addWisdomDirectory(const std::string &dir) {
  kernel_launcher::append_global_wisdom_directory(dir);
  wisdomDirsStorage().push_back(dir);
  Log::info("kernelwrappers wisdom directory added: " + dir);
}

bool KernelLauncher::onlineTuningRequested() {
  if (const char *value = std::getenv("KERNELWRAPPERS_TUNE_KERNELS")) {
    return *value != '\0' && std::string(value) != "0";
  }
  return false;
} 

void KernelLauncher::initializeWisdomDefaults() {
  static const bool initialized = []() {
    if (captureDirStorage().empty()) {
      if (onlineTuningRequested()) {
        const auto dir = std::filesystem::temp_directory_path() /
                         ("kernelwrappers-tune-" + std::to_string(getpid()));
        std::filesystem::create_directories(dir);
        kernel_launcher::set_global_capture_directory(dir.string());
        captureDirStorage() = dir.string();
      }
    }

    // Always search the application-independent local wisdom store.
    const char *wisdomDir = defaultWisdomDir();
    appendWisdomDirsFromEnv();
    if (wisdomDir != nullptr && *wisdomDir != '\0') {
      kernel_launcher::append_global_wisdom_directory(wisdomDir);
      wisdomDirsStorage().emplace_back(wisdomDir);
    }

    return true;
  }();

  (void)initialized;
}

kernel_launcher::KernelBuilder JITKernel::buildKernelBuilder() const {
  return kernel_launcher::build_pragma_kernel(
      getName(), kernel_launcher::KernelSource(getName() + ".cu", itsSource),
      itsTemplateArgs);
}

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_TUNER

