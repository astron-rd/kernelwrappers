#ifndef KERNELWRAPPERS_KERNELLAUNCHER_H
#define KERNELWRAPPERS_KERNELLAUNCHER_H

#include <filesystem>
#include <string>
#include <vector>

#ifdef KERNELWRAPPERS_TUNER
#include <kernel_launcher/arg.h>
#include <kernel_launcher/builder.h>
#include <kernel_launcher/config.h>
#include <kernel_launcher/cuda.h>
#include <kernel_launcher/export.h>
#include <kernel_launcher/pragma.h>
#include <kernel_launcher/utils.h>
#include <kernel_launcher/value.h>
#include <kernel_launcher/wisdom.h>
#endif

namespace kernelwrappers {

class JITKernel;

class KernelLauncher {
public:
  static void launch(JITKernel &kernel);

#ifdef KERNELWRAPPERS_TUNER
  static void addWisdomDirectory(const std::string &dir);
  static void initializeWisdomDefaults();
  static void configureTuning(JITKernel &kernel);
  static std::string &captureDirStorage();
  static std::vector<std::string> &wisdomDirsStorage();
  static const char *onlineWisdomDir();
  static bool runPythonTuner(const std::filesystem::path &capture);
  static bool restoreArrayInputs(
      const std::vector<kernel_launcher::KernelArg> &args,
      const std::vector<std::vector<uint8_t>> &inputs);
  static bool onlineTuningRequested();
private:
  static const char *defaultWisdomDir();
  static void appendWisdomDirsFromEnv();

#endif
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_KERNELLAUNCHER_H
