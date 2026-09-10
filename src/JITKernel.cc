#include <chrono>
#include <cstdlib>
#include <cudawrappers/nvrtc.hpp>
#include <filesystem>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelCache.h>
#include <kernelwrappers/Log.h>
#include <sstream>
#include <stdexcept>

#ifdef KERNELWRAPPERS_TUNER
#include <kernel_launcher/export.h>
#include <kernel_launcher/pragma.h>
#include <kernel_launcher/wisdom.h>
#endif

namespace {

// Compile the kernel source code for the target device and return PTX.
std::string compilePTX(nvrtc::Program &program, const std::string &name,
                       const kernelwrappers::CompileFlags &flags,
                       const kernelwrappers::CompileDefinitions &definitions) {
  std::vector<std::string> options;

  for (const std::string &include : nvrtc::findIncludePaths()) {
    options.emplace_back("-I" + include);
  }

  for (const std::string &flag : flags) {
    options.emplace_back(flag);
  }

  for (const auto &[key, value] : definitions) {
    options.emplace_back("-D" + key + "=" + value);
  }

  kernelwrappers::Log::debug("Starting runtime compilation of " + name);
  for (const auto &option : options) {
    kernelwrappers::Log::debug("\t" + option);
  }

  auto start = std::chrono::high_resolution_clock::now();

  try {
    program.compile(options);
  } catch (nvrtc::Error &error) {
    // Print compilation log (if any)
    const std::string log = program.getLog();
    if (log.size()) {
      kernelwrappers::Log::error("Compilation log:\n" + log);
    }

    throw error;
  }

  auto end = std::chrono::high_resolution_clock::now();
  const size_t time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();

  if (time > 1e3) {
    kernelwrappers::Log::debug("Build time: " + std::to_string(time * 1e-3) +
                               " ms");
  } else {
    kernelwrappers::Log::debug("Build time: " + std::to_string(time) + " us");
  }

  return program.getPTX();
}

#ifdef KERNELWRAPPERS_TUNER

const char *defaultCaptureDir() {
#ifdef KERNELWRAPPERS_CAPTURE_PATH
  return KERNELWRAPPERS_CAPTURE_PATH;
#else
  return nullptr;
#endif
}

const char *defaultWisdomDir() {
#ifdef KERNELWRAPPERS_WISDOM_PATH
  return KERNELWRAPPERS_WISDOM_PATH;
#else
  return nullptr;
#endif
}

std::string captureDirFromEnv() {
  if (const char *value = std::getenv("KERNEL_LAUNCHER_CAPTURE_DIR")) {
    return value;
  }
  if (const char *value = std::getenv("KERNEL_LAUNCHER_DIR")) {
    return value;
  }
  return {};
}

// The capture directory used by JITKernel, resolved lazily from the
// KERNELWRAPPERS_CAPTURE_PATH build default, the kernel_launcher environment
// variables, or an explicit setCaptureDirectory() call.
std::string &captureDirStorage() {
  static std::string dir;
  return dir;
}

std::vector<std::string> &wisdomDirsStorage() {
  static std::vector<std::string> dirs = {"."};
  return dirs;
}

void appendWisdomDirsFromEnv() {
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

bool captureRequested() {
  constexpr const char *keys[] = {
      "KERNEL_LAUNCHER_CAPTURE_FORCE", "KERNEL_LAUNCHER_CAPTURE",
      "KERNEL_LAUNCHER_FORCE_CAPTURE", "KERNEL_LAUNCHER_TUNE_FORCE",
      "KERNEL_LAUNCHER_FORCE_TUNE",    "KERNEL_LAUNCHER_TUNE"};
  for (const char *key : keys) {
    if (const char *value = std::getenv(key);
        value != nullptr && *value != '\0') {
      return true;
    }
  }
  return false;
}

#endif // KERNELWRAPPERS_TUNER

} // namespace

namespace kernelwrappers {

#ifdef KERNELWRAPPERS_TUNER

void JITKernel::setCaptureDirectory(const std::string &dir) {
  kernel_launcher::set_global_capture_directory(dir);
  captureDirStorage() = dir;
  Log::info("kernelwrappers capture directory: " + dir);
}

void JITKernel::addWisdomDirectory(const std::string &dir) {
  kernel_launcher::append_global_wisdom_directory(dir);
  wisdomDirsStorage().push_back(dir);
  Log::info("kernelwrappers wisdom directory added: " + dir);
}

const std::string &JITKernel::captureDirectory() { return captureDirStorage(); }

void JITKernel::initializeWisdomDefaults() {
  static const bool initialized = []() {
    if (captureDirStorage().empty()) {
      // Respect explicitly configured environment variables.
      const std::string envDir = captureDirFromEnv();
      if (!envDir.empty()) {
        captureDirStorage() = envDir;
      } else {
        // Fall back to the capture directory compiled into kernelwrappers.
        const char *dir = defaultCaptureDir();
        if (dir != nullptr && *dir != '\0') {
          kernel_launcher::set_global_capture_directory(dir);
          captureDirStorage() = dir;
        }
      }
    }

    // Always search the application-independent local wisdom store. Users
    // can prepend project-specific locations with addWisdomDirectory().
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

#endif // KERNELWRAPPERS_TUNER

JITKernel::JITKernel(cu::Device &device, cu::Stream &stream,
                     const std::string &source, const std::string &name,
                     const CompileFlags &flags,
                     const CompileDefinitions &definitions, bool cachable,
                     bool tunable, bool verbose
#ifdef KERNELWRAPPERS_TUNER
                     ,
                     const std::vector<kernel_launcher::Value> &templateArgs
#endif
                     )
    : Kernel(device, stream, name), itsProgram(source, name), itsName(name),
      itsFlags(insertCompileFlags(device, flags)), itsDefinitions(definitions),
      verbose_(verbose), cachable_(cachable), tunable_(tunable)
#ifdef KERNELWRAPPERS_TUNER
      ,
      itsSource(source), itsTemplateArgs(templateArgs)
#endif
{
  if (cachable_) {
    itsSourceHash = KernelCache::computeHash(source);
  }

#ifdef KERNELWRAPPERS_TUNER
  initializeWisdomDefaults();
#endif
}

void JITKernel::compile(const bool force) {
#ifdef KERNELWRAPPERS_TUNER
  if (!itsTuningConfigured) {
    configureTuning();
  }
#endif

  const bool useKernelCache = KernelCache::isEnabled() && cachable_;

  if (force || !itsModule || !itsFunction) {
    std::string ptx;
    if (useKernelCache && KernelCache::readKernelFromCache(*this, itsSourceHash,
                                                           itsDevice, ptx)) {
      Log::info("Using cached kernel for " + itsName);
    } else {
      ptx = compilePTX(itsProgram, itsName, itsFlags, itsDefinitions);
      if (useKernelCache) {
        Log::info("Writing compiled kernel " + itsName + " to cache");
        KernelCache::writeKernelToCache(*this, itsSourceHash, itsDevice, ptx);
      }
    }

    itsModule =
        std::make_unique<cu::Module>(static_cast<const void *>(ptx.data()));
    itsFunction = std::make_unique<cu::Function>(*itsModule, itsName.c_str());

    if (verbose_) {
      Log::info("Function " + std::string(itsFunction->name()) + ":\n" +
                "  nr. of registers used : " +
                std::to_string(
                    itsFunction->getAttribute(CU_FUNC_ATTRIBUTE_NUM_REGS)) +
                "\n  nr. of bytes of shared memory used (static) : " +
                std::to_string(itsFunction->getAttribute(
                    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)));
    }

    configureLaunch();
  }
}

void JITKernel::launch() {
#ifdef KERNELWRAPPERS_TUNER
  compile();

  if (itsCapturePending) {
    if (itsCaptureSkip > 0) {
      --itsCaptureSkip;
    } else {
      if (tunable_) {
        captureLaunch();
      } else {
        Log::warn("Kernel " + itsName +
                  " is not marked as tunable, but a capture was requested");
      }
      return;
    }
  }
#else
  // Ensure that a compiled kernel exists before launch.
  // If a compiled kernel is already available, for instance when the user
  // called compile() explicitly, this will be a no-op.
  compile();
#endif

  itsStream.launchKernel(*itsFunction, itsGridDims.x, itsGridDims.y,
                         itsGridDims.z, itsBlockDims.x, itsBlockDims.y,
                         itsBlockDims.z, 0, itsKernelArgs);
}

#ifdef KERNELWRAPPERS_TUNER

kernel_launcher::KernelBuilder JITKernel::buildKernelBuilder() const {
  return kernel_launcher::build_pragma_kernel(
      getName(), kernel_launcher::KernelSource(getName() + ".cu", itsSource),
      itsTemplateArgs);
}

void JITKernel::configureTuning() {
  itsBestConfig.reset();
  itsCapturePending = false;
  itsCaptureSkip = -1;

  // Plain JIT kernels remain supported. Only pragma-annotated sources have a
  // Kernel Launcher configuration space and can have matching wisdom.
  if (itsSource.find("#pragma kernel") == std::string::npos) {
    itsTuningConfigured = true;
    return;
  }

  const kernel_launcher::KernelBuilder builder = buildKernelBuilder();
  const kernel_launcher::ProblemSize size = problemSize();

  int captureSkip = -1;
  const auto device = kernel_launcher::CudaDevice::current();
  if (captureRequested()) {
    const kernel_launcher::WisdomSettings settings =
        kernel_launcher::default_wisdom_settings();
    settings.load_config(builder.tuning_key(), builder, size, device,
                         &captureSkip);
  }

  // Later additions are explicit overrides, including the default local
  // wisdom store appended during initialization.
  for (auto it = wisdomDirsStorage().rbegin(); it != wisdomDirsStorage().rend();
       ++it) {
    kernel_launcher::WisdomResult result;
    kernel_launcher::Config config = kernel_launcher::load_best_config(
        *it, builder.tuning_key(), builder, device.name(), device.arch(), size,
        &result);
    std::cout << "Found wisdom for " << builder.tuning_key() << std::endl;
    if (result == kernel_launcher::WisdomResult::Ok) {
      itsBestConfig.emplace(std::move(config));
      break;
    }
  }

  itsCapturePending = captureSkip >= 0;
  itsCaptureSkip = captureSkip;
  if (itsCapturePending) {
    Log::info("Kernel " + itsName + " will trigger a capture for tuning");
  }
  itsTuningConfigured = true;
}

void JITKernel::captureLaunch() {
  itsCapturePending = false;

  const kernel_launcher::WisdomSettings settings =
      kernel_launcher::default_wisdom_settings();
  const kernel_launcher::KernelBuilder builder = buildKernelBuilder();
  const kernel_launcher::ProblemSize size = problemSize();
  const std::vector<kernel_launcher::KernelArg> args = createTunerInput();

  // Capture the input buffers in their current state before launching.
  std::vector<std::vector<uint8_t>> inputs;
  std::vector<std::vector<uint8_t>> outputs;
  try {
    itsStream.synchronize();
    for (const kernel_launcher::KernelArg &arg : args) {
      if (arg.is_array()) {
        inputs.emplace_back(arg.copy_array());
      }
    }
  } catch (const std::exception &err) {
    Log::warn("Failed to capture inputs of kernel " + itsName + ": " +
              err.what());
    return;
  }

  itsStream.launchKernel(*itsFunction, itsGridDims.x, itsGridDims.y,
                         itsGridDims.z, itsBlockDims.x, itsBlockDims.y,
                         itsBlockDims.z, 0, itsKernelArgs);

  try {
    itsStream.synchronize();
    for (const kernel_launcher::KernelArg &arg : args) {
      if (arg.is_array()) {
        outputs.emplace_back(arg.copy_array());
      }
    }
  } catch (const std::exception &err) {
    Log::warn("Failed to capture outputs of kernel " + itsName + ": " +
              err.what());
    return;
  }

  try {
    // kernel_launcher's write_file does not create missing directories, so
    // make sure the capture directory exists before exporting.
    if (!captureDirStorage().empty()) {
      std::filesystem::create_directories(captureDirStorage());
    }

    settings.capture_kernel(builder.tuning_key(), builder, size, args, inputs,
                            outputs);
    Log::info("Captured kernel launch of " + itsName + " for tuning");
  } catch (const std::exception &err) {
    Log::warn("Failed to write tuning capture for kernel " + itsName + ": " +
              err.what());
  }
}

#endif // KERNELWRAPPERS_TUNER

void JITKernel::setArg(size_t index, const cu::DeviceMemory &mem) {
  doSetArg(index, mem.parameter());
}

void JITKernel::setArg(size_t index, const void **val) {
  doSetArg(index, (const void *)val);
}

void JITKernel::doSetArg(size_t index, const void *argp) {
  if (index >= itsKernelArgs.size()) {
    itsKernelArgs.resize(index + 1);
  }
  itsKernelArgs[index] = argp;
}

void JITKernel::setEnqueueWorkSizes(cu::Device &device, Grid gridSize,
                                    Block blockSize,
                                    std::unique_ptr<std::string> errorStrings) {
  std::ostringstream errMsgs;

  // Enforce by the hardware supported work sizes to see errors early
  Block maxBlockSize = getMaxBlockDims(device);
  if (blockSize.x > maxBlockSize.x || blockSize.y > maxBlockSize.y ||
      blockSize.z > maxBlockSize.z) {
    errMsgs << "  - blockSize must be at most " << maxBlockSize << std::endl;
  }

  const int maxThreadsPerBlock =
      device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  if (blockSize.x * blockSize.y * blockSize.z > maxThreadsPerBlock) {
    errMsgs << "  - blockSize total must be at most " << maxThreadsPerBlock
            << " threads/block" << std::endl;
  }

  // gridSize may (in theory) be all zero (no work), so allow.
  // Do reject an all zero blockSize. We need to mod or div by it.
  if (blockSize.x == 0 || blockSize.y == 0 || blockSize.z == 0) {
    errMsgs << "  - blockSize dimensions must be non-zero" << std::endl;
  } else {
    Grid maxGridWorkSize = getMaxGridDims(device);
    if (gridSize.x > maxGridWorkSize.x || gridSize.y > maxGridWorkSize.y ||
        gridSize.z > maxGridWorkSize.z)
      errMsgs << "  - gridSize must be at most " << maxGridWorkSize
              << std::endl;
  }

  // On error, store error messages in output parameter, or throw.
  std::string errStr(errMsgs.str());
  if (!errStr.empty()) {
    if (errorStrings) {
      *errorStrings = errStr;
    } else {
      std::stringstream message;
      message << "setEnqueueWorkSizes(): unsupported gridSize " << gridSize
              << " and/or blockSize " << blockSize << " selected:" << std::endl
              << errStr;
      throw std::runtime_error(message.str());
    }
  }

  if (verbose_) {
    std::ostringstream ss;
    ss << "  grid size: " << gridSize;
    Log::info(ss.str());
    ss.str("");
    ss << "  block size: " << blockSize;
    Log::info(ss.str());
  }

  itsGridDims = gridSize;
  itsBlockDims = blockSize;
}

CompileFlags JITKernel::compileFlags() const {
  CompileFlags flags;
  flags.insert("--restrict");
  flags.insert("-lineinfo");
  flags.insert("-rdc=true");
  flags.insert("--split-compile=" +
               std::to_string(kNrParallelCompilationsThreads));
  return flags;
};

CompileDefinitions JITKernel::compileDefinitions() const {
  CompileDefinitions definitions;
  return definitions;
}

} // namespace kernelwrappers
