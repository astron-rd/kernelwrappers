#ifndef KERNELWRAPPERS_JITKERNEL_H
#define KERNELWRAPPERS_JITKERNEL_H

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>
#include <kernelwrappers/Kernel.h>
#include <kernelwrappers/KernelCacheHash.h>
#include <kernelwrappers/KernelConfig.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifdef KERNELWRAPPERS_TUNER
#include <kernel_launcher/arg.h>
#include <kernel_launcher/builder.h>
#include <kernel_launcher/config.h>
#include <kernel_launcher/cuda.h>
#include <kernel_launcher/utils.h>
#include <kernel_launcher/value.h>
#include <kernel_launcher/wisdom.h>
#endif

namespace kernelwrappers {

class JITKernel : public Kernel {
#ifdef KERNELWRAPPERS_TUNER
public:
  // Append a directory to search for wisdom files.
  static void addWisdomDirectory(const std::string &dir);
#endif

protected:
  JITKernel(cu::Device &device, cu::Stream &stream, const std::string &source,
            const std::string &name, const CompileFlags &flags,
            const CompileDefinitions &definitions, bool cachable = true,
            bool tunable = false, bool verbose = false
#ifdef KERNELWRAPPERS_TUNER
            ,
            const std::vector<kernel_launcher::Value> &templateArgs = {}
#endif
  );

  void setArg(size_t index, const cu::DeviceMemory &mem);

  void setArg(size_t index, const void **val);

  template <typename T> void setArg(size_t index, const T &val) {
    doSetArg(index, &val);
  }

  void setEnqueueWorkSizes(cu::Device &device, Grid gridSize, Block blockSize,
                           std::unique_ptr<std::string> errorStrings = nullptr);

  virtual CompileFlags compileFlags() const;

  virtual CompileDefinitions compileDefinitions() const;

public:
  // Compile the kernel source (if not already compiled) so it can be launched.
  // When \a force is true, the kernel is recompiled even if it was already
  // compiled. Compilation is a no-op if the kernel is already compiled.
  void compile(const bool force = false);

  // Launch the kernel on the GPU, compiling it first if necessary.
  void launch() override;

  const cu::Module &getModule() const { return *itsModule; }
  const std::string &getName() const { return itsName; }
  const CompileFlags &getCompileFlags() const { return itsFlags; }
  const CompileDefinitions &getCompileDefinitions() const {
    return itsDefinitions;
  }

protected:
  void setTunable(bool tunable) { tunable_ = tunable; }

#ifdef KERNELWRAPPERS_TUNER
protected:
  // The problem size used to select a wisdom entry and to label captures.
  virtual kernel_launcher::ProblemSize problemSize() const { return {}; }

  // Kernel arguments (device pointers and scalar values) used for capture.
  virtual std::vector<kernel_launcher::KernelArg> createTunerInput() const {
    return {};
  }

  // The configuration loaded by tune(). Only valid after tune().
  const std::optional<kernel_launcher::Config> &getTunedConfig() const {
    return itsBestConfig;
  }
#endif

protected:
  // Hook to configure the kernel launch (grid/block sizes, ...) once the
  // module is available. Called by compile().
  virtual void configureLaunch() {}

private:
  static CompileFlags insertCompileFlags(cu::Device &device,
                                         CompileFlags flags) {
    flags.emplace("--gpu-architecture=" + device.getArch());
    return flags;
  }

  nvrtc::Program itsProgram;

  const std::string itsName;
  const CompileFlags itsFlags;
  const CompileDefinitions itsDefinitions;

#ifdef KERNELWRAPPERS_TUNER
  const std::string itsSource;
  const std::vector<kernel_launcher::Value> itsTemplateArgs;
#endif

  KernelCacheHash itsSourceHash;

  // Module for the kernel
  std::unique_ptr<cu::Module> itsModule;

  // Function for kernel execution
  std::unique_ptr<cu::Function> itsFunction;

  // Function arguments as set.
  std::vector<const void *> itsKernelArgs;

  // Helper function to modify itsKernelArgs.
  void doSetArg(size_t index, const void *argp);

  // The grid of blocks dimensions for kernel execution.
  Grid itsGridDims;

  // The block of threads dimensions for kernel execution.
  Block itsBlockDims;

  // Print resource usage and launch configuration.
  bool verbose_;

  // Whether this kernel is cachable or not.
  bool cachable_;

  // Whether this kernel is tunable or not.
  bool tunable_;

  // Number of threads NVRTC should use for parallel PTX compilation.
  constexpr static int kNrParallelCompilationsThreads = 8;

#ifdef KERNELWRAPPERS_TUNER
  // Best known configuration, loaded by tune() from a wisdom file (or the
  // default configuration of the kernel's configuration space).
  std::optional<kernel_launcher::Config> itsBestConfig;

  // Wisdom is selected once, before the first JIT compilation.
  bool itsTuningConfigured = false;
  bool itsOnlineTuning = false;
  bool itsOnlineTuningAttempted = false;

  kernel_launcher::KernelBuilder buildKernelBuilder() const;
  void configureTuning();

  // Run the current launch with capture: copy inputs, launch, copy outputs and
  // export a tuning capture.
  bool captureLaunch();

  // Initialize wisdom directories and the temporary online capture directory.
  static void initializeWisdomDefaults();
#endif
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_JITKERNEL_H
