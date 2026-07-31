#ifndef KERNELWRAPPERS_JITKERNEL_H
#define KERNELWRAPPERS_JITKERNEL_H

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>
#include <kernelwrappers/Kernel.h>
#include <kernelwrappers/KernelCacheHash.h>
#include <kernelwrappers/KernelConfig.h>
#include <memory>
#include <string>
#include <vector>

namespace kernelwrappers {

/*
 * A kernel that is an actual CUDA JIT-compiled function.
 *
 * The kernel source is compiled at runtime with NVRTC (unless a matching
 * compiled kernel is found in the KernelCache), after which it can be launched
 * on the associated device and stream.
 */
class JITKernel : public Kernel {
protected:
  // Construct a kernel.
  JITKernel(cu::Device &device, cu::Stream &stream, const std::string &source,
            const std::string &name, const CompileFlags &flags,
            const CompileDefinitions &definitions, bool cachable = true,
            bool verbose = false);

  // Set kernel DeviceMemory object argument number \a index to \a mem.
  // \a mem must outlive kernel execution.
  void setArg(size_t index, const cu::DeviceMemory &mem);

  // Set pointer to kernel device memory object (as void *) number \a index
  // to \a val. \a *val must outlive kernel execution.
  // Note: Prefer to use setArg() passing a DeviceMemory ref over this overload.
  void setArg(size_t index, const void **val);

  // Set kernel immediate argument number \a index to \a val.
  // \a val must outlive kernel execution.
  // Not for device memory objects (be it as DeviceMemory or as void *).
  template <typename T> void setArg(size_t index, const T &val) {
    doSetArg(index, &val);
  }

  // Set the passed execution configuration if supported on the hardware
  // in the stream for this kernel.
  // If not supported and nullptr was passed in errorStrings, an exc is thrown.
  // If not supported and errorStrings is valid, an error string is written
  // to the errorStrings pointer.
  void setEnqueueWorkSizes(cu::Device &device, Grid gridSize, Block blockSize,
                           std::string *errorStrings = nullptr);

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

  // Number of threads NVRTC should use for parallel PTX compilation.
  constexpr static int kNrParallelCompilationsThreads = 8;
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_JITKERNEL_H
