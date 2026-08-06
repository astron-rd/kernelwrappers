#include <chrono>
#include <cudawrappers/nvrtc.hpp>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelCache.h>
#include <kernelwrappers/Log.h>
#include <sstream>
#include <stdexcept>

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

} // namespace

namespace kernelwrappers {

JITKernel::JITKernel(cu::Device &device, cu::Stream &stream,
                     const std::string &source, const std::string &name,
                     const CompileFlags &flags,
                     const CompileDefinitions &definitions, bool cachable,
                     bool verbose)
    : Kernel(device, stream, name), itsProgram(source, name), itsName(name),
      itsFlags(insertCompileFlags(device, flags)), itsDefinitions(definitions),
      verbose_(verbose), cachable_(cachable) {
  if (cachable_) {
    itsSourceHash = KernelCache::computeHash(source);
  }
}

void JITKernel::compile(const bool force) {
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
  // Ensure that a compiled kernel exists before launch.
  // If a compiled kernel is already available, for instance when the user
  // called compile() explicitly, this will be a no-op.
  compile();

  itsStream.launchKernel(*itsFunction, itsGridDims.x, itsGridDims.y,
                         itsGridDims.z, itsBlockDims.x, itsBlockDims.y,
                         itsBlockDims.z, 0, itsKernelArgs);
}

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
