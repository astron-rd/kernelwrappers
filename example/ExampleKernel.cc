#include "ExampleKernel.h"

// Defines ExampleProcessingStep_source from the embedded .cu file.
#include <example/kernels/ExampleProcessingStep.cu.o.h>

/*
 * Returns ceil(n/divisor).
 */
template <typename T> inline T ceilDiv(T n, T divisor) {
  return (n + divisor - 1) / divisor;
}

ExampleKernel::ExampleKernel(cu::Device &device, cu::Stream &stream,
                             const std::string &functionName,
                             const cu::DeviceMemory &input,
                             const cu::DeviceMemory &output, size_t nrElements)
    : JITKernel(device, stream, ExampleProcessingStep_source, functionName,
                compileFlags(), compileDefinitions()),
      input_(input), output_(output), nrElements_(nrElements) {}

kernelwrappers::CompileFlags ExampleKernel::compileFlags() const {
  kernelwrappers::CompileFlags flags = JITKernel::compileFlags();
  flags.insert("--device-as-default-execution-space");
  return flags;
}

kernelwrappers::CompileDefinitions ExampleKernel::compileDefinitions() const {
  kernelwrappers::CompileDefinitions defs = JITKernel::compileDefinitions();
  defs["NR_ELEMENTS"] = std::to_string(nrElements_);
  return defs;
}

void ExampleKernel::configureLaunch() {
  constexpr unsigned kThreadsPerBlock = 256;
  kernelwrappers::Grid grid(ceilDiv(nrElements_, size_t(kThreadsPerBlock)));
  kernelwrappers::Block block(kThreadsPerBlock);
  setEnqueueWorkSizes(itsDevice, grid, block);
}

void ExampleKernel::run(float parameter) {
  setArg(0, input_);
  setArg(1, output_);
  setArg(2, parameter);
  launch();
}
