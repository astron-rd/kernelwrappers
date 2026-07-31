#ifndef KERNELWRAPPERS_EXAMPLE_EXAMPLEKERNEL_H
#define KERNELWRAPPERS_EXAMPLE_EXAMPLEKERNEL_H

#include <cstddef>
#include <cudawrappers/cu.hpp>
#include <kernelwrappers/JITKernel.h>
#include <string>

/*
 * CUDA source of the kernels of this step, embedded into the binary by
 * target_embed_source (see example/CMakeLists.txt). The generated header
 * defines this string; it is declared extern here so that only one
 * translation unit includes that generated header.
 */
extern const std::string ExampleProcessingStep_source;

/*
 * A single JIT-compiled kernel of the step.
 *
 * The kernel function name is passed to the constructor, so one class can
 * wrap every kernel that shares the same source, compile flags/definitions
 * and launch configuration.
 */
class ExampleKernel : public kernelwrappers::JITKernel {
public:
  ExampleKernel(cu::Device &device, cu::Stream &stream,
                const std::string &functionName, const cu::DeviceMemory &input,
                const cu::DeviceMemory &output, size_t nrElements);

  // Set the kernel arguments and launch the kernel.
  void run(float parameter);

protected:
  kernelwrappers::CompileFlags compileFlags() const override;

  kernelwrappers::CompileDefinitions compileDefinitions() const override;

  void configureLaunch() override;

private:
  const cu::DeviceMemory &input_;
  const cu::DeviceMemory &output_;
  size_t nrElements_;
};

#endif // KERNELWRAPPERS_EXAMPLE_EXAMPLEKERNEL_H
