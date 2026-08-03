#ifndef KERNELWRAPPERS_EXAMPLE_EXAMPLEKERNEL_H
#define KERNELWRAPPERS_EXAMPLE_EXAMPLEKERNEL_H

#include <cstddef>
#include <cudawrappers/cu.hpp>
#include <kernelwrappers/JITKernel.h>
#include <string>

extern const std::string ExampleProcessingStep_source;

class ExampleKernel : public kernelwrappers::JITKernel {
public:
  ExampleKernel(cu::Device &device, cu::Stream &stream,
                const std::string &functionName, const cu::DeviceMemory &input,
                const cu::DeviceMemory &output, size_t nrElements);

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
