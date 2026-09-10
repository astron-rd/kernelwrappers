#ifndef KERNELWRAPPERS_EXAMPLE_OFFSETKERNEL_H
#define KERNELWRAPPERS_EXAMPLE_OFFSETKERNEL_H

#include <cstddef>
#include <cudawrappers/cu.hpp>
#include <kernelwrappers/JITKernel.h>
#include <string>

extern const std::string OffsetKernel_source;

class OffsetKernel : public kernelwrappers::JITKernel {
public:
  OffsetKernel(cu::Device &device, cu::Stream &stream,
               const cu::DeviceMemory &output, size_t nrElements);

  void run(float offset);

protected:
  kernelwrappers::CompileFlags compileFlags() const override;

  void configureLaunch() override;

#ifdef KERNELWRAPPERS_TUNER
  kernel_launcher::ProblemSize problemSize() const override;

  std::vector<kernel_launcher::KernelArg> createTunerInput() const override;
#endif

private:
  const cu::DeviceMemory &output_;
  size_t nrElements_;
  float offset_ = 0.0f;
};

#endif // KERNELWRAPPERS_EXAMPLE_OFFSETKERNEL_H
