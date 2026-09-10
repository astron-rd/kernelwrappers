#ifndef KERNELWRAPPERS_EXAMPLE_SCALEKERNEL_H
#define KERNELWRAPPERS_EXAMPLE_SCALEKERNEL_H

#include <cstddef>
#include <cudawrappers/cu.hpp>
#include <kernelwrappers/JITKernel.h>
#include <string>

extern const std::string ScaleKernel_source;

class ScaleKernel : public kernelwrappers::JITKernel {
public:
  ScaleKernel(cu::Device &device, cu::Stream &stream,
              const cu::DeviceMemory &input, const cu::DeviceMemory &output,
              size_t nrElements);

  void run(float factor);
protected:
  kernelwrappers::CompileFlags compileFlags() const override;

  void configureLaunch() override;

#ifdef KERNELWRAPPERS_TUNER
  kernel_launcher::ProblemSize problemSize() const override;

  std::vector<kernel_launcher::KernelArg> createTunerInput() const override;
#endif

private:
  const cu::DeviceMemory &input_;
  const cu::DeviceMemory &output_;
  size_t nrElements_;
  float factor_ = 0.0f;
};

#endif // KERNELWRAPPERS_EXAMPLE_SCALEKERNEL_H
