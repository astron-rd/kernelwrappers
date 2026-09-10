#ifndef KERNELWRAPPERS_EXAMPLE_EXAMPLEPROCESSINGSTEP_H
#define KERNELWRAPPERS_EXAMPLE_EXAMPLEPROCESSINGSTEP_H

#include <cstddef>
#include <cudawrappers/cu.hpp>
#include <kernelwrappers/KernelCompilationQueue.h>

#include "OffsetKernel.h"
#include "ScaleKernel.h"

/*
 * Example processing step: applies out = in * scale + offset.
 *
 */
class ExampleProcessingStep {
public:
  ExampleProcessingStep(cu::Device &device, cu::Context &context,
                        cu::Stream &stream, const cu::DeviceMemory &input,
                        const cu::DeviceMemory &output, size_t nrElements);

  void compileKernels();

  void process(float scale, float offset);

#ifdef KERNELWRAPPERS_TUNER
  static size_t tuningNrElements();
#endif

private:
  kernelwrappers::KernelCompilationQueue compilationQueue;
  cu::Context &context;
  ScaleKernel scaleKernel;
  OffsetKernel offsetKernel;
};

#endif // KERNELWRAPPERS_EXAMPLE_EXAMPLEPROCESSINGSTEP_H
