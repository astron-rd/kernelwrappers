#include "ExampleProcessingStep.h"

ExampleProcessingStep::ExampleProcessingStep(cu::Device &device,
                                             cu::Context &context,
                                             cu::Stream &stream,
                                             const cu::DeviceMemory &input,
                                             const cu::DeviceMemory &output,
                                             size_t nrElements)
    : context(context), scaleKernel(device, stream, input, output, nrElements),
      offsetKernel(device, stream, output, nrElements) {
  compilationQueue.add(scaleKernel);
  compilationQueue.add(offsetKernel);
}

void ExampleProcessingStep::compileKernels() {
  compilationQueue.compileAll(context);
}

void ExampleProcessingStep::process(float scale, float offset) {
  scaleKernel.run(scale);
  offsetKernel.run(offset);
}

#ifdef KERNELWRAPPERS_TUNER

size_t ExampleProcessingStep::tuningNrElements() { return 100000000; }

#endif // KERNELWRAPPERS_TUNER
