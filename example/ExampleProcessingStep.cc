#include "ExampleProcessingStep.h"

ExampleProcessingStep::ExampleProcessingStep(cu::Device &device,
                                             cu::Context &context,
                                             cu::Stream &stream,
                                             const cu::DeviceMemory &input,
                                             const cu::DeviceMemory &output,
                                             size_t nrElements)
    : context(context),
      scaleKernel(device, stream, "scale_kernel", input, output, nrElements),
      offsetKernel(device, stream, "offset_kernel", input, output, nrElements) {
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
