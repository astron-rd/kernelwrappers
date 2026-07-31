#include <cstdlib>
#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>
#include <iostream>
#include <kernelwrappers/Log.h>
#include <string_view>
#include <vector>

#include "ExampleProcessingStep.h"

namespace {

void verifyResults(const std::vector<float> &result, size_t nrElements) {
  for (size_t i = 0; i < nrElements; ++i) {
    // The step computes output = input * scale + offset, with input[i] == i,
    // scale == 2.0f and offset == 10.0f.
    const float expected = static_cast<float>(i) * 2.0f + 10.0f;
    if (result[i] != expected) {
      std::cerr << "Verification failed at index " << i << ": expected "
                << expected << ", got " << result[i] << std::endl;
      std::exit(1);
    }
  }
  std::cout << "Output verified OK" << std::endl;
}

} // namespace

int main() {
  // Integrate the library's logging with your own logger.
  kernelwrappers::Log::setCallback(
      [](kernelwrappers::Log::Level, std::string_view message) {
        std::cout << "[kernelwrappers] " << message << std::endl;
      });
  kernelwrappers::Log::setLevel(kernelwrappers::Log::Level::Debug);

  try {
    cu::Device device(0);
    cu::Context context(0, device);
    context.setCurrent();

    cu::Stream stream;

    const size_t nrElements = 64;
    const size_t bytes = nrElements * sizeof(float);

    std::vector<float> hostInput(nrElements);
    for (size_t i = 0; i < nrElements; ++i) {
      hostInput[i] = static_cast<float>(i);
    }

    cu::DeviceMemory input(bytes);
    cu::DeviceMemory output(bytes);
    stream.memcpyHtoDAsync(input, hostInput.data(), bytes);

    // The step owns its kernels and offloads their compilation to the
    // KernelCompilationQueue: compileKernels() compiles them all in parallel.
    ExampleProcessingStep step(device, context, stream, input, output,
                               nrElements);
    step.compileKernels();

    step.process(2.0f, 10.0f);
    stream.synchronize();

    std::vector<float> hostOutput(nrElements);
    stream.memcpyDtoHAsync(hostOutput.data(), output, bytes);
    stream.synchronize();

    verifyResults(hostOutput, nrElements);
  } catch (const cu::Error &e) {
    std::cerr << "CUDA error: " << e.what() << std::endl;
    return 1;
  } catch (const nvrtc::Error &e) {
    std::cerr << "NVRTC error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
