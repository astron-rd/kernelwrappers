#include "ScaleKernel.h"

#include <example/kernels/ScaleKernel.cu.o.h>

namespace {

template <typename T> T ceilDiv(T n, T divisor) {
  return (n + divisor - 1) / divisor;
}

#ifdef KERNELWRAPPERS_TUNER
constexpr size_t kTuningNrElements = 10000000;
#endif

} // namespace

ScaleKernel::ScaleKernel(cu::Device &device, cu::Stream &stream,
                         const cu::DeviceMemory &input,
                         const cu::DeviceMemory &output, size_t nrElements)
    : JITKernel(device, stream, ScaleKernel_source, "scale_kernel",
                compileFlags(), compileDefinitions()),
      input_(input), output_(output), nrElements_(nrElements) {
  setTunable(true);
}

kernelwrappers::CompileFlags ScaleKernel::compileFlags() const {
  kernelwrappers::CompileFlags flags = JITKernel::compileFlags();
  flags.insert("--device-as-default-execution-space");
  return flags;
}

void ScaleKernel::configureLaunch() {
  unsigned threadsPerBlock = 256;
  unsigned itemsPerThread = 1;

#ifdef KERNELWRAPPERS_TUNER
  if (const auto &tunedConfig = getTunedConfig()) {
    threadsPerBlock = static_cast<unsigned>(
        tunedConfig->at("threads_per_block").to_integer());
    itemsPerThread =
        static_cast<unsigned>(tunedConfig->at("items_per_thread").to_integer());
  }
#endif

  const size_t itemsPerBlock =
      static_cast<size_t>(threadsPerBlock) * itemsPerThread;
  setEnqueueWorkSizes(itsDevice,
                      kernelwrappers::Grid(ceilDiv(nrElements_, itemsPerBlock)),
                      kernelwrappers::Block(threadsPerBlock));
}

#ifdef KERNELWRAPPERS_TUNER

kernel_launcher::ProblemSize ScaleKernel::problemSize() const {
  return kernel_launcher::ProblemSize(static_cast<uint32_t>(nrElements_));
}

std::vector<kernel_launcher::KernelArg> ScaleKernel::createTunerInput() const {
  const uint32_t n = static_cast<uint32_t>(nrElements_);
  return {
      kernel_launcher::KernelArg::from_array<float>(
          reinterpret_cast<float *>(static_cast<CUdeviceptr>(input_)), n),
      kernel_launcher::KernelArg::from_array<float>(
          reinterpret_cast<float *>(static_cast<CUdeviceptr>(output_)), n),
      kernel_launcher::KernelArg::from_scalar<float>(factor_),
      kernel_launcher::KernelArg::from_scalar<int>(static_cast<int>(n)),
  };
}

#endif // KERNELWRAPPERS_TUNER

void ScaleKernel::run(float factor) {
  factor_ = factor;
  setArg(0, input_);
  setArg(1, output_);
  setArg(2, factor);
  setArg(3, static_cast<int>(nrElements_));
  launch();
}
