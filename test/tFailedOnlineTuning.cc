#include "TestUtil.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelLauncher.h>
#include <optional>
#include <string>
#include <unistd.h>

using namespace kernelwrappers;

namespace {

class FailedTuningKernel : public JITKernel {
public:
  FailedTuningKernel(cu::Device &device, cu::Stream &stream,
                     const cu::DeviceMemory &input,
                     const cu::DeviceMemory &output)
      : JITKernel(device, stream, source(), "failed_tuning_kernel", {}, {}),
        input_(input), output_(output) {
    setTunable(true);
  }

  void run() {
    setArg(0, input_);
    setArg(1, output_);
    setArg(2, n_);
    launch();
  }

protected:
  void configureLaunch() override {
    setEnqueueWorkSizes(itsDevice, Grid(1), Block(32));
  }

  kernel_launcher::ProblemSize problemSize() const override {
    return kernel_launcher::ProblemSize(n_);
  }

  std::vector<kernel_launcher::KernelArg> createTunerInput() const override {
    return {
        kernel_launcher::KernelArg::from_array<float>(
            reinterpret_cast<float *>(static_cast<CUdeviceptr>(input_)), n_),
        kernel_launcher::KernelArg::from_array<float>(
            reinterpret_cast<float *>(static_cast<CUdeviceptr>(output_)), n_),
        kernel_launcher::KernelArg::from_scalar<int>(n_),
    };
  }

private:
  static std::string source() {
    return "extern \"C\"\n"
           "#pragma kernel tune(threads=32, 64)\n"
           "#pragma kernel problem_size(n)\n"
           "#pragma kernel block_size(threads)\n"
           "#pragma kernel grid_divisor(threads)\n"
           "#pragma kernel buffers(input[n], output[n])\n"
           "#pragma kernel tuning_key(\"failed_online_tuning\")\n"
           "__global__ void failed_tuning_kernel(const float *input, "
           "float *output, int n) {\n"
           "  const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) output[i] = input[i] * 2.0f;\n"
           "}";
  }

  const cu::DeviceMemory &input_;
  const cu::DeviceMemory &output_;
  const int n_ = 1;
};

class EnvironmentGuard {
public:
  EnvironmentGuard(const char *name, const std::string &value) : name_(name) {
    if (const char *previous = std::getenv(name)) {
      previous_ = previous;
    }
    setenv(name, value.c_str(), 1);
  }

  ~EnvironmentGuard() {
    if (previous_) {
      setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

private:
  std::string name_;
  std::optional<std::string> previous_;
};

} // namespace

TEST_CASE_METHOD(TestFixture, "JITKernel launches after online tuning failure",
                 "[online-tuning]") {
  if (skip_) {
    SKIP("CUDA unavailable");
  }

  EnvironmentGuard tune("KERNELWRAPPERS_TUNE_KERNELS", "1");
  EnvironmentGuard wisdom("KERNEL_LAUNCHER_WISDOM",
                         "/dev/null/kernelwrappers-wisdom");
  const float inputValue = 3.0f;
  float outputValue = 0.0f;
  cu::DeviceMemory input(sizeof(inputValue));
  cu::DeviceMemory output(sizeof(outputValue));
  stream_->memcpyHtoDAsync(input, &inputValue, sizeof(inputValue));

  FailedTuningKernel kernel(*device_, *stream_, input, output);
  kernel.run();
  stream_->memcpyDtoHAsync(&outputValue, output, sizeof(outputValue));
  stream_->synchronize();

  CHECK(outputValue == 6.0f);
}

TEST_CASE_METHOD(TestFixture, "JITKernel launches after capture directory failure",
                 "[online-tuning]") {
  if (skip_) {
    SKIP("CUDA unavailable");
  }

  EnvironmentGuard tune("KERNELWRAPPERS_TUNE_KERNELS", "1");
  const float inputValue = 3.0f;
  float outputValue = 0.0f;
  cu::DeviceMemory input(sizeof(inputValue));
  cu::DeviceMemory output(sizeof(outputValue));
  stream_->memcpyHtoDAsync(input, &inputValue, sizeof(inputValue));

  FailedTuningKernel kernel(*device_, *stream_, input, output);
  KernelLauncher::captureDirStorage() = "/dev/null";
  CHECK_NOTHROW(kernel.run());
  stream_->memcpyDtoHAsync(&outputValue, output, sizeof(outputValue));
  stream_->synchronize();

  CHECK(outputValue == 6.0f);
}
