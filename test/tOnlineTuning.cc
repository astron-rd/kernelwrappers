#include "TestUtil.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <kernelwrappers/JITKernel.h>
#include <optional>
#include <string>
#include <unistd.h>

using namespace kernelwrappers;

namespace {

class OnlineTuningKernel : public JITKernel {
public:
  OnlineTuningKernel(cu::Device &device, cu::Stream &stream,
                     const cu::DeviceMemory &input,
                     const cu::DeviceMemory &output, std::string key)
      : JITKernel(device, stream, source(key), "online_tuning_kernel", {}, {}),
        input_(input), output_(output) {
    setTunable(true);
  }

  void run() {
    setArg(0, input_);
    setArg(1, output_);
    setArg(2, n_);
    launch();
  }

  bool usedTunedConfig() const { return usedTunedConfig_; }

protected:
  void configureLaunch() override {
    unsigned threads = 32;
    if (const auto &config = getTunedConfig()) {
      threads = static_cast<unsigned>(config->at("threads").to_integer());
      usedTunedConfig_ = true;
    }
    setEnqueueWorkSizes(itsDevice, Grid(1), Block(threads));
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
  static std::string source(const std::string &key) {
    return "extern \"C\"\n"
           "#pragma kernel tune(threads=32, 64)\n"
           "#pragma kernel problem_size(n)\n"
           "#pragma kernel block_size(threads)\n"
           "#pragma kernel grid_divisor(threads)\n"
           "#pragma kernel buffers(input[n], output[n])\n"
           "#pragma kernel tuning_key(\"" +
           key +
           "\")\n"
           "__global__ void online_tuning_kernel(const float *input, "
           "float *output, int n) {\n"
           "  const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
           "  if (i < n) output[i] = input[i] * 2.0f;\n"
           "}";
  }

  const cu::DeviceMemory &input_;
  const cu::DeviceMemory &output_;
  const int n_ = 1;
  bool usedTunedConfig_ = false;
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

TEST_CASE_METHOD(TestFixture, "JITKernel tunes online", "[online-tuning]") {
  if (skip_) {
    SKIP("CUDA unavailable");
  }

  const auto id =
      std::to_string(getpid()) + "_" +
      std::to_string(
          std::chrono::high_resolution_clock::now().time_since_epoch().count());
  const std::string key = "online_tuning_" + id;
  const auto wisdomDir =
      std::filesystem::temp_directory_path() / ("kernelwrappers-wisdom-" + id);
  std::filesystem::create_directories(wisdomDir);
  EnvironmentGuard tune("KERNELWRAPPERS_TUNE_KERNELS", "1");
  EnvironmentGuard wisdom("KERNEL_LAUNCHER_WISDOM", wisdomDir.string());

  const float inputValue = 3.0f;
  float outputValue = 0.0f;
  cu::DeviceMemory input(sizeof(inputValue));
  cu::DeviceMemory output(sizeof(outputValue));
  stream_->memcpyHtoDAsync(input, &inputValue, sizeof(inputValue));

  OnlineTuningKernel kernel(*device_, *stream_, input, output, key);
  kernel.run();
  stream_->memcpyDtoHAsync(&outputValue, output, sizeof(outputValue));
  stream_->synchronize();

  CHECK(outputValue == 6.0f);
  CHECK(kernel.usedTunedConfig());
  CHECK(std::filesystem::exists(wisdomDir / (key + ".wisdom")));
  std::filesystem::remove_all(wisdomDir);
}
