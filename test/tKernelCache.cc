#include "TestUtil.h"

#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelCache.h>
#include <string>
#include <unistd.h>

using namespace kernelwrappers;

namespace {

class CachableTestKernel : public JITKernel {
public:
  CachableTestKernel(
      cu::Device &device, cu::Stream &stream, const std::string &source,
      const std::string &name, const CompileFlags &flags = CompileFlags(),
      const CompileDefinitions &definitions = CompileDefinitions(),
      bool cachable = true)
      : JITKernel(device, stream, source, name, flags, definitions, cachable,
                  false) {}
};

std::string makeUniqueKernelName(const std::string &prefix) {
  const auto now =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return prefix + "_" + std::to_string(getpid()) + "_" + std::to_string(now);
}

std::string makeKernelSource(const std::string &kernelName) {
  return std::string("extern \"C\" __global__ void ") + kernelName +
         "(const float* in, float* out) { const int idx = blockIdx.x * "
         "blockDim.x + threadIdx.x; if (idx == 0) out[0] = in[0]; }";
}

} // namespace

TEST_CASE_METHOD(TestFixture, "KernelCache writes and reads PTX",
                 "[kernel-cache]") {
  if (skip_) {
    return;
  }

  const std::string kernelName = makeUniqueKernelName("cache_write_read");
  const std::string source = makeKernelSource(kernelName);
  const KernelCacheHash sourceHash = KernelCache::computeHash(source);

  CachableTestKernel kernel(device_, stream_, source, kernelName);

  std::string ptx;
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernel, sourceHash, device_, ptx));

  kernel.compile();

  REQUIRE(KernelCache::readKernelFromCache(kernel, sourceHash, device_, ptx));
  CHECK_FALSE(ptx.empty());
}

TEST_CASE_METHOD(TestFixture,
                 "KernelCache key changes when compile definitions change",
                 "[kernel-cache]") {
  if (skip_) {
    return;
  }

  const std::string kernelName = makeUniqueKernelName("cache_definitions");
  const std::string source = makeKernelSource(kernelName);
  const KernelCacheHash sourceHash = KernelCache::computeHash(source);

  CompileDefinitions definitionsA;
  definitionsA["CACHE_VARIANT"] = "1";

  CompileDefinitions definitionsB;
  definitionsB["CACHE_VARIANT"] = "2";

  CachableTestKernel kernelA(device_, stream_, source, kernelName,
                             CompileFlags(), definitionsA, true);
  CachableTestKernel kernelB(device_, stream_, source, kernelName,
                             CompileFlags(), definitionsB, true);

  kernelA.compile();

  std::string ptx;
  REQUIRE(KernelCache::readKernelFromCache(kernelA, sourceHash, device_, ptx));
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernelB, sourceHash, device_, ptx));

  kernelB.compile();

  REQUIRE(KernelCache::readKernelFromCache(kernelB, sourceHash, device_, ptx));
}

TEST_CASE_METHOD(TestFixture,
                 "KernelCache does not store PTX for non-cachable kernels",
                 "[kernel-cache]") {
  if (skip_) {
    return;
  }

  const std::string kernelName = makeUniqueKernelName("cache_disabled");
  const std::string source = makeKernelSource(kernelName);
  const KernelCacheHash sourceHash = KernelCache::computeHash(source);

  CachableTestKernel kernel(device_, stream_, source, kernelName,
                            CompileFlags(), CompileDefinitions(), false);

  kernel.compile();

  std::string ptx;
  CHECK_FALSE(
      KernelCache::readKernelFromCache(kernel, sourceHash, device_, ptx));
}