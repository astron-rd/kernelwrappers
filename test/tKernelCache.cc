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
    SKIP("CUDA unavailable");
  }
  if (!KernelCache::isEnabled()) {
    SKIP("KernelCache disabled (KERNELWRAPPERS_CACHE_DISABLE=1)");
  }

  const std::string kernelName = makeUniqueKernelName("cache_write_read");
  const std::string source = makeKernelSource(kernelName);
  const KernelCacheHash sourceHash = KernelCache::computeHash(source);

  CachableTestKernel kernel(*device_, *stream_, source, kernelName);

  std::string ptx;
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernel, sourceHash, *device_, ptx));

  kernel.compile();

  REQUIRE(KernelCache::readKernelFromCache(kernel, sourceHash, *device_, ptx));
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

  CachableTestKernel kernelA(*device_, *stream_, source, kernelName,
                             CompileFlags(), definitionsA, true);
  CachableTestKernel kernelB(*device_, *stream_, source, kernelName,
                             CompileFlags(), definitionsB, true);

  kernelA.compile();

  std::string ptx;
  REQUIRE(KernelCache::readKernelFromCache(kernelA, sourceHash, *device_, ptx));
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernelB, sourceHash, *device_, ptx));

  kernelB.compile();

  REQUIRE(KernelCache::readKernelFromCache(kernelB, sourceHash, *device_, ptx));
}

TEST_CASE_METHOD(TestFixture,
                 "KernelCache key changes when kernel name changes",
                 "[kernel-cache]") {
  if (skip_) {
    return;
  }

  const std::string kernelNameA = makeUniqueKernelName("cache_name_A");
  const std::string kernelNameB = makeUniqueKernelName("cache_name_B");
  const std::string sourceA = makeKernelSource(kernelNameA);
  const std::string sourceB = makeKernelSource(kernelNameB);
  const KernelCacheHash sourceHashA = KernelCache::computeHash(sourceA);
  const KernelCacheHash sourceHashB = KernelCache::computeHash(sourceB);

  CachableTestKernel kernelA(*device_, *stream_, sourceA, kernelNameA);
  CachableTestKernel kernelB(*device_, *stream_, sourceB, kernelNameB);

  kernelA.compile();

  std::string ptx;
  REQUIRE(
      KernelCache::readKernelFromCache(kernelA, sourceHashA, *device_, ptx));
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernelB, sourceHashB, *device_, ptx));
}

TEST_CASE_METHOD(TestFixture,
                 "KernelCache key changes when compile flags change",
                 "[kernel-cache]") {
  if (skip_) {
    return;
  }

  const std::string kernelName = makeUniqueKernelName("cache_flags");
  const std::string source = makeKernelSource(kernelName);
  const KernelCacheHash sourceHash = KernelCache::computeHash(source);

  CompileFlags flagsA;
  flagsA.insert("--use_fast_math");

  CompileFlags flagsB;
  flagsB.insert("-lineinfo");

  CachableTestKernel kernelA(*device_, *stream_, source, kernelName, flagsA);
  CachableTestKernel kernelB(*device_, *stream_, source, kernelName, flagsB);

  kernelA.compile();

  std::string ptx;
  REQUIRE(KernelCache::readKernelFromCache(kernelA, sourceHash, *device_, ptx));
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernelB, sourceHash, *device_, ptx));
}

TEST_CASE_METHOD(TestFixture, "KernelCache key changes when source changes",
                 "[kernel-cache]") {
  if (skip_) {
    return;
  }

  const std::string kernelName = makeUniqueKernelName("cache_source");
  const std::string nameA = "kernelSourceA";
  const std::string nameB = "kernelSourceB";

  const std::string sourceA =
      std::string("extern \"C\" __global__ void ") + nameA +
      "(const float* in, float* out) { const int idx = blockIdx.x * "
      "blockDim.x + threadIdx.x; if (idx == 0) out[0] = in[0]; }";
  const std::string sourceB =
      std::string("extern \"C\" __global__ void ") + nameB +
      "(const float* in, float* out) { const int idx = blockIdx.x * "
      "blockDim.x + threadIdx.x; if (idx == 0) out[0] = 2.0f * in[0]; }";
  const KernelCacheHash sourceHashA = KernelCache::computeHash(sourceA);
  const KernelCacheHash sourceHashB = KernelCache::computeHash(sourceB);

  CachableTestKernel kernelA(*device_, *stream_, sourceA, nameA);
  CachableTestKernel kernelB(*device_, *stream_, sourceB, nameB);

  kernelA.compile();

  std::string ptx;
  REQUIRE(
      KernelCache::readKernelFromCache(kernelA, sourceHashA, *device_, ptx));
  REQUIRE_FALSE(
      KernelCache::readKernelFromCache(kernelB, sourceHashB, *device_, ptx));
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

  CachableTestKernel kernel(*device_, *stream_, source, kernelName,
                            CompileFlags(), CompileDefinitions(), false);

  kernel.compile();

  std::string ptx;
  CHECK_FALSE(
      KernelCache::readKernelFromCache(kernel, sourceHash, *device_, ptx));
}