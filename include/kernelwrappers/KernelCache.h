#ifndef KERNELWRAPPERS_KERNELCACHE_H
#define KERNELWRAPPERS_KERNELCACHE_H

#include <cudawrappers/cu.hpp>
#include <kernelwrappers/KernelCacheHash.h>
#include <string>
#include <string_view>

#ifndef KERNELWRAPPERS_CACHE_PATH
#define KERNELWRAPPERS_CACHE_PATH "/tmp/kernelwrappers/KernelCache"
#endif

namespace kernelwrappers {

class JITKernel;

/*
 * Disk cache for the PTX of runtime-compiled kernels.
 *
 * The cache is keyed by a manifest of everything that influences the compiled
 * output (kernel name, source hash, compile flags, compile definitions, GPU
 * architecture and CUDA version), so a cache hit is only possible for
 * bit-identical inputs.
 *
 * The cache can be disabled entirely by setting the
 * KERNELWRAPPERS_CACHE_DISABLE environment variable to "1".
 */
class KernelCache {
public:
  static bool writeKernelToCache(const JITKernel &kernel,
                                 const KernelCacheHash &sourceHash,
                                 const cu::Device &device,
                                 const std::string &ptx);
  static bool readKernelFromCache(const JITKernel &kernel,
                                  const KernelCacheHash &sourceHash,
                                  const cu::Device &device, std::string &ptx);
  static bool isEnabled();

  static KernelCacheHash computeHash(const std::string &input);

private:
  static std::string getCacheFilePath(const JITKernel &kernel,
                                      const KernelCacheHash &sourceHash,
                                      const cu::Device &device);

  static constexpr std::string_view cacheDir =
      KERNELWRAPPERS_CACHE_PATH; // Directory to store cached kernels

  static constexpr size_t kHashLength =
      16; // Length of hash to use in cache file names
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_KERNELCACHE_H
