#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelCache.h>
#include <kernelwrappers/Log.h>
#include <sstream>
#include <vector>
#include <xxhash.h>

namespace kernelwrappers {

bool KernelCache::writeKernelToCache(const JITKernel &kernel,
                                     const KernelCacheHash &sourceHash,
                                     const cu::Device &device,
                                     const std::string &ptx) {
  const std::string cacheFile = getCacheFilePath(kernel, sourceHash, device);
  if (!std::filesystem::exists(cacheDir)) {
    std::filesystem::create_directories(cacheDir);
  }

  std::ofstream outFile(cacheFile, std::ios::binary);

  if (!outFile.is_open()) {
    return false;
  }

  outFile.write(ptx.data(), ptx.size());
  outFile.close();
  return true;
}

bool KernelCache::readKernelFromCache(const JITKernel &kernel,
                                      const KernelCacheHash &sourceHash,
                                      const cu::Device &device,
                                      std::string &ptx) {
  const std::string cacheFile = getCacheFilePath(kernel, sourceHash, device);

  if (!std::filesystem::exists(cacheFile)) {
    return false;
  }

  std::ifstream inFile(cacheFile, std::ios::binary);
  if (!inFile.is_open()) {
    return false;
  }

  std::ostringstream ss;
  ss << inFile.rdbuf();
  ptx = ss.str();
  inFile.close();

  return true;
}

std::string KernelCache::getCacheFilePath(const JITKernel &kernel,
                                          const KernelCacheHash &sourceHash,
                                          const cu::Device &device) {
  const KernelCacheHash kernelNameHash = computeHash(kernel.getName());

  std::vector<std::string> sortedFlags(kernel.getCompileFlags().begin(),
                                       kernel.getCompileFlags().end());
  std::sort(sortedFlags.begin(), sortedFlags.end());
  std::string flagsConcatenated;
  for (const auto &flag : sortedFlags) {
    flagsConcatenated += flag;
  }
  const KernelCacheHash flagsHash = computeHash(flagsConcatenated);

  std::vector<std::string> definitionsConcatenated;
  for (const auto &def : kernel.getCompileDefinitions()) {
    definitionsConcatenated.emplace_back(def.first + "=" + def.second);
  }
  std::string definitionsStr;
  for (const auto &def : definitionsConcatenated) {
    definitionsStr += def;
  }

  const KernelCacheHash definitionsHash = computeHash(definitionsStr);

  std::ostringstream manifest;
  manifest << "kernel=" << kernel.getName() << '\n'
           << "source_hash=" << sourceHash << '\n'
           << "flags_hash=" << flagsHash << '\n'
           << "definitions_hash=" << definitionsHash << '\n'
           << "gpu_arch=" << device.getArch() << '\n'
           << "cuda_version=" << std::to_string(CUDA_VERSION) << '\n';

  Log::trace("Kernel cache manifest for " + kernel.getName() + ":\n" +
             manifest.str());

  const KernelCacheHash manifestHash = computeHash(manifest.str());
  const std::string manifestHashShort = manifestHash.substr(
      0, std::min<size_t>(kHashLength, manifestHash.size()));

  return cacheDir / std::filesystem::path(kernel.getName() + "_" +
                                          manifestHashShort + ".ptx");
}

bool KernelCache::isEnabled() {
  const char *cacheEnv;
  if (!(cacheEnv = std::getenv("KERNELWRAPPERS_CACHE_DISABLE"))) {
    return true;
  }

  return std::string(cacheEnv) != "1";
}

KernelCacheHash KernelCache::computeHash(const std::string &input) {
  const XXH128_hash_t hash = XXH3_128bits(input.data(), input.size());

  std::ostringstream os;
  os << std::hex << std::setfill('0') << std::setw(16) << hash.high64
     << std::setw(16) << hash.low64;

  return os.str();
}

} // namespace kernelwrappers
