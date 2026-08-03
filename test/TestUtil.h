#ifndef KERNELWRAPPERS_TESTUTIL_H
#define KERNELWRAPPERS_TESTUTIL_H

#include <cudawrappers/cu.hpp>
#include <stdexcept>

namespace kernelwrappers {

// Class to initialize CUDA
class BenchmarkFixtureBase {
public:
  BenchmarkFixtureBase() { cu::init(); }
};

// Class to initialize common cu:: members
class TestFixture : public BenchmarkFixtureBase {
public:
  TestFixture() : device_(0), context_(0, device_), stream_() {
    // Set up gpu environment
    try {
      if (!cu::Device::getCount()) {
        skip_ = true;
      }
    } catch (std::runtime_error &) {
      skip_ = true;
    }
  }

  bool skip_ = false;
  cu::Device device_;
  cu::Context context_;
  cu::Stream stream_;
};

} // namespace kernelwrappers

#endif
