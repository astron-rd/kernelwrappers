#ifndef KERNELWRAPPERS_TESTUTIL_H
#define KERNELWRAPPERS_TESTUTIL_H

#include <cudawrappers/cu.hpp>

namespace kernelwrappers {

// Class to initialize CUDA
class BenchmarkFixtureBase {
public:
  BenchmarkFixtureBase() {
    try {
      cu::init();
    } catch (...) {
      init_failed_ = true;
    }
  }

protected:
  bool init_failed_ = false;
};

// Class to initialize common cu:: members
class TestFixture : public BenchmarkFixtureBase {
public:
  TestFixture() {
    try {
      if (init_failed_ || !cu::Device::getCount()) {
        skip_ = true;
        return;
      }

      device_ = new cu::Device(0);
      context_ = new cu::Context(0, *device_);
      context_->setCurrent();
      stream_ = new cu::Stream();
    } catch (...) {
      skip_ = true;
    }
  }

  ~TestFixture() {
    delete stream_;
    delete context_;
    delete device_;
  }

  TestFixture(const TestFixture &) = delete;
  TestFixture &operator=(const TestFixture &) = delete;

  bool skip_ = false;
  cu::Device *device_ = nullptr;
  cu::Context *context_ = nullptr;
  cu::Stream *stream_ = nullptr;
};

} // namespace kernelwrappers

#endif
