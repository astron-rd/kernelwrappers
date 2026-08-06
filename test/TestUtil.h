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

      device_ = std::make_unique<cu::Device>(0);
      context_ = std::make_unique<cu::Context>(0, *device_);
      context_->setCurrent();
      stream_ = std::make_unique<cu::Stream>();
    } catch (...) {
      skip_ = true;
    }
  }

  TestFixture(const TestFixture &) = delete;
  TestFixture &operator=(const TestFixture &) = delete;

  bool skip_ = false;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Stream> stream_;
};

} // namespace kernelwrappers

#endif
