#ifndef KERNELWRAPPERS_KERNEL_H
#define KERNELWRAPPERS_KERNEL_H

#include <cudawrappers/cu.hpp>
#include <string>

namespace kernelwrappers {

/*
 * A wrapper for a generic kernel that can be executed on a GPU.
 *
 * This base class only tracks the device, stream and name of a kernel.
 * Domain-specific behaviour (argument handling, launch configuration, ...)
 * is left to the derived classes.
 */
class Kernel {
public:
  virtual ~Kernel() = default;

  // Launch the kernel on the GPU.
  virtual void launch() = 0;

  // The GPU Device associated with this kernel.
  cu::Device &device() const { return itsDevice; }

  // The GPU Stream associated with this kernel.
  cu::Stream &stream() const { return itsStream; }

  // The name of this kernel.
  const std::string &name() const { return itsName; }

protected:
  // Construct a kernel.
  Kernel(cu::Device &device, cu::Stream &stream, const std::string &name);

  // The GPU Device associated with this kernel.
  cu::Device &itsDevice;

  // The GPU Stream associated with this kernel.
  cu::Stream &itsStream;

  // The name of this kernel.
  const std::string itsName;
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_KERNEL_H
