#include <kernelwrappers/Kernel.h>

namespace kernelwrappers {

Kernel::Kernel(cu::Device &device, cu::Stream &stream, const std::string &name)
    : itsDevice(device), itsStream(stream), itsName(name) {}

} // namespace kernelwrappers
