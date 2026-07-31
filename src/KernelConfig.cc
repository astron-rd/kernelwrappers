#include <kernelwrappers/KernelConfig.h>
#include <sstream>
#include <stdexcept>

namespace kernelwrappers {

Grid::Grid(unsigned int x_, unsigned int y_, unsigned int z_)
    : x(x_), y(y_), z(z_) {}

std::ostream &operator<<(std::ostream &os, const Grid &grid) {
  os << "[" << grid.x << ", " << grid.y << ", " << grid.z << "]";
  return os;
}

Block::Block(unsigned int x_, unsigned int y_, unsigned int z_)
    : x(x_), y(y_), z(z_) {
  // Cannot enforce this as an obj invariant (x, y, z public on purpose),
  // but intended to trigger bugs early.
  if (x == 0 || y == 0 || z == 0) {
    std::stringstream message;
    message << "Block(): block dims must be non-zero: " << x << " " << y << " "
            << z;
    throw std::runtime_error(message.str());
  }
}

std::ostream &operator<<(std::ostream &os, const Block &block) {
  os << "[" << block.x << ", " << block.y << ", " << block.z << "]";
  return os;
}

Block getMaxBlockDims(cu::Device &device) {
  Block block;
  block.x = (unsigned)device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
  block.y = (unsigned)device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
  block.z = (unsigned)device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
  return block;
}

Grid getMaxGridDims(cu::Device &device) {
  Grid grid;
  grid.x = (unsigned)device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
  grid.y = (unsigned)device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
  grid.z = (unsigned)device.getAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
  return grid;
}

} // namespace kernelwrappers
