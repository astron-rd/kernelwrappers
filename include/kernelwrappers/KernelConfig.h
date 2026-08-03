#ifndef KERNELWRAPPERS_KERNELCONFIG_H
#define KERNELWRAPPERS_KERNELCONFIG_H

#include <cudawrappers/cu.hpp>
#include <map>
#include <ostream>
#include <set>
#include <vector_types.h>

namespace kernelwrappers {

// Map for storing compile definitions that will be passed to the GPU kernel
// compiler on the command line.
struct CompileDefinitions : std::map<std::string, std::string> {};

// Set for storing compile flags that will be passed to the GPU kernel
// compiler on the command line.
struct CompileFlags : std::set<std::string> {};

// Struct representing a CUDA Grid.
struct Grid {
  Grid(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
  unsigned int x;
  unsigned int y;
  unsigned int z;
  friend std::ostream &operator<<(std::ostream &os, const Grid &grid);
  operator dim3() { return dim3(x, y, z); }
};

// Struct representing a CUDA Block.
struct Block {
  Block(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
  Block(const dim3 &block) : x(block.x), y(block.y), z(block.z) {}
  unsigned int x;
  unsigned int y;
  unsigned int z;
  friend std::ostream &operator<<(std::ostream &os, const Block &block);
  operator dim3() { return dim3(x, y, z); }
};

// Return the maximum dimensions of a block of threads.
struct Block getMaxBlockDims(cu::Device &device);

// Return the maximum dimensions of a grid of blocks.
struct Grid getMaxGridDims(cu::Device &device);

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_KERNELCONFIG_H
