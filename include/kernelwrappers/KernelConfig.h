#ifndef KERNELWRAPPERS_KERNELCONFIG_H
#define KERNELWRAPPERS_KERNELCONFIG_H

#include <cudawrappers/cu.hpp>
#include <map>
#include <ostream>
#include <set>
#include <vector_types.h>

namespace kernelwrappers {

// Map for storing compile definitions that will be passed to the GPU kernel
// compiler on the command line. The key is the name of the preprocessor
// variable, a value should only be assigned if the preprocessor variable
// needs to have a value.
struct CompileDefinitions : std::map<std::string, std::string> {};

// Set for storing compile flags that will be passed to the GPU kernel
// compiler on the command line. Flags generally don't have an associated
// value; if they do the value should become part of the flag in this set.
struct CompileFlags : std::set<std::string> {};

// Struct representing a CUDA Grid, which is similar to the @c dim3 type
// in the CUDA Runtime API.
struct Grid {
  Grid(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1);
  unsigned int x;
  unsigned int y;
  unsigned int z;
  friend std::ostream &operator<<(std::ostream &os, const Grid &grid);
  operator dim3() { return dim3(x, y, z); }
};

// Struct representing a CUDA Block, which is similar to the @c dim3 type
// in the CUDA Runtime API.
//
// @invariant x > 0, y > 0, z > 0
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
