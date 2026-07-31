#ifndef KERNELWRAPPERS_KERNELCOMPILATIONQUEUE_H
#define KERNELWRAPPERS_KERNELCOMPILATIONQUEUE_H

#include <cstddef>
#include <cudawrappers/cu.hpp>
#include <vector>

namespace kernelwrappers {

class JITKernel;

/*
 * Collects kernels that need to be compiled, and compiles them in parallel.
 *
 * GPU kernel compilation can be a significant one-time cost. This queue allows
 * deferring compilation of all kernels (e.g. of an application or pipeline),
 * and compiling them together across multiple threads, one kernel per thread.
 *
 * Each worker thread calls cu::Context::setCurrent() before compiling, since
 * CUDA driver calls require a current context on the calling thread.
 */
class KernelCompilationQueue {
public:
  // Add a kernel to the compilation queue.
  void add(JITKernel &kernel);

  // Number of kernels currently in the queue.
  size_t size() const { return kernels.size(); }

  // Compile all queued kernels in parallel.
  //
  // \a context is made current on every worker thread before compiling.
  // After this call the queue is empty (all kernels have been compiled).
  void compileAll(cu::Context &context);

  // Discard all kernels from the queue without compiling them.
  void clear();

private:
  std::vector<JITKernel *> kernels;
};

} // namespace kernelwrappers

#endif // KERNELWRAPPERS_KERNELCOMPILATIONQUEUE_H
