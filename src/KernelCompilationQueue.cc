#include <algorithm>
#include <atomic>
#include <kernelwrappers/JITKernel.h>
#include <kernelwrappers/KernelCompilationQueue.h>
#include <thread>

namespace kernelwrappers {

void KernelCompilationQueue::add(JITKernel &kernel) {
  kernels.push_back(&kernel);
}

void KernelCompilationQueue::clear() { kernels.clear(); }

void KernelCompilationQueue::compileAll(cu::Context &context) {
  if (kernels.empty()) {
    return;
  }

  const size_t total_kernels = kernels.size();
  const unsigned hw_threads = std::thread::hardware_concurrency();
  const size_t max_threads = std::min(
      total_kernels, static_cast<size_t>((hw_threads > 0) ? hw_threads : 1));

  std::atomic<size_t> next_index{0};
  std::vector<std::thread> workers;
  workers.reserve(max_threads);

  for (size_t t = 0; t < max_threads; ++t) {
    workers.emplace_back([this, &context, &next_index, total_kernels]() {
      // Every thread needs a current CUDA context before driver calls.
      context.setCurrent();

      while (true) {
        const size_t i = next_index.fetch_add(1, std::memory_order_relaxed);
        if (i >= total_kernels) {
          break;
        }
        kernels[i]->compile();
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  kernels.clear();
}

} // namespace kernelwrappers
