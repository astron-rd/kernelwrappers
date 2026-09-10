# kernelwrappers

A small, reusable library for projects that JIT-compile and run CUDA kernels at
runtime with NVRTC.

## Dependencies

- CUDA (driver API, NVRTC)
- [cudawrappers](https://github.com/nlesc-recruit/cudawrappers) (fetched
  automatically by CMake)
- OpenSSL

## Building

```sh
cmake -S . -B build
cmake --build build
```

## Example

`example/` shows a multi-kernel processing step (`ExampleProcessingStep`) that
offloads compilation of its kernels to `KernelCompilationQueue`. The CUDA kernel
source lives in a separate `.cu` file, embedded into the binary at build time
with cudawrappers. Run it with:

```sh
./build/example/kernelwrappers_example
```

When Kernel Launcher support is enabled, each application can expose a `tune`
companion executable. The included example builds one at `build/example/tune`.
It captures its representative workload and invokes the tuner without requiring
users to set capture environment variables or run Python scripts themselves:

```sh
./build/example/tune scale_kernel offset_kernel
# or tune every pragma kernel launched by the example
./build/example/tune --all
```

The example kernels select a 10,000,000-element representative workload for
captures. Normal application runs retain the 64-element example workload.

Wisdom is stored under `~/.kernelwrappers/Wisdom` by default and is loaded
automatically by subsequent normal application runs.
