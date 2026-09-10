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

Set `KERNELWRAPPERS_TUNE_KERNELS=1` when launching an application to perform
capture-and-tune in process. Each tunable kernel is then run normally with its
tuned configuration, or its defaults if Python tuning fails.

The example kernels select a 10,000,000-element representative workload for
captures. Normal application runs retain the 64-element example workload.

Wisdom is stored under `~/.kernelwrappers/Wisdom` by default and is loaded
automatically by subsequent normal application runs.
