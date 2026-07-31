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
