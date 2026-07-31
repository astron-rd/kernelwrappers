/*
 * Example CUDA kernels for the kernelwrappers example.
 *
 * This file is embedded into the example binary at build time (see
 * example/CMakeLists.txt, target_embed_source) and JIT-compiled at runtime
 * with NVRTC. NR_ELEMENTS is provided as a compile definition.
 */

extern "C" __global__ void scale_kernel(const float *input, float *output,
                                        float factor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NR_ELEMENTS) {
    output[i] = input[i] * factor;
  }
}

extern "C" __global__ void offset_kernel(const float *input, float *output,
                                         float offset) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NR_ELEMENTS) {
    output[i] = input[i] + offset;
  }
}
