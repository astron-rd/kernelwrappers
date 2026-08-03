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
