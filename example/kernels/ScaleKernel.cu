extern "C"
#pragma kernel tune(threads_per_block = 32, 64, 128, 256, 512)
#pragma kernel tune(items_per_thread = 1, 2, 4)
#pragma kernel set(items_per_block = threads_per_block * items_per_thread)
#pragma kernel problem_size(n)
#pragma kernel block_size(threads_per_block)
#pragma kernel grid_divisor(items_per_block)
#pragma kernel buffers(input[n], output[n])
#pragma kernel tuning_key("scale_kernel")
    __global__ void
    scale_kernel(const float *input, float *output, float factor, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    output[i] = input[i] * factor;
  }
}
