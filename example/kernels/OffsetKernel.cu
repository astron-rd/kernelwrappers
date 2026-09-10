extern "C"
#pragma kernel tune(threads_per_block=32, 64, 128, 256, 512)
#pragma kernel tune(items_per_thread=1, 2, 4)
#pragma kernel set(items_per_block=threads_per_block * items_per_thread)
#pragma kernel problem_size(n)
#pragma kernel block_size(threads_per_block)
#pragma kernel grid_divisor(items_per_block)
#pragma kernel buffers(output[n])
#pragma kernel tuning_key("offset_kernel")
__global__ void offset_kernel(float *output, float offset, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += gridDim.x * blockDim.x) {
    output[i] = output[i] + offset;
  }
}
