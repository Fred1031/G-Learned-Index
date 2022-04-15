#include <cstdio>

__global__ void foo(int *d_vecs, int *d_offsets, int *d_vals, int vec_num) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= vec_num)
    return;

  int begin = d_offsets[tid], end = d_offsets[tid + 1];
  int val = d_vals[tid];
  for (int i = begin; i < end; ++i)
    d_vecs[i] = val;
}

int main() {
  constexpr int vec_num = 4;
  int sizes[vec_num] = {5, 6, 3, 2};
  int offsets[vec_num + 1] = {0};
  int vals[vec_num] = {1, 2, 3, 4};
  for (int i = 0; i < vec_num; ++i)
    offsets[i + 1] = offsets[i] + sizes[i];

  int *d_vecs;
  cudaMallocManaged(&d_vecs, sizeof(int) * offsets[vec_num]);

  int *d_offsets;
  cudaMalloc(&d_offsets, sizeof(int) * (vec_num + 1));
  cudaMemcpy(d_offsets, offsets, sizeof(int) * (vec_num + 1),
             cudaMemcpyHostToDevice);

  int *d_vals;
  cudaMalloc(&d_vals, sizeof(int) * vec_num);
  cudaMemcpy(d_vals, vals, sizeof(int) * vec_num, cudaMemcpyHostToDevice);

  foo<<<1, 32>>>(d_vecs, d_offsets, d_vals, vec_num);

  cudaDeviceSynchronize();

  bool pass = true;
  for (int i = 0; i < vec_num; ++i) {
    int begin = offsets[i], end = offsets[i + 1];
    for (int j = begin; j < end; ++j) {
      if (d_vecs[j] != vals[i])
        pass = false;
    }
  }

  printf("pass = %d\n", pass);

  cudaFree(d_vecs);
  cudaFree(d_offsets);
  cudaFree(d_vals);
}
