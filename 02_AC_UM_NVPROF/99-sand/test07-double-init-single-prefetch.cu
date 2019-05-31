#include <stdio.h>

__global__
void kernel(float *a, float *b, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = a[i] * b[i];
  }
}

void init(float *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 2.0f;
  }
}

int main(int argc, char **argv)
{
  int deviceId;
  cudaGetDevice(&deviceId);

  int E = 20;
  if (argc > 1) E = atoi(argv[1]);
  
  int N = 2<<E;
  printf("N is 2<<%d: %d\n", E, 2<<E);

  float *a;
  float *b;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);
  cudaMallocManaged(&b, size);
  init(a, N);
  init(b, N);

  cudaMemPrefetchAsync(a, N, deviceId);

  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  kernel<<<numberOfBlocks, threadsPerBlock>>>(a, b, N);
  cudaDeviceSynchronize();
  printf("Done\n");
}
