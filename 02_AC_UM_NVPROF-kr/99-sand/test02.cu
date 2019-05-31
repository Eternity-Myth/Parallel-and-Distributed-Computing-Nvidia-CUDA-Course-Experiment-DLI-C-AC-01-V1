#include <stdio.h>

__global__
void kernel(int *a, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = a[i] * 2;
  }
}


void init(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 2;
  }
}

int main(int argc, char **argv)
{
  int E = 20;
  if (argc > 1) E = atoi(argv[1]);
  
  int N = 2<<E;
  printf("N is 2<<%d: %d\n", E, 2<<E);

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);
  init(a, N);

  size_t threadsPerBlock = 256;
  size_t numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  kernel<<<numberOfBlocks, threadsPerBlock>>>(a, N);
  cudaDeviceSynchronize();
  printf("Done\n");
}
