#include <stdio.h>


__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
  {
    a[i] = initialValue;
  }
}

int main()
{
  /*
   * `N` は変更しないでください。
   */

  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);

  /*
   * 何らかの理由により、スレッドの数を `256` に固定する必要がある場合、`threads_per_block` は変更しないでください。
   */

  size_t threads_per_block = 256;

  /*
   * 以下は、グリッドに少なくとも `N` 個の要素と同じ数のスレッドがあることを確認するための CUDA 慣用句です。
   */

  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  int initialValue = 6;

  initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
  cudaDeviceSynchronize();

  /*
   * `a` のすべての値が初期化されたことを確認します。
   */

  for (int i = 0; i < N; ++i)
  {
    if(a[i] != initialValue)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initialValue, i, a[i]);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a);
}
