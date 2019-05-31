#include <stdio.h>

/*
 * 現在、`initializeElementsTo` は、`i` が `N` より大きいスレッドで実行されると、
 * `a` の範囲外の値にアクセスしようとします。
 *
 * カーネルの定義をリファクタリングして、範囲外のアクセスを防止します。
 */

__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  a[i] = initialValue;
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
   * 何らかの理由により、スレッドの数を `256` に固定する必要がある場合、
   * `threads_per_block` は変更しないでください。
   */

  size_t threads_per_block = 256;

  /*
   * `N` と `threads_per_block` の固定値を考慮して、
   * 実行構成に対して有効な値を `number_of_blocks` に代入します。
   */

  size_t number_of_blocks = 0;

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
