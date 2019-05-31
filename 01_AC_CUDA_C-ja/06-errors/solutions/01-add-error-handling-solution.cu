#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  /*
   * 先ほどのコード (現在コメントアウト済み) では、
   * `a` の範囲外の要素にアクセスしようとしていました。
   */

  // for (int i = idx; i < N + stride; i += stride)
  for (int i = idx; i < N; i += stride)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 10000;
  int *a;

  size_t size = N * sizeof(int);
  cudaMallocManaged(&a, size);

  init(a, N);

  /*
   * 先ほどのコード (現在コメントアウト済み) では、
   * ブロックあたりの最大数 (1024) を超えるスレッドで
   * カーネルを起動しようとしていました。
   */

  size_t threads_per_block = 1024;
  /* size_t threads_per_block = 2048; */
  size_t number_of_blocks = 32;

  cudaError_t syncErr, asyncErr;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

  /*
   * 上記のカーネルの起動エラーと、非同期の `doubleElements`
   * カーネルの実行中に発生したエラーの両方をキャッチします。
   */

  syncErr = cudaGetLastError();
  asyncErr = cudaDeviceSynchronize();

  /*
   * エラーが存在する場合は、そのエラーを出力します。
   */

  if (syncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(syncErr));
  if (asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  cudaFree(a);
}
