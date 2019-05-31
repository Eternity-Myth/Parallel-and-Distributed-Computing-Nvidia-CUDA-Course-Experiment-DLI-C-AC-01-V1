#include <stdio.h>

/*
 * ホスト上で配列値を初期化します。
 */

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/*
 * GPU 上で要素を並列で 2 倍にします。
 */

__global__
void doubleElements(int *a, int N)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    a[i] *= 2;
  }
}

/*
 * ホスト上ですべての要素が 2 倍になっていることを確認します。
 */

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
  int N = 100;
  int *a;

  size_t size = N * sizeof(int);

  /*
   * このメモリの割り当てをリファクタリングして、
   * ホストとデバイスの両方で使用できるポインタ `a` を提供します。
   */

  a = (int *)malloc(size);

  init(a, N);

  size_t threads_per_block = 10;
  size_t number_of_blocks = 10;

  /*
   * この起動は、ポインタ `a` がデバイスで使用できるようになるまで機能しません。
   */

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  /*
   * ホストとデバイスの両方のアクセス用に割り当てた
   * メモリを解放するためにリファクタリングします。
   */

  free(a);
}
