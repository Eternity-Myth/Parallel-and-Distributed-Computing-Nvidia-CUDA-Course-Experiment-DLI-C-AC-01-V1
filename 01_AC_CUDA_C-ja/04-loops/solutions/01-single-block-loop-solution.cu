#include <stdio.h>


/*
 * 先ほどあった引数 `N` がないことに注目してください。
 */

__global__ void loop()
{
  /*
   * このカーネルは、元の for ループを 1 回だけ反復します。
   * このカーネルによって何回目の「反復」が実行されているかは、
   * `threadIdx.x` を使用して確認できます。
   */

  printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
  /*
   * これは「ループ」の「反復」回数を設定する実行コンテキストです。
   */

  loop<<<1, 10>>>();
  cudaDeviceSynchronize();
}
