#include <stdio.h>

__global__ void loop()
{
  /*
   * 以下の慣用表現では、各スレッドにグリッド全体で一意のインデックスを割り当てます。
   */

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

int main()
{
  /*
   * 演習の制約を満たしたうえで動作する追加の実行構成は次のとおりです。
   *
   * <<<5, 2>>>
   * <<<10, 1>>>
   */

  loop<<<2, 5>>>();
  cudaDeviceSynchronize();
}
