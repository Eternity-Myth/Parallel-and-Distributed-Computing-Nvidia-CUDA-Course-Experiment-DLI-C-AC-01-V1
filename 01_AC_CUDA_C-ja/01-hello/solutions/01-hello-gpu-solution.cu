#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * `__global__` を追加すると、この関数が GPU 上で起動します。
 */

__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{
  helloCPU();


  /*
   * <<<...>>> 構文を使用した実行構成を追加すると、
   * この関数が GPU 上でカーネルとして起動します。
   */

  helloGPU<<<1, 1>>>();

  /*
   * `cudaDeviceSynchronize` はすべての GPU カーネルが
   * 完了するまで  CPU ストリームをブロックします。
   */

  cudaDeviceSynchronize();
}
