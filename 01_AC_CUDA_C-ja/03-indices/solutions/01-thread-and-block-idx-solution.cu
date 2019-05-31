#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  }
}

int main()
{
  /*
   * これは、カーネルが起動したときに成功メッセージを出力する実行コンテキストの 1 つです。
   */

  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();

  /*
   * カーネルの実行は非同期なので、その完了時に同期する必要があることを忘れないでください。
   */

  cudaDeviceSynchronize();
}
