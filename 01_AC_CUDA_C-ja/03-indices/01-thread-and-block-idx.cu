#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  } else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

int main()
{
  /*
   * カーネルが「Success!」を出力するように実行構成を更新します。
   */

  printSuccessForCorrectExecutionConfiguration<<<1, 1>>>();
}
