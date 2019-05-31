#include <stdio.h>

int main()
{
  /*
   * 以下の出力文字列で現在アクティブな GPU の要求された
   * プロパティを出力するために、これらの変数に値を代入します。
   */

  int deviceId;
  int computeCapabilityMajor;
  int computeCapabilityMinor;
  int multiProcessorCount;
  int warpSize;

  /*
   * 以下の出力文字列を変更する必要はありません。
   */

  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
