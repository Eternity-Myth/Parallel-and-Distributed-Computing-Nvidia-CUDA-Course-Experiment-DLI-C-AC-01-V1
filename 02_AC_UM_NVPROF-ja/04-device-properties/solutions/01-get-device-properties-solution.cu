#include <stdio.h>

int main()
{
  /*
   * デバイスを照会するには、まず、デバイス ID が必要です。
   */

  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);

  /*
   * `props` には、現在のデバイスに関するいくつかのプロパティが含まれています。
   */

  int computeCapabilityMajor = props.major;
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount = props.multiProcessorCount;
  int warpSize = props.warpSize;


  printf("Device ID: %d\nNumber of SMs: %d\nCompute Capability Major: %d\nCompute Capability Minor: %d\nWarp Size: %d\n", deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);
}
