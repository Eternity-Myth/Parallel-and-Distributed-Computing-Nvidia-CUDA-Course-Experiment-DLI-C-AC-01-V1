#include <stdio.h>

void init(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 2;
  }
}

int main(int argc, char **argv)
{
  int E = 20;
  if (argc > 1) E = atoi(argv[1]);
  
  int N = 2<<E;
  printf("N is 2<<%d: %d\n", E, 2<<E);

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);
  init(a, N);
}
