__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * "`cudaMallocManaged` の動作を詳しく確認するために実験を行います。
   *
   * ユニファイド メモリに GPU だけがアクセスした場合、どうなるでしょうか?
   * ユニファイド メモリに CPU だけがアクセスした場合、どうなるでしょうか? 
   * ユニファイド メモリに最初に GPU、次に CPU がアクセスした場合、どうなるでしょうか?
   * ユニファイド メモリに最初に CPU、次に GPU がアクセスした場合、どうなるでしょうか? 
   *
   * 各実験の前にユニファイド メモリ の動作、特にページ フォールトについて仮説を立ててから、`nvprof` を実行して検証します。
   */

  cudaFree(a);
}
