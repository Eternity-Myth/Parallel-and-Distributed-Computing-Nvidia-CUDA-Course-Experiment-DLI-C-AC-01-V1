#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * `helloGPU` の定義を、GPU 上で起動できるカーネルに
 * リファクタリングします。「Hello from the GPU!」と
 * 出力されるようにメッセージを更新します。
 */

void helloGPU()
{
  printf("Hello also from the CPU.\n");
}

int main()
{

  helloCPU();

  /*
   * この `helloGPU` の呼び出しをリファクタリングして、
   * GPU 上のカーネルとして起動するようにします。
   */

  helloGPU();

  /*
   * この下に、CPU スレッドを続行する前に `helloGPU`
   * カーネルの完了時に同期させるコードを追加します。
   */
}
