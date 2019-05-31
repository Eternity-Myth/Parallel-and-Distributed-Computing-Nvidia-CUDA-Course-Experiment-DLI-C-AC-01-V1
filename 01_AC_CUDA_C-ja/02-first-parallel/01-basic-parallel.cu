#include <stdio.h>

/*
 * firstParallel をリファクタリングして、GPU 上で実行できるようにします。
 */

void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{
  /*
   * この firstParallel の呼び出しをリファクタリングして、GPU 上で並列実行するようにします。
   */

  firstParallel();

  /*
   * 以下では、CPU が処理を進める前に GPU カーネルが完了するのを待機するためのコードが必要です。
   */

}
