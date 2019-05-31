#include <stdio.h>

/*
 * `loop` を CUDA カーネルにリファクタリングします。
 * 新しいカーネルは元のループを 1 回だけ反復する必要があります。
 */

void loop(int N)
{
  for (int i = 0; i < N; ++i)
  {
    printf("This is iteration number %d\n", i);
  }
}

int main()
{
  /*
   * `loop` をカーネルとして起動するようにリファクタリングする場合は
   * 必ず実行構成を使用して、実行する「反復」回数を制御してください。
   * この演習では、1 ブロックのスレッドのみを使用します。
   */

  int N = 10;
  loop(N);
}
