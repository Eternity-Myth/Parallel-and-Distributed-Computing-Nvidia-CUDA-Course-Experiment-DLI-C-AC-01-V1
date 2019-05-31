#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"

#define SOFTENING 1e-9f

/*
 * 各ボディには、x、y、z 座標の位置と x、y、z 方向の速度が含まれています。
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * この関数は変更しないでください。
 * この演習の制約として、この関数はホスト関数のままにします。
 */

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

/*
 * この関数は、システム内のすべてのボディが他のすべてに
 * 与える重力の影響を計算しますが、位置は更新しません。
 */

void bodyForce(Body *p, float dt, int n) {
  for (int i = 0; i < n; ++i) {
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = rsqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {

  /*
   * ここで `nBodies` の値を変更しないでください。
   * 変更したい場合は、コマンド ラインに値を渡してください。
   */

  int nBodies = 2<<11;
  int salt = 0;
  if (argc > 1) nBodies = 2<<atoi(argv[1]);

  /*
   * この salt は評価のためにあります。変更すると自動的にエラーが発生します。
   */

  if (argc > 2) salt = atoi(argv[2]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  int bytes = nBodies * sizeof(Body);
  float *buf;

  buf = (float *)malloc(bytes);

  Body *p = (Body*)buf;

  /*
   * この演習の制約として、`randomizeBodies` はホスト関数のままにする必要があります。
   */

  randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

  double totalTime = 0.0;

  /*
   * このシミュレーションは、10 サイクルの時間にわたって実行されます。
   * ボディ間の重力相互作用を計算し、ボディの位置を調整して反映させます。
   */

  /*******************************************************************/
  // Do not modify these 2 lines of code.
  for (int iter = 0; iter < nIters; iter++) {
    StartTimer();
  /*******************************************************************/

  /*
   * `bodyForce` で実行される処理だけでなく、位置を調整する処理もリファクタリングする場合があります。
   */

    bodyForce(p, dt, nBodies); // compute interbody forces

  /*
   * この位置の調整は、このラウンドの `bodyForce` が完了するまで実行できません。
   * また、次のラウンドの `bodyForce` は、調整が完了するまで開始できません。
   */

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

  /*******************************************************************/
  // Do not modify the code in this section.
    const double tElapsed = GetTimer() / 1000.0;
    totalTime += tElapsed;
  }

  double avgTime = totalTime / (double)(nIters);
  float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
  checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
  checkAccuracy(buf, nBodies);
  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
  salt += 1;
#endif
  /*******************************************************************/

  /*
   * 以下のコードは自由に変更できます。
   */

  free(buf);
}
