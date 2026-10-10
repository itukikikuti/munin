// ネットワーク図の配置(力学モデル)。重いので、常駐サーバーとは別のスレッドで計算する
//
// すでに配置済みの点はそのまま使い、新しい点だけ、つながっている点の近くに置いてから全体を整える。
// 点どうしは反発し、つながっている点どうしはばねで引き合い、全体は中心へ少し引かれる。

import { parentPort, workerData } from 'node:worker_threads';

const { n, edges, known } = workerData as { n: number; edges: Float64Array; known: ([number, number] | null)[] };
const X = new Float64Array(2 * n);
const isKnown = new Uint8Array(n);
known.forEach((p, i) => {
  if (p) {
    X[2 * i] = p[0];
    X[2 * i + 1] = p[1];
    isKnown[i] = 1;
  }
});
const E = edges.length / 4;
const nbs: number[][] = Array.from({ length: n }, () => []);
for (let k = 0; k < E; k++) {
  nbs[edges[4 * k]].push(edges[4 * k + 1]);
  nbs[edges[4 * k + 1]].push(edges[4 * k]);
}
// 再現できるように、決まった乱数を使う
let seed = 1;
const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
const spread = Math.sqrt(n) * 40;
let fresh = 0;
for (let i = 0; i < n; i++) {
  if (isKnown[i]) continue;
  fresh++;
  const nb = nbs[i].find((j) => isKnown[j]);
  const base = nb === undefined ? [0, 0] : [X[2 * nb], X[2 * nb + 1]];
  const r = nb === undefined ? spread : 80;
  X[2 * i] = base[0] + rand() * r;
  X[2 * i + 1] = base[1] + rand() * r;
}
if (fresh > 0) {
  const iters = fresh > n * 0.2 ? 300 : 80;
  const F = new Float64Array(2 * n);
  for (let it = 0; it < iters; it++) {
    const t = 1 - it / iters;
    F.fill(0);
    for (let i = 0; i < n; i++) {   // 反発(全ての点の組)
      const xi = X[2 * i], yi = X[2 * i + 1];
      let fx = 0, fy = 0;
      for (let j = 0; j < n; j++) {
        const dx = xi - X[2 * j], dy = yi - X[2 * j + 1];
        const d2 = dx * dx + dy * dy + 0.01;
        const k = 6000 / (d2 * Math.sqrt(d2));
        fx += dx * k;
        fy += dy * k;
      }
      F[2 * i] += fx;
      F[2 * i + 1] += fy;
    }
    for (let k = 0; k < E; k++) {   // ばね(つながっている点の組)
      const a = edges[4 * k], b = edges[4 * k + 1], w = edges[4 * k + 2];
      const length = edges[4 * k + 3] === 1 ? 60 : 110;
      const dx = X[2 * b] - X[2 * a], dy = X[2 * b + 1] - X[2 * a + 1];
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const f = ((dist - length) * 0.03 * (0.3 + w)) / dist;
      F[2 * a] += dx * f; F[2 * a + 1] += dy * f;
      F[2 * b] -= dx * f; F[2 * b + 1] -= dy * f;
    }
    const lim = 25 * t + 1;
    for (let i = 0; i < n; i++) {
      let fx = F[2 * i] - X[2 * i] * 0.004, fy = F[2 * i + 1] - X[2 * i + 1] * 0.004;   // 中心へ
      const m = Math.sqrt(fx * fx + fy * fy);
      if (m > lim) { fx *= lim / m; fy *= lim / m; }
      X[2 * i] += fx;
      X[2 * i + 1] += fy;
    }
  }
}
parentPort!.postMessage(X);
