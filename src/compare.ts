// 2つの評価結果を、同じ問題どうしで比べる(対応のある比較)
//
//   node src/cli.ts compare valid valid_sleep              eval/memory_results_<名前>.jsonl どうしを比べる
//   node src/cli.ts compare valid valid --method RAG       同じ実行の中で、記憶モデルと別の方式を比べる
//
// 平均の差が1〜2ポイントだと、件数が少ないうちは偶然と区別できない。
// 同じ問題で「良くなった数」と「悪くなった数」を数え、符号検定で偶然かどうかを確かめる。
// 平均の差の幅は、問題を選び直して計算し直す(ブートストラップ)ことで見積もる。

import fs from 'node:fs';
import path from 'node:path';
import { DATA } from './config.ts';

const OUT = path.join(DATA, 'eval');
const BOOTSTRAP = 5000;

function load(name: string): Map<string, any> {
  const lines = fs.readFileSync(path.join(OUT, `memory_results_${name}.jsonl`), 'utf8').split('\n').filter(Boolean);
  return new Map(lines.map((l) => JSON.parse(l)).filter((x) => x.scores).map((x) => [x.key, x]));
}

function binom(n: number, k: number): number {
  let r = 1;
  for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i;
  return r;
}

/** 勝ちと負けの数が、五分五分(違いがない)から偶然生じる確率(両側) */
function signTest(wins: number, losses: number): number {
  const n = wins + losses;
  if (!n) return 1;
  const k = Math.min(wins, losses);
  let p = 0;
  for (let i = 0; i <= k; i++) p += binom(n, i) / 2 ** n;
  return Math.min(1, 2 * p);
}

function compareOne(a: number[], b: number[], la: string, lb: string, metric: string): void {
  const diffs = a.map((x, i) => b[i] - x);
  const wins = diffs.filter((d) => d > 1e-9).length;
  const losses = diffs.filter((d) => d < -1e-9).length;
  const mean = diffs.reduce((s, d) => s + d, 0) / diffs.length;
  let seed = 7;
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648);
  const boots: number[] = [];
  for (let t = 0; t < BOOTSTRAP; t++) {
    let s = 0;
    for (let i = 0; i < diffs.length; i++) s += diffs[Math.floor(rand() * diffs.length)];
    boots.push(s / diffs.length);
  }
  boots.sort((x, y) => x - y);
  const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
  const sgn = (x: number) => (x >= 0 ? '+' : '') + pct(x);
  const p = signTest(wins, losses);
  const avg = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;
  console.log(`${metric}: ${la} ${pct(avg(a))} → ${lb} ${pct(avg(b))} (差 ${sgn(mean)}、95%の幅 ${sgn(boots[Math.floor(BOOTSTRAP * 0.025)])}〜${sgn(boots[Math.floor(BOOTSTRAP * 0.975)])})`);
  console.log(`  ${lb} が良い ${wins}件 / 悪い ${losses}件 / 同じ ${diffs.length - wins - losses}件 → p=${p.toFixed(3)} ${p < 0.05 ? '差がある' : '偶然と区別できない'}`);
}

export function compare(argv: string[]): void {
  const pos = argv.filter((a, i) => !a.startsWith('--') && argv[i - 1] !== '--method');
  const mi = argv.indexOf('--method');
  const method = mi >= 0 ? argv[mi + 1] : '記憶モデル';
  const noDev = argv.includes('--no-dev');
  const [base, other] = pos;
  const a = load(base), b = load(other);
  const keys = [...a.keys()].filter((k) => b.has(k) && !(noDev && a.get(k).dev));
  console.log(`共通の問題 ${keys.length}件`);
  const lb = method === '記憶モデル' ? other : `${other}:${method}`;
  ['再現率', '適合率'].forEach((metric, i) =>
    compareOne(keys.map((k) => a.get(k).scores['記憶モデル'][i]), keys.map((k) => b.get(k).scores[method][i]), base, lb, metric));
}
