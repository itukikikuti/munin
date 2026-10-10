"""2つの評価結果を、同じ問題どうしで比べる(対応のある比較)

  python compare_runs.py valid valid_sleep            ※ eval/memory_results_<名前>.jsonl どうしを比べる
  python compare_runs.py valid valid --method RAG     ※ 同じ実行の中で、記憶モデルと別の方式を比べる

平均の差が1〜2ポイントだと、件数が少ないうちは偶然と区別できない。
同じ問題で「良くなった数」と「悪くなった数」を数え、符号検定で偶然かどうかを確かめる。
平均の差の幅は、問題を選び直して計算し直す(ブートストラップ)ことで見積もる。
"""
import argparse
import json
import math
import os
import random
import sys

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("BRAIN_DATA", paths.DATA), "eval")
BOOTSTRAP = 5000


def load(name):
    with open(os.path.join(OUT, f"memory_results_{name}.jsonl"), encoding="utf-8") as f:
        return {x["key"]: x for x in map(json.loads, f) if x.get("scores")}


def sign_test(wins, losses):
    """勝ちと負けの数が、五分五分(違いがない)から偶然生じる確率(両側)"""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(a, b, label_a, label_b, metric):
    diffs = [b_ - a_ for a_, b_ in zip(a, b)]
    wins = sum(d > 1e-9 for d in diffs)
    losses = sum(d < -1e-9 for d in diffs)
    mean = sum(diffs) / len(diffs)
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP))
    lo, hi = boots[int(BOOTSTRAP * 0.025)], boots[int(BOOTSTRAP * 0.975)]
    p = sign_test(wins, losses)
    verdict = "差がある" if p < 0.05 else "偶然と区別できない"
    print(f"{metric}: {label_a} {sum(a) / len(a):.1%} → {label_b} {sum(b) / len(b):.1%} "
          f"(差 {mean:+.1%}、95%の幅 {lo:+.1%}〜{hi:+.1%})")
    print(f"  {label_b} が良い {wins}件 / 悪い {losses}件 / 同じ {len(diffs) - wins - losses}件 → p={p:.3f} {verdict}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--method", default="記憶モデル", help="other 側で比べる方式(base 側はいつも記憶モデル)")
    ap.add_argument("--no-dev", action="store_true", help="このプロジェクト自体を作っているやり取りを除く")
    args = ap.parse_args()
    a, b = load(args.base), load(args.other)
    keys = [k for k in a if k in b and not (args.no_dev and a[k].get("dev"))]
    print(f"共通の問題 {len(keys)}件")
    label_b = args.other if args.method == "記憶モデル" else f"{args.other}:{args.method}"
    for i, metric in enumerate(["再現率", "適合率"]):
        compare([a[k]["scores"]["記憶モデル"][i] for k in keys], [b[k]["scores"][args.method][i] for k in keys],
                args.base, label_b, metric)


if __name__ == "__main__":
    main()
