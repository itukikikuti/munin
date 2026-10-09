"""出来事の並べ方の重み(EP_W_*)を、評価の当たり外れから推定する

  python fit_ranking.py valid_import      ※ eval/memory_results_<名前>.jsonl を使う

記憶モデルは、候補の上位50件には正解の情報を半分近く拾えているのに、上位5件に絞ると半分以下になる。
最後の並べ方(要旨・概念・新しさ・細部・キーワード・場所の重みづけの和)を、手で決めた重みではなく、
実際に関係していたかどうかから推定する(ACT-R でパラメータを実験データから推定するのと同じ考え方)。

合わせ込みすぎを避けるため、評価期間の前半で推定して後半で確かめ、逆向きでも確かめる。
"""
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("BRAIN_DATA", HERE), "eval")
FEATURES = ["gist", "concept", "base", "detail", "keyword", "place"]
K = 5
GRID = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
BASE_GRID = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]
PLACE_GRID = [0.0, 0.25, 0.5, 1.0]


def prepare(items):
    """候補の特徴を行列に、当たった正解の情報をビットの並びにしておく(格子探索を速くするため)"""
    out = []
    for x in items:
        index = {c: i for i, c in enumerate(x["truth"])}
        feats = np.array([[c["f"].get(f, 0.0) for f in FEATURES] for c in x["cands"]])
        masks = [sum(1 << index[h] for h in c["hits"]) for c in x["cands"]]
        out.append((feats, masks, len(x["truth"])))
    return out


def recall_at_k(items, w):
    """重み w で候補を並べ替えたときの、上位 K 件の再現率の平均"""
    wv = np.array([w[f] for f in FEATURES])
    total = 0.0
    for feats, masks, n in items:
        top = np.argsort(-(feats @ wv), kind="stable")[:K]
        got = 0
        for i in top:
            got |= masks[i]
        total += bin(got).count("1") / n
    return total / len(items)


def fit(items):
    """要旨の重みを1に固定して(全体の大きさは並び順に関係しない)、残りの重みを格子で探す"""
    best, best_w = -1, None
    for c, d, k, p in itertools.product(GRID, GRID, GRID, PLACE_GRID):
        for b in BASE_GRID:
            w = {"gist": 1.0, "concept": c, "base": b, "detail": d, "keyword": k, "place": p}
            r = recall_at_k(items, w)
            if r > best:
                best, best_w = r, w
    return best_w, best


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    import brain
    name = sys.argv[1] if len(sys.argv) > 1 else "valid"
    with open(os.path.join(OUT, f"memory_results_{name}.jsonl"), encoding="utf-8") as f:
        items = prepare([x for x in map(json.loads, f) if x.get("cands")])
    now = {"gist": brain.EP_W_GIST, "concept": brain.EP_W_CONCEPT, "base": brain.EP_W_BASE,
           "detail": brain.EP_W_DETAIL, "keyword": brain.EP_W_KEYWORD, "place": brain.EP_W_PLACE}
    half = len(items) // 2
    a, b = items[:half], items[half:]
    print(f"{len(items)}件(前半 {len(a)} / 後半 {len(b)})")
    print(f"今の重み {now}: 前半 {recall_at_k(a, now):.1%} / 後半 {recall_at_k(b, now):.1%} / 全体 {recall_at_k(items, now):.1%}")
    for label, train, test in [("前半で推定 → 後半", a, b), ("後半で推定 → 前半", b, a)]:
        w, r = fit(train)
        print(f"{label}: 重み {w} 推定に使った側 {r:.1%} / 確かめた側 {recall_at_k(test, w):.1%}"
              f"(今の重みでは {recall_at_k(test, now):.1%})")
    w, r = fit(items)
    print(f"全体で推定: 重み {w} {r:.1%}")


if __name__ == "__main__":
    main()
