"""実際のセッション記録で、関連することを思い出せるかを測る(LLM は使わない)

  python evaluate_memory.py              ※ 学習 900件 → 評価 300件
  python evaluate_memory.py 900 300

過去のやり取り(学習分)で記憶を作ったあと、その後のやり取り(評価分)を時刻順に1件ずつ、
「依頼から関連することを思い出させる → 実際の出来事を覚えさせる」の順で体験させる。

正解: そのやり取りで実際に使われた語・ファイル・場所(依頼と Claude の回答から取り出した概念)のうち、
      それより前から記憶にあったもの。ただし、依頼文にそのまま書いてある語と、今の会話にすでに出てきた
      情報は除く(Claude が会話の中ですでに知っているので、差し込む価値がない)。
どの方式も、今の会話のやり取りは思い出す対象から外す。
比べる方式(どれも過去のやり取りを5件思い出す):
  記憶モデル : 要旨・概念の活性・新しさを合わせた ACT-R の活性で選ぶ
  RAG        : 依頼と要旨の意味の近さだけで選ぶ
  直近       : 最新のやり取り5件
  キーワード : 依頼文に出てくる概念を多く含むやり取り
測るもの:
  再現率 : 正解の情報のうち、思い出したやり取りに含まれていた割合
  適合率 : 思い出したやり取りのうち、実際に関係していた(そのやり取りで使われた情報を2つ以上含む)割合
確信度は、評価分の前半で ACT-R の検索確率の閾値と傾きを推定し、後半で当たり具合を確かめる。

結果は eval/memory_report.md と eval/memory_results.jsonl に保存する。本番の記憶は変えない。
"""
import json
import math
import os
import sys
import time

import numpy as np

from brain import EP_S, EP_TAU, EPISODES_SHOWN, Brain
from embed import Embedder
from import_sessions import Life, load_turns, store_records
from store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval")
K = EPISODES_SHOWN
METHODS = ["記憶モデル", "RAG", "直近", "キーワード"]
CONF_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def relevant(concepts, used):
    """思い出したやり取りが、実際に関係していたか(そのやり取りで使われた情報を2つ以上含む)"""
    return len(set(concepts) & used) >= 2


def score(episodes, truth, used):
    """思い出したやり取りの集まりについて、(再現率, 適合率)

    truth: 差し込む価値のある情報(今の会話にまだ出ていない、使われた情報)
    used : そのやり取りで使われた情報すべて
    """
    info = set().union(*[set(e) for e in episodes]) if episodes else set()
    rec = len(info & truth) / len(truth)
    prec = sum(relevant(e, used) for e in episodes) / len(episodes) if episodes else 0.0
    return rec, prec


def fit_retrieval(acts, labels, steps=3000, lr=0.5):
    """ACT-R の検索確率 P = 1 / (1 + exp(-(A - τ) / s)) の τ と s を、実際の当たり外れから推定する(ロジスティック回帰)"""
    x = np.array(acts)
    y = np.array(labels, float)
    mu, sd = x.mean(), x.std() or 1.0
    z = (x - mu) / sd
    w, b = 1.0, 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(w * z + b)))
        w -= lr * ((p - y) * z).mean()
        b -= lr * (p - y).mean()
    s = sd / w if w else EP_S
    return mu - b * s, s


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    turns = load_turns(n_train + n_test)
    train, test = turns[:-n_test], turns[-n_test:]
    print(f"学習 {len(train)}件 ({train[0]['t'][:10]}〜{train[-1]['t'][:10]}) / "
          f"評価 {len(test)}件 ({test[0]['t'][:10]}〜{test[-1]['t'][:10]})")
    embedder = Embedder(os.path.join(HERE, "embeddings.json"))
    embedder.ensure_gists([t["user"] for t in turns], progress=True)

    state = os.path.join(OUT, "state_memory.json")
    if os.path.exists(state):
        os.remove(state)
    store = Store(os.path.join(HERE, "records.sqlite"))   # やり取りの記録は事実なので、本番と共有する
    store_records(store, turns)
    brain = Brain(state, embedder=embedder, store=store)
    life = Life(brain, turns[0])
    for i, t in enumerate(train):
        if i % 50 == 0:
            print(f"\r  学習 {i}/{len(train)}", end="", file=sys.stderr, flush=True)
        life.experience(t)
    print(file=sys.stderr)
    embedder.ensure(sorted(brain.known_concepts()), progress=True)

    results = []
    prev_session = None
    for i, t in enumerate(test):
        print(f"\r  評価 {i}/{len(test)}", end="", file=sys.stderr, flush=True)
        life.advance(t)
        first = t["session"] != prev_session
        prev_session = t["session"]
        known = brain.known_concepts()
        actual = life.episode(t)
        prompt = t["user"].lower()
        session = os.path.basename(t["session"])
        in_conversation = {c for tr in brain.traces if tr.get("session") == session for c in tr["concepts"]}
        truth = {c for c in actual if c in known and c.lower() not in prompt and c not in in_conversation}

        r = brain.recollect(t["user"], session=session)
        sims = brain.last_trace_sim
        traces = [tr for tr in brain.traces if tr.get("session") != session]
        rag = sorted(traces, key=lambda tr: -sims.get(id(tr), 0))[:K]
        recent = sorted(traces, key=lambda tr: -tr["t"])[:K]
        kw = sorted(traces, key=lambda tr: (-sum(c.lower() in prompt for c in tr["concepts"]), -tr["t"]))[:K]
        picks = {"記憶モデル": [e["concepts"] for e in r["ranked"][:K]], "RAG": [tr["concepts"] for tr in rag],
                 "直近": [tr["concepts"] for tr in recent], "キーワード": [tr["concepts"] for tr in kw]}
        shown = ([e["concepts"] for e in r["episodes"]] + [[c["name"] for c in r["concepts"]]]
                 + [[p["name"] for p in r["places"]]] + [e["concepts"] for e in r["recent"]])
        results.append({
            "prompt": t["user"][:150], "first": first, "truth": sorted(truth),
            "scores": {m: score(picks[m], truth, set(actual)) for m in METHODS} if truth else None,
            "shown_recall": score(shown, truth, set(actual))[0] if truth else None,
            "episodes": [{"activation": e["activation"], "relevant": relevant(e["concepts"], set(actual))}
                         for e in r["ranked"]],
            "text_len": len(r["text"]),
            # 差し込む文章の中に、実際に使われた情報がそのまま出てくる割合
            "text_recall": sum(c.lower() in r["text"].lower() for c in truth) / len(truth) if truth else None,
        })
        life.remember(t, concepts=actual)
    print(file=sys.stderr)

    with open(os.path.join(OUT, "memory_results.jsonl"), "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    report = make_report(results, train, test, time.time() - t0)
    with open(os.path.join(OUT, "memory_report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def make_report(results, train, test, seconds):
    scored = [x for x in results if x["scores"]]
    first = [x for x in scored if x["first"]]
    later = [x for x in scored if not x["first"]]
    mean = lambda xs, m, k: sum(x["scores"][m][k] for x in xs) / max(len(xs), 1)
    lines = ["# 関連することを思い出せるかの自動評価", "",
             f"- 学習: {len(train)}件({train[0]['t'][:10]}〜{train[-1]['t'][:10]})",
             f"- 評価: {len(test)}件({test[0]['t'][:10]}〜{test[-1]['t'][:10]})。"
             f"正解の情報(以前から記憶にあった、実際に使われた情報)があるもの {len(scored)}件",
             f"- 所要時間: {seconds:.0f}秒", "",
             f"## 再現率と適合率(どの方式も過去のやり取りを{K}件思い出す)", "",
             "再現率: 実際に使われた情報のうち、思い出したやり取りに含まれていた割合。"
             "適合率: 思い出したやり取りのうち、実際に関係していた割合。", "",
             "| 方式 | 再現率(全体) | 適合率(全体) | 再現率(会話の最初) | 再現率(会話の途中) |",
             "|---|---|---|---|---|"]
    for m in METHODS:
        lines.append(f"| {m} | {mean(scored, m, 0):.1%} | {mean(scored, m, 1):.1%} | "
                     f"{mean(first, m, 0):.1%}({len(first)}件) | {mean(later, m, 0):.1%}({len(later)}件) |")
    shown_rec = sum(x["shown_recall"] for x in scored) / max(len(scored), 1)
    text_rec = sum(x["text_recall"] for x in scored) / max(len(scored), 1)
    avg_len = sum(x["text_len"] for x in results) / max(len(results), 1)
    lines += ["", f"記憶モデルが実際に差し込む内容(確かさで絞ったやり取り+語・場所+直前の作業)の再現率は "
              f"{shown_rec:.1%}。差し込む文章の中に、実際に使われた情報がそのまま書かれている割合は {text_rec:.1%}。"
              f"差し込む文章の長さは平均 {avg_len:.0f}文字。", ""]

    # 確信度: 前半で ACT-R の検索確率の τ と s を推定し、後半で確かめる
    half = len(results) // 2
    fit_data = [(e["activation"], e["relevant"]) for x in results[:half] for e in x["episodes"]]
    tau, s = fit_retrieval([a for a, _ in fit_data], [r for _, r in fit_data])
    later_eps = [e for x in results[half:] for e in x["episodes"]]
    lines += ["## 確信度(思い出したやり取りが実際に関係していた割合)", "",
              "評価分の前半で、ACT-R の検索確率の閾値 τ と傾き s を、実際の当たり外れから推定した"
              f"(今の設定 τ={EP_TAU}, s={EP_S} → 推定 τ={tau:.3f}, s={s:.3f})。後半で当たり具合を比べる。"
              "確信度が正しく働いていれば、確信度と正解率がおおむね一致する。", "",
              "| 確信度 | 今の設定: 件数 | 今の設定: 正解率 | 推定後: 件数 | 推定後: 正解率 |", "|---|---|---|---|---|"]
    prob = lambda a, t, s_: 1 / (1 + math.exp(-(a - t) / s_))
    acc = lambda es: f"{sum(e['relevant'] for e in es) / len(es):.1%}" if es else "-"
    for lo, hi in CONF_BINS:
        cur = [e for e in later_eps if lo <= prob(e["activation"], EP_TAU, EP_S) < hi]
        fit = [e for e in later_eps if lo <= prob(e["activation"], tau, s) < hi]
        lines.append(f"| {lo:.1f}〜{min(hi, 1):.1f} | {len(cur)} | {acc(cur)} | {len(fit)} | {acc(fit)} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
