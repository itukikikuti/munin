"""実際のセッション記録で、思い出す精度を測る(合わせ込みのない自動評価)

  python evaluate_sessions.py              ※ 学習 900件 → 評価 300件
  python evaluate_sessions.py 900 300

過去のやり取り(学習分)で記憶を作ったあと、その後のやり取り(評価分)を時刻順に1件ずつ、
「指示から場所を思い出させる → 実際の出来事を覚えさせる」の順で体験させる(予測してから学ぶ)。
正解は、そのやり取りで実際に作業したフォルダ(作業フォルダと、編集・読み込みしたファイルの場所)。
どの時点の予測も、それより前のやり取りしか使っていない。

単純な方式とも比べる:
  直前と同じ   : 前回作業したプロジェクト
  最頻         : これまで一番多く作業したプロジェクト
  キーワード   : 語が一番重なる過去の指示のプロジェクト(TF-IDF)
  埋め込み検索 : 意味が一番近い過去の指示のプロジェクト(RAG と同じやり方)

結果は eval/report.md と eval/results.jsonl に保存する。本番の記憶(brain_state.json)は変えない。
"""
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

from brain import DEICTIC, Brain, content_words
from embed import Embedder
from import_sessions import Life, analyze, load_turns, nouns, project_root

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "eval")
PROMPT_CHARS = 2000      # 埋め込み検索で使う指示の長さ(貼り付けたログなどで長すぎるものは先頭だけ)
CONF_BINS = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
METHODS = ["記憶モデル(文脈あり)", "記憶モデル(文脈なし)", "直前と同じ", "最頻", "キーワード", "埋め込み検索"]


def project(p):
    return project_root(p) or p if p else None


class PromptIndex:
    """過去の指示の埋め込み(RAG 方式の比較用)。計算したものはファイルに保存して使い回す"""

    def __init__(self, embedder):
        self.e = embedder
        self.path = os.path.join(OUT, "prompt_embeddings")
        self.vecs = {}
        if os.path.exists(self.path + ".npy"):
            keys = json.load(open(self.path + ".json", encoding="utf-8"))
            self.vecs = dict(zip(keys, np.load(self.path + ".npy")))

    @staticmethod
    def key(text):
        return hashlib.sha1(text[:PROMPT_CHARS].encode()).hexdigest()

    def ensure(self, texts):
        missing = list(dict.fromkeys(t[:PROMPT_CHARS] for t in texts if self.key(t) not in self.vecs))
        for i in range(0, len(missing), 16):
            batch = missing[i:i + 16]
            for t, v in zip(batch, self.e._request(batch)):
                self.vecs[self.key(t)] = v
            print(f"\r  指示の埋め込み {min(i + 16, len(missing))}/{len(missing)}", end="", file=sys.stderr, flush=True)
        if missing:
            print(file=sys.stderr)
            keys = list(self.vecs)
            np.save(self.path + ".npy", np.array([self.vecs[k] for k in keys]))
            json.dump(keys, open(self.path + ".json", "w", encoding="utf-8"))

    def get(self, text):
        return self.vecs[self.key(text)]


class Baselines:
    """比較用の単純な方式(どれも、その時点より前のやり取りだけを使う)"""

    def __init__(self, prompts, vocab):
        self.prompts = prompts
        self.vocab = vocab
        self.prev = None
        self.freq = Counter()
        self.past = []        # (語の重み, 埋め込み, プロジェクト)

    def tfidf(self, terms):
        v = {w: c * math.log((self.vocab.turns + 1) / (self.vocab.df[w] + 1)) for w, c in Counter(terms).items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1
        return {w: x / n for w, x in v.items()}

    def predict(self, t):
        out = {"直前と同じ": self.prev, "最頻": self.freq.most_common(1)[0][0] if self.freq else None,
               "キーワード": None, "埋め込み検索": None}
        if self.past:
            q = self.tfidf(t["user_terms"])
            scores = [sum(x * kw.get(w, 0) for w, x in q.items()) for kw, _, _ in self.past]
            best = int(np.argmax(scores))
            if scores[best] > 0:
                out["キーワード"] = self.past[best][2]
            E = np.array([e for _, e, _ in self.past])
            out["埋め込み検索"] = self.past[int(np.argmax(E @ self.prompts.get(t["user"])))][2]
        return out

    def learn(self, t, proj):
        if proj:
            self.prev = proj
            self.freq[proj] += 1
            self.past.append((self.tfidf(t["user_terms"]), self.prompts.get(t["user"]), proj))


def main():
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
    prompts = PromptIndex(embedder)
    prompts.ensure([t["user"] for t in turns])

    state = os.path.join(OUT, "state.json")
    if os.path.exists(state):
        os.remove(state)
    brain = Brain(state, embedder=embedder)
    life = Life(brain, turns[0])
    base = Baselines(prompts, life.vocab)

    # 学習: 過去のやり取りを体験する(比較用の方式にも同じものを見せる)
    for i, t in enumerate(train):
        if i % 50 == 0:
            print(f"\r  学習 {i}/{len(train)}", end="", file=sys.stderr, flush=True)
        life.advance(t)
        analyze(t)
        base.learn(t, project(t["places"][0]) if t["places"] else None)
        life.remember(t)
    print(file=sys.stderr)
    embedder.ensure(sorted(brain.known_concepts()), progress=True)
    print(f"学習後の記憶: 概念 {len(brain.known_concepts())}個 / 皮質のリンク {len(brain.cortex)}本 / "
          f"海馬 {len(brain.hippocampus)}件 / 眠った回数 {life.nights}晩")

    # 評価: 指示から思い出させてから、実際の出来事を覚えさせる
    results = []
    for i, t in enumerate(test):
        print(f"\r  評価 {i}/{len(test)}", end="", file=sys.stderr, flush=True)
        life.advance(t)
        analyze(t)
        truth = t["places"]
        _, _, mem = brain.recall(t["user"], use_context=False, reinforce=False)
        _, _, ctx = brain.recall(t["user"])
        preds = base.predict(t)
        results.append({
            "t": t["t"], "prompt": t["user"][:200], "truth": truth,
            "referential": bool(DEICTIC.search(t["user"])) or not content_words(t["user"][:500]),
            "記憶モデル(文脈あり)": {"place": ctx["place"], "conf": ctx["confidence"], "uncertain": ctx["uncertain"],
                                 "ambiguous": ctx["ambiguous"]},
            "記憶モデル(文脈なし)": {"place": mem["place"], "conf": mem["confidence"], "uncertain": mem["uncertain"],
                                 "ambiguous": mem["ambiguous"]},
            **{k: {"place": v, "conf": 1.0, "uncertain": False} for k, v in preds.items()},
        })
        base.learn(t, project(truth[0]) if truth else None)
        life.remember(t)
    print(file=sys.stderr)

    with open(os.path.join(OUT, "results.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = make_report(results, train, test, time.time() - t0)
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def judge(r, method):
    """正解か(プロジェクト単位)、フォルダまで正解か、答えたか(うろ覚え・答えなしは答えていない扱い)"""
    p = r[method]
    answered = p["place"] is not None and not p["uncertain"]
    truth_proj = {project(x) for x in r["truth"]}
    hit = answered and project(p["place"]) in truth_proj
    exact = answered and p["place"] in r["truth"]
    return answered, hit, exact


def make_report(results, train, test, seconds):
    with_truth = [r for r in results if r["truth"]]
    without = [r for r in results if not r["truth"]]
    lines = [f"# 想起の自動評価", "",
             f"- 学習: {len(train)}件({train[0]['t'][:10]}〜{train[-1]['t'][:10]})",
             f"- 評価: {len(test)}件({test[0]['t'][:10]}〜{test[-1]['t'][:10]})。"
             f"うち作業した場所が分かるもの {len(with_truth)}件、場所なし {len(without)}件",
             f"- 所要時間: {seconds:.0f}秒", "",
             "## 正解率(作業した場所が分かる指示)", "",
             "正解率は全件に対する割合(答えなかった分は不正解)。適合率は、答えた分のうち正解だった割合。", "",
             "| 方式 | 正解率 | 答えた割合 | 適合率 | フォルダまで正解 |", "|---|---|---|---|---|"]
    for m in METHODS:
        js = [judge(r, m) for r in with_truth]
        ans = sum(a for a, _, _ in js)
        hit = sum(h for _, h, _ in js)
        exact = sum(e for _, _, e in js)
        n = len(js) or 1
        lines.append(f"| {m} | {hit / n:.1%} | {ans / n:.1%} | {hit / max(ans, 1):.1%} | {exact / n:.1%} |")

    lines += ["", "## 指示の種類別の正解率", "",
              "指示語(「この」「それ」)を含む・対象が省かれている指示と、それ以外で分けた。", "",
              "| 方式 | 文脈を指す指示 | それ以外 |", "|---|---|---|"]
    ref = [r for r in with_truth if r["referential"]]
    other = [r for r in with_truth if not r["referential"]]
    for m in METHODS:
        acc = lambda rs: f"{sum(judge(r, m)[1] for r in rs) / max(len(rs), 1):.1%}({len(rs)}件)"
        lines.append(f"| {m} | {acc(ref)} | {acc(other)} |")

    lines += ["", "## 確信度と実際の正解率(記憶モデル・文脈あり)", "",
              "確信度が正しく働いていれば、確信度が高いほど正解率も高くなる。", "",
              "| 確信度 | 件数 | 正解率 |", "|---|---|---|"]
    m = METHODS[0]
    for lo, hi in CONF_BINS:
        rs = [r for r in with_truth if r[m]["place"] and lo <= r[m]["conf"] < hi]
        hits = sum(project(r[m]["place"]) in {project(x) for x in r["truth"]} for r in rs)
        lines.append(f"| {lo:.2f}〜{min(hi, 1):.2f} | {len(rs)} | {hits / max(len(rs), 1):.1%} |")
    none = sum(1 for r in with_truth if not r[m]["place"])
    lines.append(f"| 何も思い浮かばない | {none} | - |")

    answered_without = sum(1 for r in without if judge(r, m)[0])
    lines += ["", "## 場所が分からない指示", "",
              f"どのプロジェクトでもない作業(ホームでの雑談など){len(without)}件のうち、"
              f"記憶モデル(文脈あり)がどこかの場所を答えたのは {answered_without}件。", ""]

    lines += ["## 間違えた例(記憶モデル・文脈あり、最初の25件)", "",
              "| 指示 | 正解 | 答え | 確信度 | 埋め込み検索の答え |", "|---|---|---|---|---|"]
    misses = [r for r in with_truth if not judge(r, m)[1]][:25]
    for r in misses:
        prompt = r["prompt"].replace("\n", " ").replace("|", "/")[:50]
        a = r[m]
        lines.append(f"| {prompt} | {project(r['truth'][0])} | {a['place'] or '(なし)'}"
                     f"{'(うろ覚え)' if a['uncertain'] else ''} | {a['conf']:.2f} | {r['埋め込み検索']['place']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
