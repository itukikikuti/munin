"""実際のセッション記録で、関連することを思い出せるかを測る(LLM は使わない)

  python evaluate_memory.py                      ※ 調整用の期間で評価する(普段はこれ)
  python evaluate_memory.py --mode test          ※ 最終確認用の期間で評価する(調整が終わったときだけ)
  python evaluate_memory.py --ablate sleep       ※ 仕組みを1つ外して評価する(どの仕組みが効いているかを切り分ける)
  python evaluate_memory.py --train 900          ※ 学習に使うやり取りを、評価期間の直前の900件に絞る(速く試したいとき)

期間の分け方: TEST_FROM 以降は最終確認用に取り置き、調整のあいだは使わない。その手前の VALID_N 件を調整用にする。
パラメータを調整用の期間に合わせ込んでも、最終確認用の期間で本当の実力を確かめられる。

過去のやり取り(学習分)で記憶を作ったあと、評価分のやり取りを時刻順に1件ずつ、
「依頼から関連することを思い出させる → 実際の出来事を覚えさせる」の順で体験させる。

正解: そのやり取りで実際に使われた語・ファイル・場所(依頼と Claude の回答から取り出した概念)のうち、
      それより前から記憶にあったもの。ただし、依頼文にそのまま書いてある語、今の会話にすでに出てきた
      情報、今の作業フォルダ(とそれを含む場所)は除く(Claude がすでに知っているので、差し込む価値がない)。
どの方式も、今の会話のやり取りは思い出す対象から外す。
比べる方式(どれも過去のやり取りを5件思い出す):
  記憶モデル : 要旨・概念の活性・新しさ・細部を合わせた ACT-R の活性で選ぶ
  RAG        : 依頼と要旨の意味の近さだけで選ぶ
  直近       : 最新のやり取り5件
  キーワード : 依頼文に出てくる概念を多く含むやり取り
測るもの:
  再現率 : 正解の情報のうち、思い出したやり取りに含まれていた割合
  適合率 : 思い出したやり取りのうち、実際に関係していた(そのやり取りで使われた情報を2つ以上含む)割合
確信度は、評価分の前半で ACT-R の検索確率の閾値と傾きを推定し、後半で当たり具合を確かめる。

結果は eval/memory_report_<期間>[_<外した仕組み>].md に保存する。本番の記憶は変えない。
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import brain as brain_module
from brain import EP_S, EP_TAU, EPISODES_SHOWN, Brain, at_place, is_place, recollection_text
from embed import Embedder
from import_sessions import Life, load_turns, record_key, store_records
from store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("BRAIN_DATA", HERE)   # 埋め込みのキャッシュとやり取りの記録の置き場所(コードを別の場所にコピーして評価するとき用)
OUT = os.path.join(DATA, "eval")
K = EPISODES_SHOWN
METHODS = ["記憶モデル", "RAG", "直近", "キーワード"]
# 最終確認用の期間の始まり(UTC)。これ以降のやり取りは、調整のあいだは使わない。
# 日時で固定しておくと、新しいやり取りが増えても調整用の期間はずれず、増えた分は最終確認用に入る
TEST_FROM = "2026-10-05T03:00"
VALID_N = 300           # 調整用の期間のやり取りの数(最終確認用の期間の直前)
CONF_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
SECTIONS = ["todos", "recent", "episodes", "places", "concepts"]
CANDIDATES = 50        # 候補の段階での再現率を測るときの件数
DEV_PLACE = "brain-memory"


def is_dev(t):
    """brain-memory 自体を作っているやり取りか(試しに思い出させた質問や結果が多く、評価を歪めうる)"""
    return any(DEV_PLACE in p.lower() for p in t["places"] + [t["cwd"] or ""])


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["valid", "test"], default="valid")
    ap.add_argument("--train", type=int, default=None, help="学習に使う、評価期間の直前のやり取りの数")
    ap.add_argument("--ablate", default="", help="外す仕組み(カンマ区切り): " + ", ".join(brain_module.ABLATIONS))
    ap.add_argument("--tag", default="", help="結果のファイル名に付ける名前(設定を変えて比べるとき)")
    ap.add_argument("--set", action="append", default=[], metavar="名前=値",
                    help="brain.py の定数を変えて評価する(例: --set SUPERSEDE_PENALTY=0.3)")
    args = ap.parse_args()
    for kv in args.set:
        name, value = kv.split("=", 1)
        if not hasattr(brain_module, name):
            raise SystemExit(f"brain.py に {name} という定数はない")
        setattr(brain_module, name, type(getattr(brain_module, name))(eval(value)))
    brain_module.ABLATE = {a for a in args.ablate.split(",") if a}
    unknown = brain_module.ABLATE - set(brain_module.ABLATIONS)
    if unknown:
        raise SystemExit(f"知らない仕組み: {', '.join(unknown)}")
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    store = Store(os.path.join(DATA, "records.sqlite"))   # やり取りの記録は事実なので、本番と共有する
    turns = load_turns(store=store)
    n_valid = sum(t["t"] < TEST_FROM for t in turns)
    if args.mode == "valid":
        turns = turns[:n_valid]          # 最終確認用の期間は見ない
        train, test = turns[:-VALID_N], turns[-VALID_N:]
    else:
        train, test = turns[:n_valid], turns[n_valid:]
    if args.train:
        train = train[-args.train:]
        turns = train + test
    print(f"学習 {len(train)}件 ({train[0]['t'][:10]}〜{train[-1]['t'][:10]}) / "
          f"評価 {len(test)}件 ({test[0]['t'][:10]}〜{test[-1]['t'][:10]})")
    embedder = Embedder(os.path.join(DATA, "embeddings.json"))
    embedder.ensure_gists([t["user"] for t in turns], progress=True)

    state = os.path.join(OUT, "state_memory.json")
    if os.path.exists(state):
        os.remove(state)
    store_records(store, turns)
    brain = Brain(state, embedder=embedder, store=store)
    brain.check_places = False   # 過去の時点で思い出すので、今の時点で場所があるかは確かめない(未来の情報が混ざる)
    life = Life(brain, turns[0])
    for i, t in enumerate(train):
        if i % 50 == 0:
            print(f"\r  学習 {i}/{len(train)}", end="", file=sys.stderr, flush=True)
        life.experience(t)
    print(file=sys.stderr)
    embedder.ensure(sorted(brain.known_concepts()), progress=True)
    # 「それより前から記憶にあった」かどうかは、記憶モデルの状態(忘れたかどうか)ではなく、
    # 過去のやり取りに出てきたかどうかで決める(睡眠を外すなど、記憶モデルを変えても正解の範囲が変わらないように)
    seen = {c for tr in brain.traces for c in tr["concepts"]} | brain.known_concepts()

    results = []
    prev_session = None
    for i, t in enumerate(test):
        print(f"\r  評価 {i}/{len(test)}", end="", file=sys.stderr, flush=True)
        life.advance(t)
        first = t["session"] != prev_session
        prev_session = t["session"]
        actual = life.episode(t)
        prompt = t["user"].lower()
        session = os.path.basename(t["session"])
        in_conversation = {c for tr in brain.traces if tr.get("session") == session for c in tr["concepts"]}
        # 今の作業フォルダ(とそれを含む場所)は、Claude がすでに知っているので、思い出す価値がない
        here = {c for c in actual if t["cwd"] and is_place(c) and at_place(t["cwd"], c)}
        truth = {c for c in actual if c in seen and c.lower() not in prompt and c not in in_conversation
                 and c not in here}

        r = brain.recollect(t["user"], session=session, cwd=t["cwd"])
        sims = brain.last_trace_sim
        traces = [tr for tr in brain.traces if tr.get("session") != session]
        rag_all = sorted(traces, key=lambda tr: -sims.get(id(tr), 0))
        rag = rag_all[:K]
        recent = sorted(traces, key=lambda tr: -tr["t"])[:K]
        kw = sorted(traces, key=lambda tr: (-sum(c.lower() in prompt for c in tr["concepts"]), -tr["t"]))[:K]
        picks = {"記憶モデル": [e["concepts"] for e in r["ranked"][:K]], "RAG": [tr["concepts"] for tr in rag],
                 "直近": [tr["concepts"] for tr in recent], "キーワード": [tr["concepts"] for tr in kw]}
        shown = ([e["concepts"] for e in r["episodes"]] + [[c["name"] for c in r["concepts"]]]
                 + [[p["name"] for p in r["places"]]] + [e["concepts"] for e in r["recent"]])
        results.append({
            "key": record_key(t), "prompt": t["user"][:150], "first": first, "truth": sorted(truth),
            "dev": is_dev(t),
            "scores": {m: score(picks[m], truth, set(actual)) for m in METHODS} if truth else None,
            # 候補の段階(上位 CANDIDATES 件)で、正解の情報を含むやり取りが拾えていたか
            "candidate_recall": {
                "記憶モデル": score([tr["concepts"] for tr in brain.last_candidates[:CANDIDATES]], truth, set(actual))[0],
                "RAG": score([tr["concepts"] for tr in rag_all[:CANDIDATES]], truth, set(actual))[0],
            } if truth else None,
            "shown_recall": score(shown, truth, set(actual))[0] if truth else None,
            "episodes": [{"activation": e["activation"], "relevant": relevant(e["concepts"], set(actual))}
                         for e in r["ranked"]],
            "text_len": len(r["text"]),
            # 差し込む文章のうち、各部分(やること・直前の作業・出来事・場所・語)が占める長さ
            "text_parts": {k: len(recollection_text({**r, **{o: [] for o in SECTIONS if o != k}})) for k in SECTIONS},
            # 候補ごとの特徴と当たり外れ(fit_ranking.py で、並べ方の重みを推定するのに使う)
            "cands": [{"f": brain.last_features[id(tr)], "rel": relevant(tr["concepts"], set(actual)),
                       "hits": sorted(set(tr["concepts"]) & truth)} for tr in brain.last_candidates[:CANDIDATES]]
                     if truth else None,
            # 差し込む文章の中に、実際に使われた情報がそのまま出てくる割合
            "text_recall": sum(c.lower() in r["text"].lower() for c in truth) / len(truth) if truth else None,
        })
        life.remember(t, concepts=actual)
        seen |= set(actual)
    print(file=sys.stderr)

    name = args.mode + (f"_{'_'.join(sorted(brain_module.ABLATE))}" if brain_module.ABLATE else "") \
        + (f"_{args.tag}" if args.tag else "")
    # 1件ずつの結果(compare_runs.py で、同じ問題どうしの勝ち負けを比べるのに使う)
    with open(os.path.join(OUT, f"memory_results_{name}.jsonl"), "w", encoding="utf-8") as f:
        for x in results:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    report = make_report(results, train, test, time.time() - t0, args)
    with open(os.path.join(OUT, f"memory_report_{name}.md"), "w", encoding="utf-8") as f:
        f.write(report)
    print(report)


def make_report(results, train, test, seconds, args):
    scored = [x for x in results if x["scores"]]
    first = [x for x in scored if x["first"]]
    later = [x for x in scored if not x["first"]]
    mean = lambda xs, m, k: sum(x["scores"][m][k] for x in xs) / max(len(xs), 1)
    lines = ["# 関連することを思い出せるかの自動評価", "",
             f"- 期間: {'調整用' if args.mode == 'valid' else '最終確認用'}"
             + (f" / 外した仕組み: {', '.join(sorted(brain_module.ABLATE))}" if brain_module.ABLATE else ""),
             f"- 変えた設定: {', '.join(args.set) or 'なし'}",
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
    clean = [x for x in scored if not x["dev"]]
    lines += ["", f"brain-memory 自体を作っているやり取り({len(scored) - len(clean)}件)を除いた {len(clean)}件では:", "",
              "| 方式 | 再現率 | 適合率 |", "|---|---|---|"]
    for m in METHODS:
        lines.append(f"| {m} | {mean(clean, m, 0):.1%} | {mean(clean, m, 1):.1%} |")
    cand = lambda m: sum(x["candidate_recall"][m] for x in scored) / max(len(scored), 1)
    lines += ["", f"候補の段階(上位{CANDIDATES}件)での再現率: 記憶モデル {cand('記憶モデル'):.1%} / RAG {cand('RAG'):.1%}。"
              f"上位{K}件との差が、最後の並べ方で落としている分。"]
    shown_rec = sum(x["shown_recall"] for x in scored) / max(len(scored), 1)
    text_rec = sum(x["text_recall"] for x in scored) / max(len(scored), 1)
    avg_len = sum(x["text_len"] for x in results) / max(len(results), 1)
    # 正解の情報がない(過去の記憶が要らなかった)やり取りでは、何も差し込まないのが理想
    idle = [x for x in results if not x["scores"] and not x["first"]]
    idle_len = sum(x["text_len"] for x in idle) / max(len(idle), 1)
    idle_empty = sum(x["text_len"] == 0 for x in idle) / max(len(idle), 1)
    part = lambda k: sum(x["text_parts"][k] for x in results) / max(len(results), 1)
    parts = "、".join(f"{name} {part(k):.0f}" for k, name in
                     zip(SECTIONS, ["やること", "直前の作業", "出来事", "場所", "語"]))
    lines += ["", f"記憶モデルが実際に差し込む内容(確かさで絞ったやり取り+語・場所+直前の作業)の再現率は "
              f"{shown_rec:.1%}。差し込む文章の中に、実際に使われた情報がそのまま書かれている割合は {text_rec:.1%}。"
              f"差し込む文章の長さは平均 {avg_len:.0f}文字(部分ごとの平均: {parts}。それぞれ単独で出したときの長さ)。",
              f"過去の記憶が要らなかったやり取り(会話の途中、{len(idle)}件)では、平均 {idle_len:.0f}文字を差し込み、"
              f"{idle_empty:.0%}で何も差し込まなかった。", ""]

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
