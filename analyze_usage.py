"""Claude Code で使ったときの記録(logs/recall_log.jsonl)から、差し込んだ記憶が実際に使われたかを調べる

  python analyze_usage.py              ※ 全期間
  python analyze_usage.py --since 2026-10-15

フック(hook.py)は、メッセージごとに「何を思い出して差し込んだか」を記録している。
それを、そのあと Claude が実際にしたこと(セッション記録の返答とツールの実行)と突き合わせる。

使われたとみなす証拠:
  開いた   : Claude が memory_detail.py でその記憶の鍵を開いた(はっきり使った)
  語が出た : その記憶に含まれる語・ファイル・場所(依頼文にない、2文字以上のもの)が、
             そのあとの Claude の返答や操作に出てきた(使ったかもしれない。偶然の一致も含む)
確信度ごとに使われた割合を出し、確信度が当てになっているか(高いものほど使われているか)を見る。
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

from brain import Brain, is_place

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "logs", "recall_log.jsonl")
BINS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]


def text_of(content):
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text")


def after_prompt(transcript, prompt):
    """セッション記録の中で、その依頼のあとに Claude がしたこと(返答と操作の文、次の依頼まで)"""
    if not transcript or not os.path.exists(transcript):
        return None
    out, found = [], False
    with open(transcript, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("isSidechain"):
                continue
            msg = d.get("message") or {}
            if d.get("type") == "user" and isinstance(msg.get("content"), str):
                if found:
                    break
                found = msg["content"].strip() == prompt
            elif found and d.get("type") == "assistant":
                for b in msg.get("content") or []:
                    if b.get("type") == "text":
                        out.append(b["text"])
                    elif b.get("type") == "tool_use":
                        out.append(json.dumps(b.get("input") or {}, ensure_ascii=False))
    return "\n".join(out) if found else None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    args = ap.parse_args()
    with open(LOG, encoding="utf-8") as f:
        entries = [e for e in map(json.loads, f) if e["time"] >= args.since]
    brain = Brain(os.path.join(HERE, "brain_state.json"), embedder=None, store=None)
    concepts_of = {tr.get("ref"): tr["concepts"] for tr in brain.traces if tr.get("ref")}

    by_bin = defaultdict(lambda: [0, 0, 0])   # 確信度の区間 -> [差し込んだ数, 開いた数, 語が出た数]
    checked = opened_any = hinted_any = 0
    lengths, ms = [], []
    for e in entries:
        lengths.append(len(e["text"]))
        ms.append(e["ms"])
        after = after_prompt(e.get("transcript_path"), e["prompt"])
        if after is None:
            continue
        checked += 1
        prompt = e["prompt"].lower()
        opened = hinted = False
        for ep in e["episodes"]:
            b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= ep["conf"] < hi)
            by_bin[b][0] += 1
            if re.search(re.escape(ep["ref"][:8]), after):
                by_bin[b][1] += 1
                opened = True
            words = [c for c in concepts_of.get(ep["ref"], []) if len(c) >= 2 and c.lower() not in prompt]
            words = [os.path.basename(c.rstrip("\\/")) if is_place(c) else c for c in words]
            if any(w and w.lower() in after.lower() for w in words):
                by_bin[b][2] += 1
                hinted = True
        opened_any += opened
        hinted_any += hinted

    n = max(len(entries), 1)
    print(f"差し込んだ回数 {len(entries)}回 / セッション記録と突き合わせられたもの {checked}回")
    print(f"差し込みの長さ 平均{sum(lengths) / n:.0f}文字 / 何も差し込まなかった {sum(x == 0 for x in lengths)}回")
    print(f"所要時間(サーバー側) 平均{sum(ms) / n:.0f}ms / 最大{max(ms, default=0)}ms")
    print(f"記憶の鍵を開いた {opened_any}回 / 差し込んだ記憶の語がそのあと出てきた {hinted_any}回(突き合わせた{checked}回中)")
    print()
    print("| 確信度 | 差し込んだ記憶 | 鍵を開いた | 語が出てきた |")
    print("|---|---|---|---|")
    for i, (lo, hi) in enumerate(BINS):
        shown, opened, hinted = by_bin[i]
        rate = lambda k: f"{k / shown:.0%}" if shown else "-"
        print(f"| {lo:.1f}〜{min(hi, 1):.1f} | {shown} | {opened}({rate(opened)}) | {hinted}({rate(hinted)}) |")


if __name__ == "__main__":
    main()
