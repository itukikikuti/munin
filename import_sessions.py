"""Claude Code のセッション記録から記憶を作る(LLM は使わない)

  python import_sessions.py   ※ 今の記憶は消えて置き換わる

1回のユーザー指示と、それに続く Claude の作業を1つの出来事として海馬に入れる。
  場所       : 作業フォルダ(cwd)と、そのプロジェクトのルート
  話題の語   : ユーザーの発言と Claude の最後の返答から、形態素解析で取り出した名詞
  ファイル   : 編集・作成したファイルの名前
実際の日時どおりに時間を進め、日付が変わるたびに眠る。
"""
import glob
import json
import math
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from janome.tokenizer import Tokenizer

from brain import Brain

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))
PROJECTS = os.path.expanduser("~/.claude/projects")
HOME = os.path.expanduser("~")
JST = timezone(timedelta(hours=9))

USER_TERMS = 5          # 1回の指示から取り出す語の数
REPLY_TERMS = 3         # Claude の返答から取り出す語の数
MAX_FILES = 3
TEXT_LIMIT = 1500       # 貼り付けられたログなどで長すぎる文は先頭だけ見る
MIN_DF = 2              # 2回以上の出来事に出てくる語だけ使う(一度きりの語は雑音が多い)
MAX_DF_RATIO = 0.03     # 3%を超える出来事に出てくる語は、ありふれすぎていて手がかりにならない

STOPWORDS = set("""
感じ 場合 状態 方法 必要 確認 修正 対応 問題 部分 内容 今回 以下 以上 一つ 二つ 全部 全体 結果 理由
意味 説明 自分 説明 今日 明日 昨日 最初 最後 途中 前回 次回 他 別 後 前 中 上 下 時 時間 日 回 個 件
ファイル フォルダ コード エラー 実行 作成 変更 追加 削除 使用 利用 設定 表示 機能 処理 動作 値 名前
ユーザー 質問 回答 返答 指示 お願い 了解 承知 ありがとう 可能 不要 不明 OK ok
""".split())

TOKENIZER = Tokenizer()
SKIP_POS1 = {"非自立", "代名詞", "特殊", "接尾", "副詞可能", "数"}


def nouns(text):
    """複合名詞を取り出す(連続する名詞をつなげる)"""
    out, buf = [], []
    for tok in TOKENIZER.tokenize(text[:TEXT_LIMIT]):
        pos = tok.part_of_speech.split(",")
        if pos[0] == "名詞" and (buf or pos[1] not in SKIP_POS1):
            buf.append(tok.surface)
            continue
        if buf:
            out.append("".join(buf))
            buf = []
    if buf:
        out.append("".join(buf))
    return [w for w in (w.strip() for w in out) if usable(w)]


def usable(w):
    if len(w) < 2 or len(w) > 20 or w in STOPWORDS:
        return False
    if re.fullmatch(r"[\u3040-\u309f]+", w):     # ひらがなだけ(「こと」「やつ」など)
        return False
    if re.fullmatch(r"[\d\W_]+", w):              # 数字・記号だけ
        return False
    if re.search(r"[\\/<>{}()\[\]=;:\"'`|]", w):  # パスやコードの断片
        return False
    if re.fullmatch(r"[A-Za-z]{1,2}", w):
        return False
    return True


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    return ""


def place_of(cwd):
    """作業フォルダ → 記憶する場所(ホームや一時フォルダは、どのプロジェクトでもないので除く)"""
    if not cwd:
        return []
    low = cwd.lower()
    if low.rstrip("\\/") == HOME.lower() or "\\temp\\" in low or "\\appdata\\" in low:
        return []
    places = [cwd]
    m = re.match(r"(?i)^(c:\\dev\\[^\\]+)", cwd)
    if m and m.group(1).lower() != low:
        places.append(m.group(1))
    m = re.match(r"(?i)^(.*\\documents\\obsidian)(\\|$)", cwd)
    if m and m.group(1).lower() != low:
        places.append(m.group(1))
    return places


def read_turns():
    """全セッションを読み、ユーザー指示ごとの出来事にまとめる"""
    turns = []
    files = glob.glob(os.path.join(PROJECTS, "*", "*.jsonl"))
    for i, path in enumerate(files):
        print(f"\r  読み込み {i + 1}/{len(files)}", end="", file=sys.stderr, flush=True)
        cur = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
                    continue
                msg = d.get("message") or {}
                if d.get("type") == "user":
                    text = text_of(msg.get("content"))
                    if not text.strip() or text.lstrip().startswith("<"):
                        continue  # ツールの結果やコマンド出力は指示ではない
                    cur = {"t": d.get("timestamp"), "cwd": d.get("cwd"), "user": text,
                           "reply": "", "files": []}
                    turns.append(cur)
                elif d.get("type") == "assistant" and cur is not None:
                    for b in msg.get("content") or []:
                        if b.get("type") == "text" and b.get("text", "").strip():
                            cur["reply"] = b["text"]  # 最後の返答(まとめ)を残す
                        elif b.get("type") == "tool_use" and b.get("name") in ("Edit", "Write", "NotebookEdit"):
                            fp = (b.get("input") or {}).get("file_path") or (b.get("input") or {}).get("notebook_path")
                            if fp:
                                name = os.path.basename(fp)
                                if name not in cur["files"]:
                                    cur["files"].append(name)
    print(file=sys.stderr)
    return [t for t in turns if t["t"]]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    turns = read_turns()
    turns.sort(key=lambda t: t["t"])
    print(f"出来事 {len(turns)}件")

    # 形態素解析して、語ごとに「いくつの出来事に出てくるか」を数える
    for i, t in enumerate(turns):
        if i % 100 == 0:
            print(f"\r  形態素解析 {i}/{len(turns)}", end="", file=sys.stderr, flush=True)
        t["user_terms"] = nouns(t["user"])
        t["reply_terms"] = nouns(t["reply"])
    print(file=sys.stderr)
    df = Counter()
    for t in turns:
        df.update(set(t["user_terms"]) | set(t["reply_terms"]) | set(t["files"]))
    max_df = max(MIN_DF + 1, len(turns) * MAX_DF_RATIO)

    def pick(terms, k):
        tf = Counter(terms)
        scored = [(c * math.log(len(turns) / df[w]), w) for w, c in tf.items()
                  if MIN_DF <= df[w] <= max_df]
        return [w for _, w in sorted(scored, reverse=True)[:k]]

    # 実際の日時どおりに出来事を体験させ、日付が変わるたびに眠る
    if os.path.exists(STATE):
        os.remove(STATE)
    brain = Brain(STATE, embedder=None)
    start = datetime.fromisoformat(turns[0]["t"].replace("Z", "+00:00")).astimezone(JST)
    origin = start.replace(hour=0, minute=0, second=0, microsecond=0)
    day = origin.date()
    learned = nights = 0
    for t in turns:
        at = datetime.fromisoformat(t["t"].replace("Z", "+00:00")).astimezone(JST)
        while at.date() > day:  # 日付が変わった: その夜に眠る
            day += timedelta(days=1)
            brain.now = max(brain.now, (datetime.combine(day, datetime.min.time(), JST) - origin)
                            .total_seconds() / 3600)
            brain.sleep()
            nights += 1
        brain.now = max(brain.now, (at - origin).total_seconds() / 3600)
        concepts = (place_of(t["cwd"]) + pick(t["user_terms"], USER_TERMS)
                    + pick(t["reply_terms"], REPLY_TERMS)
                    + [f for f in t["files"] if MIN_DF <= df[f] <= max_df][:MAX_FILES])
        concepts = list(dict.fromkeys(concepts))
        if len(concepts) >= 2:
            brain.learn(concepts)
            learned += 1
    brain.context = {}
    brain.save()

    places = {c for k in brain.cortex for c in k.split("\t") if c[:2].lower() == "c:" or c.startswith("/")}
    print(f"海馬に入れた出来事 {learned}件 / 眠った回数 {nights}晩")
    print(f"皮質のリンク {len(brain.cortex)}本 / 概念 {len(brain.known_concepts())}個 / "
          f"場所 {len(places)}か所 / 海馬に残っている記録 {len(brain.hippocampus)}件")

    # 思い出すときに使う埋め込みを、先にまとめて計算しておく
    brain_e = Brain(STATE)
    if brain_e.embedder and brain_e.embedder.ensure(sorted(brain_e.known_concepts()), progress=True):
        print("埋め込みを計算しました")
    else:
        print("埋め込みモデルに接続できませんでした(思い出すときに計算します)")
    print(f"所要時間 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
