"""Claude Code のセッション記録から記憶を作る(LLM は使わない)

  python import_sessions.py            ※ 全件。今の記憶は消えて置き換わる
  python import_sessions.py 300        ※ 最近の300件だけ

1回のユーザー指示と、それに続く Claude の作業を1つの出来事として海馬に入れる。
  場所       : 作業フォルダ(cwd)と、触ったファイルのあるプロジェクトのルート
  話題の語   : ユーザーの発言と Claude の返答(全文)から、形態素解析で取り出した名詞
  ファイル   : 編集・作成・読み込みしたファイルの名前(編集したものを優先)
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

from brain import SLEEP_HOURS, Brain

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))
PROJECTS = os.path.expanduser("~/.claude/projects")
HOME = os.path.expanduser("~")
JST = timezone(timedelta(hours=9))

# 1つの出来事として海馬が束ねる量には限りがあるので、出来事ごとに要素の数を絞る
USER_TERMS = 6          # 1回の指示から取り出す語の数
REPLY_TERMS = 6         # Claude の返答から取り出す語の数
MAX_FILES = 5
MAX_PLACES = 4
MIN_DF = 2              # 2回以上の出来事に出てくる語だけ使う(一度きりの語は雑音が多い)
GENERIC_SESSION_RATIO = 0.3   # 3割を超えるセッションに出てくる語は、ありふれていて手がかりにならない
GENERIC_MIN_SESSIONS = 3
DAY_STARTS_AT = 4       # 1日の区切り(時)

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
    text = re.sub(r"[*#|`>~_\[\]]+", " ", text)  # Markdown の記号(太字、見出し、表など)
    for tok in TOKENIZER.tokenize(text):
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


def ignored_place(path):
    """どのプロジェクトでもない場所(ホーム、一時フォルダ、設定フォルダ)"""
    low = path.lower().rstrip("\\/")
    return (low == HOME.lower() or "\\temp" in low or "\\appdata\\" in low
            or "\\.claude" in low or low.startswith("/tmp"))


def project_root(path):
    """パス → プロジェクトのルート(C:\\dev\\〜 や Obsidian の保管庫)"""
    m = re.match(r"(?i)^(c:\\dev\\[^\\]+)", path)
    if m:
        return m.group(1)
    m = re.match(r"(?i)^(.*\\documents\\obsidian)(\\|$)", path)
    if m:
        return m.group(1)
    return None


def places_of(cwd, edited, read):
    """作業フォルダと触ったファイルから、記憶する場所を決める

    編集したファイルは、そのファイルのあるフォルダまで覚える(どこで作業したかの具体的な場所)。
    読んだだけのファイルは数が多いので、プロジェクトのルートだけ覚える。
    """
    places = []
    if cwd and not ignored_place(cwd):
        places.append(cwd)
        root = project_root(cwd)
        if root:
            places.append(root)
    for fp in edited:
        root = project_root(fp)
        if root and not ignored_place(fp):
            folder = os.path.dirname(fp)
            places += [folder, root] if folder.lower() != root.lower() else [root]
    for fp in read:
        root = project_root(fp)
        if root and not ignored_place(fp):
            places.append(root)
    return places


def read_turns(limit=None):
    """セッションを新しい順に読み、ユーザー指示ごとの出来事にまとめる"""
    turns = []
    seen = set()  # セッションを再開・分岐すると履歴がコピーされるので、同じ発言は1回だけ数える
    files = sorted(glob.glob(os.path.join(PROJECTS, "*", "*.jsonl")), key=os.path.getmtime, reverse=True)
    for i, path in enumerate(files):
        if limit and len(turns) >= limit * 2:
            break  # 必要な件数が十分に集まった(並べ替えたあと新しい順に limit 件使う)
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
                    key = d.get("uuid") or (d.get("timestamp"), text)
                    if key in seen:
                        cur = None  # 続く Claude の作業も、すでに数えた分
                        continue
                    seen.add(key)
                    cur = {"t": d.get("timestamp"), "cwd": d.get("cwd"), "user": text, "session": path,
                           "reply": [], "edited": [], "read": []}
                    turns.append(cur)
                elif d.get("type") == "assistant" and cur is not None:
                    for b in msg.get("content") or []:
                        if b.get("type") == "text" and b.get("text", "").strip():
                            cur["reply"].append(b["text"])
                        elif b.get("type") == "tool_use":
                            inp = b.get("input") or {}
                            fp = inp.get("file_path") or inp.get("notebook_path")
                            if not fp:
                                continue
                            kind = "edited" if b.get("name") in ("Edit", "Write", "NotebookEdit") else \
                                   "read" if b.get("name") == "Read" else None
                            if kind and fp not in cur[kind]:
                                cur[kind].append(fp)
    print(file=sys.stderr)
    return [t for t in turns if t["t"]]


def unify_case(turns):
    """大文字小文字だけが違うパス(C:\\dev\\ET と C:\\dev\\et)を、よく使われる表記にそろえる"""
    seen = Counter()
    for t in turns:
        for p in [t["cwd"] or ""] + t["edited"] + t["read"]:
            seen[p] += 1
    best = {}
    for p, c in seen.items():
        k = p.lower()
        if k not in best or c > seen[best[k]]:
            best[k] = p
    fix = lambda p: best.get(p.lower(), p) if p else p
    for t in turns:
        t["cwd"] = fix(t["cwd"])
        t["edited"] = [fix(p) for p in t["edited"]]
        t["read"] = [fix(p) for p in t["read"]]
    # 場所(プロジェクトのルート)も同じようにそろえる
    roots = Counter()
    for t in turns:
        t["places"] = places_of(t["cwd"], t["edited"], t["read"])
        roots.update(t["places"])
    best = {}
    for p, c in roots.items():
        if p.lower() not in best or c > roots[best[p.lower()]]:
            best[p.lower()] = p
    for t in turns:
        t["places"] = list(dict.fromkeys(best[p.lower()] for p in t["places"]))[:MAX_PLACES]
        names = [os.path.basename(p) for p in t["edited"] + t["read"]]
        t["files"] = list(dict.fromkeys(n for n in names if n))


class Vocabulary:
    """どの語を概念として覚えるか(その時点までに見たやり取りだけから判断する。未来の頻度は知らない)"""

    def __init__(self):
        self.turns = 0
        self.df = Counter()           # 語ごとに、いくつの出来事に出てきたか
        self.sdf = Counter()          # 語ごとに、いくつのセッションに出てきたか
        self.session_terms = {}

    def observe(self, t):
        self.turns += 1
        terms = set(t["user_terms"]) | set(t["reply_terms"]) | set(t["files"])
        self.df.update(terms)
        seen = self.session_terms.setdefault(t["session"], set())
        self.sdf.update(terms - seen)
        seen |= terms

    def usable(self, w):
        # 一度きりの語は雑音が多い。多くのセッションにまたがる語は、話題を表さないありふれた語
        max_sdf = max(GENERIC_MIN_SESSIONS, len(self.session_terms) * GENERIC_SESSION_RATIO)
        return self.df[w] >= MIN_DF and self.sdf[w] <= max_sdf

    def pick(self, terms, k):
        """TF-IDF の高い語から k 個"""
        tf = Counter(terms)
        scored = [(c * math.log(self.turns / self.df[w]), w) for w, c in tf.items() if self.usable(w)]
        return [w for _, w in sorted(scored, reverse=True)[:k]]


def analyze(t):
    """やり取りから名詞を取り出しておく"""
    if "user_terms" not in t:
        t["user_terms"] = nouns(t["user"])
        t["reply_terms"] = nouns("\n".join(t["reply"]))


class Life:
    """過去のやり取りを、実際の日時どおりに体験させる(日付が変わるたびに眠る)"""

    def __init__(self, brain, first_turn):
        self.brain = brain
        self.vocab = Vocabulary()
        start = parse_time(first_turn)
        self.origin = start.replace(hour=0, minute=0, second=0, microsecond=0)
        self.day = self.day_of(start)
        self.nights = 0
        self.learned = 0

    def hours(self, dt):
        return (dt - self.origin).total_seconds() / 3600

    @staticmethod
    def day_of(dt):
        return (dt - timedelta(hours=DAY_STARTS_AT)).date()  # 1日の区切りは午前4時(深夜の作業は前の日の続き)

    def advance(self, t):
        """そのやり取りの時刻まで時間を進める"""
        at = parse_time(t)
        b = self.brain
        while self.day_of(at) > self.day:
            # 日をまたいだ: その日の最後の作業のあとに眠る。
            # 次の作業まで十分空いていれば、起きた時刻が次の作業の直前になるように眠り始める
            self.day += timedelta(days=1)
            b.now = max(b.now, min(self.hours(at) - SLEEP_HOURS,
                                   self.hours(datetime.combine(self.day, datetime.min.time(), JST)) + DAY_STARTS_AT))
            # 次の作業までの間しか眠れない(徹夜に近い夜は短い睡眠になる)
            b.sleep(max(0.5, min(SLEEP_HOURS, self.hours(at) - b.now)))
            self.nights += 1
        b.now = max(b.now, self.hours(at))

    def episode(self, t):
        """やり取りを1つの出来事(一緒に覚える概念の組)にする"""
        analyze(t)
        self.vocab.observe(t)
        v = self.vocab
        concepts = (t["places"] + v.pick(t["user_terms"], USER_TERMS) + v.pick(t["reply_terms"], REPLY_TERMS)
                    + [f for f in t["files"] if v.usable(f)][:MAX_FILES])
        return list(dict.fromkeys(concepts))

    def remember(self, t):
        """やり取りを体験したあと、海馬に記録する"""
        concepts = self.episode(t)
        if len(concepts) >= 2:
            self.brain.learn(concepts)
            self.learned += 1

    def experience(self, t):
        self.advance(t)
        self.remember(t)


def parse_time(t):
    return datetime.fromisoformat(t["t"].replace("Z", "+00:00")).astimezone(JST)


def load_turns(limit=None):
    """セッションを読み、時刻順に並べる(limit があれば最近の limit 件)"""
    turns = read_turns(limit)
    turns.sort(key=lambda t: t["t"])
    if limit:
        turns = turns[-limit:]
    unify_case(turns)
    return turns


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    turns = load_turns(limit)
    print(f"出来事 {len(turns)}件 ({turns[0]['t'][:10]} 〜 {turns[-1]['t'][:10]})")

    if os.path.exists(STATE):
        os.remove(STATE)
    brain = Brain(STATE, embedder=None)
    life = Life(brain, turns[0])
    for i, t in enumerate(turns):
        if i % 50 == 0:
            print(f"\r  体験 {i}/{len(turns)}", end="", file=sys.stderr, flush=True)
        life.experience(t)
    print(file=sys.stderr)
    brain.context = {}
    brain.save()

    places = {c for k in brain.cortex for c in k.split("\t") if c[:2].lower() == "c:" or c.startswith("/")}
    print(f"海馬に入れた出来事 {life.learned}件 / 眠った回数 {life.nights}晩")
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
