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
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone

from janome.tokenizer import Tokenizer

import paths
from brain import SLEEP_HOURS, Brain
from intentions import extract, mentions_done
from store import RESULT_CHARS, Store, build_record, tool_summary

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = paths.STATE
LIFE = os.path.join(os.path.dirname(STATE), "life_state.json")   # 続きから覚えるための、体験の途中の状態
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
PROMPT_EXCERPT = 120    # 思い出したときに伝える、依頼の抜粋の長さ
REPLY_EXCERPT = 200     # 思い出したときに伝える、回答の抜粋の長さ(最後の返答 = まとめ)

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


def clean_prompt(text):
    """ユーザーの発言から、システムが差し込んだ部分を除く(何も残らなければ空)

    完了通知・コマンドの出力・システムからの注意書きは、ユーザーの言葉ではないので捨てる。
    貼り付けた文章(<pasted_content>)などのタグは外して中身を残す。
    コマンド(/goal など)は、引数があるときだけ「/名前 引数」として残す(引数のない /compact などは指示の中身がない)。
    """
    # アプリが依頼の前に付ける注意書き(フォルダを選ばずに始めた会話など)は外し、残りを依頼とする
    s = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S).strip()
    if s.startswith(SYSTEM_TAGS):
        return ""
    m = re.search(r"<command-name>\s*(.*?)\s*</command-name>", s, re.S)
    if m:
        args = re.search(r"<command-args>(.*?)</command-args>", s, re.S)
        args = args.group(1).strip() if args else ""
        return f"{m.group(1)} {args}" if args else ""
    s = WRAPPER_TAGS.sub("", s)   # 包んでいるタグだけ外す(依頼に書いた HTML などはそのまま)
    return s.strip()


WRAPPER_TAGS = re.compile(r"</?(?:pasted_content|artifact-[\w-]+)(?:\s[^<>]*)?>")


SYSTEM_TAGS = ("<task-notification", "<local-command-stdout", "<local-command-stderr", "<local-command-caveat",
               "<system-reminder", "<bash-stdout", "<bash-stderr", "<command-message",
               "<agent-message", "<cross-session-message")   # サブエージェントや別の会話からの連絡
INTERRUPTED = "[Request interrupted"
USER_SAID = re.compile(r"the user said:\s*(.+)", re.S)                # ツールの実行を断ったときの、ユーザーの言葉
ANSWERS = re.compile(r'"((?:[^"\\]|\\.)*)"="((?:[^"\\]|\\.)*)"')      # 質問への答え("質問"="答え")


def user_words_in_result(result):
    """ツールの結果の中にある、ユーザー自身の言葉(質問への答え、実行を断ったときの指示)"""
    if "answered" in result[:200]:
        return "\n".join(f"「{q}」→ {a}" for q, a in ANSWERS.findall(result))
    m = USER_SAID.search(result)
    return m.group(1).strip() if m else ""


def new_turn(t, cwd, text, session, uid, source="claude"):
    return {"t": t, "cwd": cwd, "user": text, "session": session, "uuid": uid, "source": source,
            "reply": [], "said": [], "edited": [], "read": [], "steps": [], "pending": {}}


def read_turns(limit=None, files=None, codex_files=None):
    """セッションを新しい順に読み、ユーザー指示ごとの出来事にまとめる

    ユーザーの言葉は、普通の発言のほかに、次の形でも残っている。どれも本人の判断や好みなので拾う。
      作業中に送った発言 : 発言ではなく、添付(queued_command)として記録される → 新しい指示として扱う
      質問への答え       : ツール(AskUserQuestion)の結果として記録される → その出来事の中のユーザーの言葉
      実行を断ったときの指示 : ツールの結果(「the user said: …」)
      途中で止めたこと   : 「[Request interrupted …」 → その方向が違ったという合図として残す
    """
    turns = []
    seen = set()  # セッションを再開・分岐すると履歴がコピーされるので、同じ発言は1回だけ数える
    if files is None:
        files = sorted(claude_files(), key=os.path.getmtime, reverse=True)
    for i, path in enumerate(files):
        if limit and len(turns) >= limit * 2:
            break  # 必要な件数が十分に集まった(並べ替えたあと新しい順に limit 件使う)
        print(f"\r  読み込み {i + 1}/{len(files)}", end="", file=sys.stderr, flush=True)
        cur = None
        queued = set()   # この会話で、作業中に送られた発言(あとで普通の発言としても記録されることがある)
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("isSidechain") or d.get("isMeta") or d.get("isCompactSummary"):
                    continue
                msg = d.get("message") or {}
                if d.get("type") == "attachment":
                    a = d.get("attachment") or {}
                    if a.get("type") != "queued_command" or a.get("commandMode") != "prompt":
                        continue
                    text = clean_prompt(text_of(a.get("prompt")))
                    key = d.get("uuid") or (a.get("timestamp"), text)
                    if not text or key in seen or text in queued:
                        continue
                    seen.add(key)
                    queued.add(text)
                    cur = new_turn(a.get("timestamp") or d.get("timestamp"), d.get("cwd"), text, path, d.get("uuid"))
                    turns.append(cur)
                    continue
                if d.get("type") == "user" and cur is not None and isinstance(msg.get("content"), list):
                    for c in msg["content"]:  # ツールの結果を、そのツールの実行に結びつける
                        if c.get("type") == "tool_result" and c.get("tool_use_id") in cur["pending"]:
                            step = cur["pending"].pop(c["tool_use_id"])
                            result = text_of(c.get("content"))
                            step["result"] = result[:RESULT_CHARS]
                            step["error"] = bool(c.get("is_error"))
                            said = user_words_in_result(result)
                            if said:
                                step["said"] = said
                                cur["said"].append(said)
                if d.get("type") == "user":
                    raw = text_of(msg.get("content"))
                    if raw.lstrip().startswith(INTERRUPTED):
                        if cur is not None:   # ユーザーが途中で止めた
                            cur["steps"].append({"kind": "interrupt"})
                        continue
                    text = clean_prompt(raw)
                    if not text:
                        continue  # ツールの結果・コマンド出力・システムの通知は指示ではない
                    key = d.get("uuid") or (d.get("timestamp"), text)
                    if key in seen:
                        cur = None  # 続く Claude の作業も、すでに数えた分
                        continue
                    seen.add(key)
                    if text in queued:
                        continue  # 作業中に送った発言が、あとで普通の発言としても記録されたもの(すでに数えた)
                    cur = new_turn(d.get("timestamp"), d.get("cwd"), text, path, d.get("uuid"))
                    turns.append(cur)
                elif d.get("type") == "assistant" and cur is not None:
                    for b in msg.get("content") or []:
                        if b.get("type") == "text" and b.get("text", "").strip():
                            cur["reply"].append(b["text"])
                            cur["steps"].append({"kind": "text", "text": b["text"]})
                        elif b.get("type") == "thinking" and b.get("thinking", "").strip():
                            # 考えた過程(なぜそうしたか)。記録には残すが、量が多いので検索の対象にはしない
                            cur["steps"].append({"kind": "thinking", "text": b["thinking"][:THINKING_CHARS]})
                        elif b.get("type") == "tool_use":
                            inp = b.get("input") or {}
                            step = {"kind": "tool", "name": b.get("name"), "summary": tool_summary(b.get("name"), inp)}
                            cur["steps"].append(step)
                            cur["pending"][b.get("id")] = step
                            fp = inp.get("file_path") or inp.get("notebook_path")
                            if not fp:
                                continue
                            kind = "edited" if b.get("name") in ("Edit", "Write", "NotebookEdit") else \
                                   "read" if b.get("name") == "Read" else None
                            if kind and fp not in cur[kind]:
                                cur[kind].append(fp)
    print(file=sys.stderr)
    turns += read_codex_turns(codex_files)
    for t in turns:
        t.pop("pending", None)
    return [t for t in turns if t["t"]]


THINKING_CHARS = 4000


def read_codex_turns(files=None):
    """Codex のセッション記録(~/.codex/sessions)を、Claude Code と同じ形の出来事にする

    完了した項目(item_completed)を正にする。承認の審査役やレビュー役のような、
    Codex が自分で立ち上げた会話(サブエージェント)は、ユーザーとのやり取りではないので除く。
    """
    turns = []
    files = sorted(codex_files()) if files is None else files
    for path in files:
        cur, sid, cwd, skip = None, os.path.basename(path), None, False
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                p = d.get("payload") or {}
                if d.get("type") == "session_meta":
                    sid, cwd = p.get("id") or sid, p.get("cwd")
                    skip = isinstance(p.get("source"), dict) or p.get("thread_source") not in (None, "user")
                    if skip:
                        break
                    continue
                if d.get("type") == "turn_context":
                    cwd = p.get("cwd") or cwd
                    continue
                if d.get("type") != "event_msg" or p.get("type") != "item_completed":
                    continue
                it = p.get("item") or {}
                kind = it.get("type")
                if kind == "UserMessage":
                    text = clean_prompt("\n".join(c.get("text", "") for c in it.get("content") or []))
                    if not text or text.startswith(CODEX_AGENT_HISTORY):
                        if text.startswith(CODEX_AGENT_HISTORY):
                            break   # 古い版の審査役の会話
                        continue
                    # 古い版は項目の id が会話の中でしか一意でないので、会話の id と合わせて鍵にする
                    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"codex:{sid}:{it.get('id')}"))
                    cur = new_turn(d.get("timestamp"), cwd, text, path, uid, source="codex")
                    turns.append(cur)
                elif cur is None:
                    continue
                elif kind == "AgentMessage":
                    text = "\n".join(c.get("text", "") for c in it.get("content") or []).strip()
                    if text:
                        cur["reply"].append(text)
                        cur["steps"].append({"kind": "text", "text": text})
                elif kind == "Plan" and it.get("text"):
                    cur["steps"].append({"kind": "text", "text": it["text"]})
                elif kind == "Reasoning":
                    text = "\n".join(it.get("summary_text") or [])
                    if text.strip():
                        cur["steps"].append({"kind": "thinking", "text": text[:THINKING_CHARS]})
                elif kind == "CommandExecution":
                    cmd = it.get("command")
                    cmd = " ".join(cmd) if isinstance(cmd, list) else str(cmd or "")
                    cur["steps"].append({"kind": "tool", "name": "Bash", "summary": tool_summary("Bash", {"command": cmd}),
                                         "result": str(it.get("aggregated_output") or "")[:RESULT_CHARS],
                                         "error": it.get("exit_code") not in (0, None)})
                    for pc in it.get("parsed_cmd") or []:
                        if pc.get("type") == "read" and pc.get("path") and pc["path"] not in cur["read"]:
                            cur["read"].append(pc["path"])
                elif kind == "FileChange":
                    for fp in (it.get("changes") or {}):
                        cur["steps"].append({"kind": "tool", "name": "Edit", "summary": tool_summary("Edit", {"file_path": fp})})
                        if fp not in cur["edited"]:
                            cur["edited"].append(fp)
                elif kind == "WebSearch":
                    cur["steps"].append({"kind": "tool", "name": "WebSearch",
                                         "summary": tool_summary("WebSearch", {"query": it.get("query") or ""})})
                elif kind == "McpToolCall":
                    cur["steps"].append({"kind": "tool", "name": f"{it.get('server')}.{it.get('tool')}",
                                         "summary": tool_summary(f"{it.get('server')}.{it.get('tool')}", it.get("arguments") or {}),
                                         "result": json.dumps(it.get("result"), ensure_ascii=False)[:RESULT_CHARS]})
    return turns


CODEX_SESSIONS = os.path.expanduser("~/.codex/sessions")


def claude_files():
    return glob.glob(os.path.join(PROJECTS, "*", "*.jsonl"))


def codex_files():
    return glob.glob(os.path.join(CODEX_SESSIONS, "**", "*.jsonl"), recursive=True)
CODEX_AGENT_HISTORY = "The following is the Codex agent history"


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
        t["user_terms"] = nouns("\n".join([t["user"]] + t.get("said", [])))   # 質問への答えなども本人の言葉
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
        self.session = None
        self.scanned = {}   # セッション記録のファイル -> 最後に読み切ったときの更新時刻

    def save(self, path):
        """続きから体験できるように、語の数え方・暦・最後に読んだファイルの状態を保存する"""
        v = self.vocab
        data = {"turns": v.turns, "df": v.df, "sdf": v.sdf,
                "session_terms": {s: sorted(ts) for s, ts in v.session_terms.items()},
                "origin": self.origin.isoformat(), "day": self.day.isoformat(), "nights": self.nights,
                "learned": self.learned, "session": self.session, "scanned": self.scanned}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def restore(cls, brain, path):
        """保存した状態から続きを体験する(なければ None)"""
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        life = cls.__new__(cls)
        life.brain = brain
        life.vocab = Vocabulary()
        life.vocab.turns = data["turns"]
        life.vocab.df = Counter(data["df"])
        life.vocab.sdf = Counter(data["sdf"])
        life.vocab.session_terms = {s: set(ts) for s, ts in data["session_terms"].items()}
        life.origin = datetime.fromisoformat(data["origin"])
        life.day = datetime.fromisoformat(data["day"]).date()
        life.nights, life.learned, life.session = data["nights"], data["learned"], data["session"]
        life.scanned = data.get("scanned", {})
        return life

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
        # 新しい会話(セッション)が始まったら、前の会話で考えていたことは持ち越さない
        if t["session"] != self.session:
            b.context = {}
            self.session = t["session"]

    def episode(self, t):
        """やり取りを1つの出来事(一緒に覚える概念の組)にする"""
        analyze(t)
        self.vocab.observe(t)
        v = self.vocab
        concepts = (t["places"] + v.pick(t["user_terms"], USER_TERMS) + v.pick(t["reply_terms"], REPLY_TERMS)
                    + [f for f in t["files"] if v.usable(f)][:MAX_FILES])
        return list(dict.fromkeys(concepts))

    def remember(self, t, concepts=None):
        """やり取りを体験したあと、海馬に記録する(concepts: 先に作ってあれば、その出来事の概念)"""
        concepts = self.episode(t) if concepts is None else concepts
        if len(concepts) >= 2:
            note = {"prompt": excerpt(t["user"], PROMPT_EXCERPT), "reply": excerpt(t["reply"][-1] if t["reply"] else "", REPLY_EXCERPT),
                    "time": parse_time(t).isoformat(), "session": os.path.basename(t["session"]), "ref": record_key(t)}
            # このやり取りで終わったことがあれば、前に覚えた「やること」を完了にしてから、新しい「やること」を覚える
            self.brain.settle(concepts, mentions_done(t["user"] + "\n" + "\n".join(t["reply"][-2:])))
            self.brain.learn(concepts, gist=t["user"], note=note)  # 要旨は依頼文(その出来事が何についてだったか)
            self.brain.intend(extract(t["user"], t["reply"], parse_time(t)), concepts, record_key(t), note["time"])
            self.learned += 1

    def experience(self, t):
        self.advance(t)
        self.remember(t)


def excerpt(text, limit):
    """思い出したときに伝える抜粋(Markdown の記号や改行を除いて、先頭から limit 文字)"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[*#|`>~_]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def record_key(t):
    """やり取りの記録の鍵(発言のID。なければ会話と時刻)"""
    return t.get("uuid") or f"{os.path.basename(t['session'])}:{t['t']}"


def store_records(store, turns):
    """やり取りの記録(全文)を保存し、細かい単位の検索に使うベクトルを計算しておく"""
    for t in turns:
        store.put(record_key(t), build_record(t))
    store.commit()
    return store.ensure_vectors(progress=True)


def learn_new(life, files=(), codex=(), settle=False):
    """セッション記録の新しいやり取りだけを、続きとして体験させる(使いながら覚える)

    files / codex: 読み直す Claude Code / Codex のセッション記録のファイル。
    すでに記録してあるやり取りは覚え直さない(記録の中身は、続きが書き足されていれば新しくする)。
    settle: 最近書き換わったファイルの最後のやり取りは、まだ途中かもしれないので覚えない
            (返答が終わったときのフックから呼ぶときは、終わっていると分かっているので False)。
    戻り値: 覚えたやり取りの数
    """
    brain, store = life.brain, life.brain.store
    turns = read_turns(files=list(files), codex_files=list(codex))
    if not turns:
        return 0
    turns.sort(key=lambda t: t["t"])
    if settle:
        recent = {p for p in list(files) + list(codex) if time.time() - os.path.getmtime(p) < SETTLE_SECONDS}
        last = {}
        for t in turns:
            if t["session"] in recent:
                last[t["session"]] = t
        turns = [t for t in turns if last.get(t["session"]) is not t]
    known = set(store.keys())
    new = [t for t in turns if record_key(t) not in known]
    unify_case(turns)
    if brain.embedder:
        brain.embedder.ensure_gists([t["user"] for t in new])
    for t in turns:   # 記録は、続きが書き足されていれば新しくする
        store.put(record_key(t), build_record(t))
    store.commit()
    store.ensure_vectors()
    for t in new:
        life.experience(t)
    if brain.embedder and new:   # 新しく覚えた概念の埋め込み(計算済みのものは飛ばされる)
        brain.embedder.ensure(sorted(brain.known_concepts()))
    return len(new)


SETTLE_SECONDS = 30 * 60   # これより最近に書き換わったセッション記録の最後のやり取りは、まだ途中とみなす


def parse_time(t):
    return datetime.fromisoformat(t["t"].replace("Z", "+00:00")).astimezone(JST)


def turn_from_record(key, rec):
    """保存してある記録から、出来事を作り直す(元のセッション記録が消えたとき用)"""
    steps = rec["steps"]
    files = lambda prefix: list(dict.fromkeys(s["summary"][len(prefix):] for s in steps
                                              if s["kind"] == "tool" and s["summary"].startswith(prefix)))
    src = rec.get("src") or {}
    return {"t": rec["time"], "cwd": rec.get("cwd"), "user": rec["prompt"], "session": src.get("file") or key,
            "uuid": key, "source": src.get("app", "claude"),
            "reply": [s["text"] for s in steps if s["kind"] == "text"],
            "said": [s["said"] for s in steps if s.get("said")],
            "edited": files("編集: ") + files("作成: "), "read": files("読み込み: "), "steps": steps}


def load_turns(limit=None, store=None):
    """セッションを読み、時刻順に並べる(limit があれば最近の limit 件)

    store があれば、元のセッション記録がもう残っていないやり取りを、保存してある記録から補う
    (Claude Code は古いセッション記録を自動で消すので、記憶の原本は store に持つ)。
    """
    turns = read_turns(limit)
    if store is not None and not limit:
        have = {record_key(t) for t in turns}
        lost = []
        for k in store.keys():
            if k in have:
                continue
            rec = store.record(k)
            if os.path.exists((rec.get("src") or {}).get("file") or ""):
                store.delete(k)   # 元の記録は残っているのに出来事にならなかった(取り込み方を変えて除いたもの)
            else:
                lost.append(turn_from_record(k, rec))
        store.commit()
        if lost:
            print(f"  元の記録が消えていたやり取り {len(lost)}件を、保存してある記録から補いました", file=sys.stderr)
        turns += lost
    turns.sort(key=lambda t: t["t"])
    if limit:
        turns = turns[-limit:]
    unify_case(turns)
    return turns


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t0 = time.time()
    if os.path.exists(STATE):
        os.remove(STATE)
    brain = Brain(STATE)
    # 読む前の更新時刻を覚えておく(読んでいる間に書き足された分は、あとで続きとして覚える)
    scanned = {p: os.path.getmtime(p) for p in claude_files() + codex_files()}
    turns = load_turns(limit, store=brain.store)
    print(f"出来事 {len(turns)}件 ({turns[0]['t'][:10]} 〜 {turns[-1]['t'][:10]})")
    if brain.embedder:  # 出来事の要旨のベクトルを、先にまとめて計算しておく
        brain.embedder.ensure_gists([t["user"] for t in turns], progress=True)
    if brain.store:     # やり取りの記録(全文)と、細かい単位の検索用のベクトル
        store_records(brain.store, turns)
    life = Life(brain, turns[0])
    for i, t in enumerate(turns):
        if i % 50 == 0:
            print(f"\r  体験 {i}/{len(turns)}", end="", file=sys.stderr, flush=True)
        life.experience(t)
    print(file=sys.stderr)
    brain.context = {}
    brain.save()
    life.scanned = scanned
    life.save(LIFE)

    places = {c for k in brain.cortex for c in k.split("\t") if c[:2].lower() == "c:" or c.startswith("/")}
    print(f"海馬に入れた出来事 {life.learned}件 / 眠った回数 {life.nights}晩")
    print(f"皮質のリンク {len(brain.cortex)}本 / 概念 {len(brain.known_concepts())}個 / "
          f"場所 {len(places)}か所 / 海馬に残っている記録 {len(brain.hippocampus)}件")

    # 思い出すときに使う概念の埋め込みを、先にまとめて計算しておく
    if brain.embedder and brain.embedder.ensure(sorted(brain.known_concepts()), progress=True):
        print("埋め込みを計算しました")
    else:
        print("埋め込みモデルに接続できませんでした(思い出すときに計算します)")
    print(f"所要時間 {time.time() - t0:.0f}秒")


if __name__ == "__main__":
    main()
