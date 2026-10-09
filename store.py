"""やり取りの記録(全文)を保存し、細かい単位で意味検索できるようにする

記憶モデルの痕跡(エピソード記憶)は、この記録への索引(鍵)だけを持つ。
海馬は記憶の中身そのものではなく、中身のありかを指す索引を持つ、という説(海馬インデックス理論)と同じ形。

  記録     : 依頼の全文、Claude の返答の全文、ツールの実行(コマンド・ファイル)とその結果
             (結果は先頭 RESULT_CHARS 文字まで。全文は元のセッション記録にあり、src で辿れる)
  チャンク : 記録を、依頼・返答の段落・コマンド・エラーなどの小さな単位に分けたもの。
             それぞれに意味のベクトルを持たせ、依頼と関係する部分を探せるようにする
"""
import json
import math
import os
import re
import sqlite3
import sys

import numpy as np

from embed import Embedder, doc_key

CHUNK_MODEL = "qwen3-embedding:0.6b"   # チャンクは数が多いので、軽くて速いモデルを使う
CHUNK_DIM = 1024
CHUNK_INSTRUCT = ("Instruct: Given a user's request about their work, "
                  "retrieve passages from past work sessions that are relevant to it\nQuery: ")
RESULT_CHARS = 2000      # ツールの結果は先頭だけ残す(長いログは元のセッション記録を参照する)
PARA_MIN = 80            # 段落が短すぎるときは、次の段落とつなげる
PARA_MAX = 600           # 段落が長すぎるときは分ける
PROMPT_PARTS = 3         # 依頼が長い(ログの貼り付けなど)ときに、チャンクにする先頭の部分の数


def split_text(text, limit=PARA_MAX, min_len=PARA_MIN):
    """文章を段落ごとのかたまりに分ける"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out, buf = [], ""
    for p in paras:
        buf = f"{buf}\n{p}" if buf else p
        if len(buf) >= min_len:
            out += [buf[i:i + limit] for i in range(0, len(buf), limit)]
            buf = ""
    if buf:
        out.append(buf[:limit])
    return out


def tool_summary(name, inp):
    """ツールの実行を一言で表す"""
    if name in ("Bash", "PowerShell"):
        desc = inp.get("description", "")
        return f"コマンド: {inp.get('command', '')[:400]}" + (f"({desc})" if desc else "")
    if name in ("Edit", "Write", "NotebookEdit"):
        return f"{'作成' if name == 'Write' else '編集'}: {inp.get('file_path') or inp.get('notebook_path', '')}"
    if name == "Read":
        return f"読み込み: {inp.get('file_path', '')}"
    if name in ("Grep", "Glob"):
        return f"検索({name}): {inp.get('pattern', '')} {inp.get('path', '')}"
    if name in ("WebSearch", "WebFetch"):
        return f"Web({name}): {inp.get('query') or inp.get('url', '')}"
    text = json.dumps(inp, ensure_ascii=False)
    return f"{name}: {text[:300]}"


def make_chunks(record):
    """記録をチャンク(種類, 文)に分ける"""
    chunks = [("依頼", p) for p in split_text(record["prompt"])[:PROMPT_PARTS]]
    for step in record["steps"]:
        if step["kind"] == "text":
            chunks += [("返答", p) for p in split_text(step["text"])]
        elif step["kind"] == "tool":
            if step.get("said"):   # 質問への答えや、実行を断ったときの指示(ユーザー自身の言葉)
                chunks += [("ユーザー", p) for p in split_text(step["said"])]
            chunks.append(("操作", step["summary"]))
            if step.get("error") and step.get("result"):
                chunks.append(("エラー", f"{step['summary']}\n{step['result'][:PARA_MAX]}"))
    return [(k, t) for k, t in chunks if len(t.strip()) >= 4]


class Store:
    def __init__(self, path, embedder=None):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, data TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, key TEXT, kind TEXT, text TEXT)")
        self.db.execute("CREATE INDEX IF NOT EXISTS chunks_key ON chunks(key)")
        self.embedder = embedder or Embedder(os.path.splitext(path)[0] + "_vectors.json",
                                             model=CHUNK_MODEL, dim=CHUNK_DIM, instruct=CHUNK_INSTRUCT)
        self._index = None   # 検索用にまとめた (チャンクの鍵の番号, ベクトル行列, 記録の鍵の一覧, チャンク)

    # ---- 保存 ----

    def put(self, key, record):
        """記録を保存し、チャンクに分ける(同じ鍵で中身も同じ記録がすでにあれば何もしない。
        取り込み方を変えて中身が変わったときは、置き換える)"""
        data = json.dumps(record, ensure_ascii=False)
        row = self.db.execute("SELECT data FROM records WHERE key=?", (key,)).fetchone()
        if row and row[0] == data:
            return
        if row:
            self.db.execute("DELETE FROM chunks WHERE key=?", (key,))
        self.db.execute("INSERT OR REPLACE INTO records VALUES (?, ?)", (key, data))
        self.db.executemany("INSERT INTO chunks (key, kind, text) VALUES (?, ?, ?)",
                            [(key, k, t) for k, t in make_chunks(record)])
        self._index = None

    def commit(self):
        self.db.commit()

    def find(self, short):
        """鍵の先頭部分から記録の鍵を探す(1件に決まらなければ None)"""
        rows = self.db.execute("SELECT key FROM records WHERE key LIKE ? LIMIT 2", (short + "%",)).fetchall()
        return rows[0][0] if len(rows) == 1 else None

    def delete(self, key):
        self.db.execute("DELETE FROM chunks WHERE key=?", (key,))
        self.db.execute("DELETE FROM records WHERE key=?", (key,))
        self._index = None

    def keys(self):
        return [k for (k,) in self.db.execute("SELECT key FROM records")]

    def record(self, key):
        row = self.db.execute("SELECT data FROM records WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def ensure_vectors(self, progress=False):
        """まだベクトルのないチャンクを計算する"""
        texts = [t for (t,) in self.db.execute("SELECT text FROM chunks")]
        ok = self.embedder.ensure_docs(texts, progress=progress)
        self._index = None
        return ok

    # ---- 検索 ----

    def _build(self):
        rows = self.db.execute("SELECT id, key, kind, text FROM chunks ORDER BY id").fetchall()
        rows = [r for r in rows if doc_key(r[3]) in self.embedder.index]
        keys = sorted({r[1] for r in rows})
        key_no = {k: i for i, k in enumerate(keys)}
        owner = np.array([key_no[r[1]] for r in rows], int)
        matrix = self.embedder.matrix[[self.embedder.index[doc_key(r[3])] for r in rows]] if rows else None
        self._index = (owner, matrix, keys, rows)

    def ensure_keyword_index(self):
        """キーワード検索の索引(SQLite の全文検索 FTS5、trigram 方式)を、チャンクに合わせて作る"""
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', "
                        "content_rowid='id', tokenize='trigram')")
        # 索引に入っている文の数と最後の id(外部の表を中身にしているので、索引そのものの表で数える)
        n_fts, last_fts = self.db.execute("SELECT count(*), max(id) FROM chunks_fts_docsize").fetchone()
        n, last = self.db.execute("SELECT count(*), max(id) FROM chunks").fetchone()
        if (n_fts, last_fts) != (n, last):   # 増えた・置き換わった(置き換えると id が新しくなる)
            self.db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
            self.db.commit()
        self._chunk_count = n

    def keyword_search(self, terms, allowed):
        """語が文字どおり含まれるチャンクを探す(型番・ファイル名・エラーメッセージのような、意味より字面が大事な手がかり用)

        3文字以上の語は全文検索の索引で、2文字の語(「受注」「波形」など)は部分一致で探す。
        多くのチャンクに出てくる語ほど重みを下げる(IDF)。
        戻り値: {記録の鍵: そのやり取りのチャンクの中で一番高い点(語の重みの合計)}
        """
        if not terms or not allowed:
            return {}
        if not hasattr(self, "_chunk_count"):
            self.ensure_keyword_index()
        n = max(1, self._chunk_count)
        hits = {}   # チャンクの id -> (記録の鍵, 点)
        for term in dict.fromkeys(terms):
            if len(term) >= 3:
                rows = self.db.execute("SELECT c.id, c.key FROM chunks_fts f JOIN chunks c ON c.id = f.rowid "
                                       "WHERE chunks_fts MATCH ? LIMIT ?", ('"' + term.replace('"', '""') + '"',
                                                                             KEYWORD_MAX_HITS)).fetchall()
            else:
                rows = self.db.execute("SELECT id, key FROM chunks WHERE text LIKE ? LIMIT ?",
                                       (f"%{term}%", KEYWORD_MAX_HITS)).fetchall()
            if not rows or len(rows) >= KEYWORD_MAX_HITS:   # どこにでも出てくる語は手がかりにならない
                continue
            idf = math.log(n / len(rows))
            for cid, key in rows:
                if key in allowed:
                    k, sc = hits.get(cid, (key, 0.0))
                    hits[cid] = (k, sc + idf)
        best = {}
        for key, sc in hits.values():
            best[key] = max(best.get(key, 0.0), sc)
        return best

    def search(self, text, allowed):
        """依頼と各チャンクの意味の近さ。allowed(記録の鍵の集合)に含まれる記録のチャンクだけを対象にする

        戻り値: {記録の鍵: [(類似度, 種類, 文), ...(近い順)]}
        """
        if self._index is None:
            self._build()
        owner, matrix, keys, rows = self._index
        if matrix is None or not allowed:
            return {}
        q = self.embedder.gist(text)
        if q is None:
            return {}
        ok = np.array([k in allowed for k in keys])[owner]
        if not ok.any():
            return {}
        idx = np.where(ok)[0]
        sims = matrix[idx] @ q
        out = {}
        for i in idx[np.argsort(-sims)][:CHUNK_CANDIDATES]:
            _, key, kind, t = rows[i]
            out.setdefault(key, []).append((float(matrix[i] @ q), kind, t))
        return out


CHUNK_CANDIDATES = 300   # 依頼と近いチャンクを、上位いくつまで見るか
KEYWORD_MAX_HITS = 2000  # キーワード検索で、これ以上のチャンクに出てくる語は、ありふれていて手がかりにならない


def build_record(t):
    """読み込んだやり取りから、保存する記録を作る"""
    return {"prompt": t["user"], "steps": t["steps"], "time": t["t"], "cwd": t["cwd"],
            "src": {"file": t["session"], "uuid": t.get("uuid"), "app": t.get("source", "claude")}}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(split_text("あ" * 50 + "\n\n" + "い" * 50 + "\n\n" + "う" * 700))
