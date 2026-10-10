"""言葉を意味のベクトルに変える(Ollama の埋め込みモデル)

脳で言えば、聞いた言葉の意味を捉えて、近い概念に結びつける入口の部分。
  概念のベクトル : 概念の名前そのもの
  要旨のベクトル : 指示や出来事の文全体(検索する側と同じ形にそろえるので、指示文を付けて計算する)
どちらも一度計算したらファイルに保存し、次からは使い回す。
"""
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

import paths

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
# 概念と依頼の意味に使うモデル。GPU のない PC では、~/.munin/config.json で軽いモデルにできる
# (install.py が GPU の有無を見て決める)
EMBED_MODEL = os.environ.get("EMBED_MODEL") or paths.config().get("embed_model", "qwen3-embedding:8b")
# モデルが出す次元をすべて使う(qwen3-embedding:8b は 4096 次元、0.6b は 1024 次元)
# qwen3-embedding は前半の次元だけでも意味を保つので、軽くしたいときは EMBED_DIM で減らせる
EMBED_DIM = int(os.environ.get("EMBED_DIM") or paths.config().get("embed_dim", 4096))
# qwen3-embedding は、検索する側の文に指示文を付けると精度が上がる
QUERY_INSTRUCT = ("Instruct: Given a user's request about their work, "
                  "retrieve the concept names it refers to\nQuery: ")
MAX_TEXT = 2000          # 貼り付けたログなどで長すぎる文は、先頭だけで要旨をとる
BATCH = 64
KEEP_ALIVE = "30m"
NUM_CTX = 2048           # 文は MAX_TEXT 文字までに切るので、この長さで足りる
QUERY_BATCH = 16


def gist_key(text):
    """要旨のベクトルを保存するときの鍵"""
    return "Q:" + hashlib.sha1(text[:MAX_TEXT].encode()).hexdigest()


def doc_key(text):
    """検索される側の文のベクトルを保存するときの鍵"""
    return "D:" + hashlib.sha1(text[:MAX_TEXT].encode()).hexdigest()


class Embedder:
    def __init__(self, cache_path, model=EMBED_MODEL, dim=EMBED_DIM, instruct=QUERY_INSTRUCT):
        """cache_path は拡張子なしの基準パス(.json に鍵、.npy にベクトルを保存)

        model・dim・instruct を変えると、別のモデルで別のキャッシュを持つ埋め込みとして使える
        """
        self.model, self.dim, self.instruct = model, dim, instruct
        self.base = os.path.splitext(cache_path)[0]
        self.index = {}             # 鍵(概念の名前、または要旨の鍵) -> 行番号
        self.rows = []              # 追加分のベクトル(必要になったときに行列へまとめる)
        self._matrix = np.zeros((0, dim), np.float32)
        self.dirty = False
        self.available = True
        self._last_query = None
        try:
            with open(self.base + ".json", encoding="utf-8") as f:
                meta = json.load(f)
            if meta["model"] == model and meta["dim"] == dim:
                self._matrix = np.load(self.base + ".npy")
                self.index = {n: i for i, n in enumerate(meta["names"])}
        except (OSError, ValueError, KeyError):
            pass

    @property
    def matrix(self):
        if self.rows:
            self._matrix = np.vstack([self._matrix, np.array(self.rows, np.float32)])
            self.rows = []
        return self._matrix

    def _add(self, key, vec):
        self.index[key] = len(self._matrix) + len(self.rows)
        self.rows.append(vec)
        self.dirty = True

    def _request(self, texts):
        # モデルを読み込み直すと数秒かかるので、使い終わってもしばらく載せたままにする。
        # 文脈長を絞らないと、既定の長さ(8b は 40960)の分だけメモリを取り、8GB の GPU では
        # 2つのモデルが同時に載らず、切り替えのたびに読み込み直しになる(1回7秒 → 絞ると 0.02秒)
        body = json.dumps({"model": self.model, "input": texts, "keep_alive": KEEP_ALIVE,
                           "options": {"num_ctx": NUM_CTX}}).encode()
        req = urllib.request.Request(f"{OLLAMA_URL}/api/embed", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            v = np.array(json.load(res)["embeddings"], np.float32)[:, :self.dim]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def save(self):
        if not self.dirty:
            return
        names = sorted(self.index, key=self.index.get)
        np.save(self.base + ".npy", self.matrix)
        with open(self.base + ".json", "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "dim": self.dim, "names": names}, f, ensure_ascii=False)
        self.dirty = False

    def _fill(self, keys, inputs, batch, progress, label):
        try:
            for i in range(0, len(keys), batch):
                for k, v in zip(keys[i:i + batch], self._request(inputs[i:i + batch])):
                    self._add(k, v)
                if progress:
                    print(f"\r  {label} {min(i + batch, len(keys))}/{len(keys)}", end="", file=sys.stderr, flush=True)
            if progress and keys:
                print(file=sys.stderr)
            if keys:
                self.save()
            return True
        except OSError:
            self.available = False
            return False

    def ensure(self, names, progress=False):
        """未計算の概念のベクトルを問い合わせる。使えなければ False"""
        missing = [n for n in dict.fromkeys(names) if n not in self.index]
        return self._fill(missing, missing, BATCH, progress, "概念の埋め込み")

    def ensure_gists(self, texts, progress=False):
        """未計算の要旨のベクトルを問い合わせる。使えなければ False"""
        todo = {gist_key(t): self.instruct + t[:MAX_TEXT] for t in texts}
        missing = [k for k in todo if k not in self.index]
        return self._fill(missing, [todo[k] for k in missing], QUERY_BATCH, progress, "要旨の埋め込み")

    def ensure_docs(self, texts, progress=False):
        """検索される側の文(指示文を付けない)のベクトルを問い合わせる。鍵は doc_key(文)"""
        todo = {doc_key(t): t[:MAX_TEXT] for t in texts}
        missing = [k for k in todo if k not in self.index]
        return self._fill(missing, [todo[k] for k in missing], BATCH, progress, "文の埋め込み")

    def gist(self, text):
        """文の要旨のベクトル(計算済みなら使い回す)。使えなければ None"""
        key = gist_key(text)
        if key in self.index:
            i = self.index[key]
            return self.matrix[i]
        if self._last_query and self._last_query[0] == key:
            return self._last_query[1]
        try:
            q = self._request([self.instruct + text[:MAX_TEXT]])[0]
        except OSError:
            self.available = False
            return None
        self._last_query = (key, q)
        return q

    def remember_gist(self, text):
        """覚える出来事の要旨のベクトルを保存し、鍵を返す"""
        key = gist_key(text)
        if key not in self.index:
            v = self.gist(text)
            if v is None:
                return None
            self._add(key, v)
        return key

    def similarities(self, text, keys):
        """文と、各鍵(概念・要旨)の意味の近さ(コサイン類似度)。使えなければ None"""
        q = self.gist(text)
        if q is None:
            return None
        rows = np.fromiter((self.index[k] for k in keys), int, len(keys))
        return self.matrix[rows] @ q
