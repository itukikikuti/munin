"""言葉を意味のベクトルに変える(Ollama の埋め込みモデル)

脳で言えば、聞いた言葉の意味を捉えて、近い概念に結びつける入口の部分。
概念のベクトルは一度計算したらファイルに保存し、次からは使い回す。
"""
import json
import os
import sys
import urllib.request

import numpy as np

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:8b")
# モデルが出す次元をすべて使う(qwen3-embedding:8b は 4096 次元)
# qwen3-embedding は前半の次元だけでも意味を保つので、軽くしたいときは EMBED_DIM で減らせる
EMBED_DIM = int(os.environ.get("EMBED_DIM", 4096))
# qwen3-embedding は、検索する側の文に指示文を付けると精度が上がる
QUERY_INSTRUCT = ("Instruct: Given a user's request about their work, "
                  "retrieve the concept names it refers to\nQuery: ")
BATCH = 64


class Embedder:
    def __init__(self, cache_path):
        """cache_path は拡張子なしの基準パス(.json に名前、.npy にベクトルを保存)"""
        self.base = os.path.splitext(cache_path)[0]
        self.index = {}                                   # 概念 -> 行番号
        self.matrix = np.zeros((0, EMBED_DIM), np.float32)
        self.available = True
        self._last_query = None
        try:
            with open(self.base + ".json", encoding="utf-8") as f:
                meta = json.load(f)
            if meta["model"] == EMBED_MODEL and meta["dim"] == EMBED_DIM:
                self.matrix = np.load(self.base + ".npy")
                self.index = {n: i for i, n in enumerate(meta["names"])}
        except (OSError, ValueError, KeyError):
            pass

    def _request(self, texts):
        body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
        req = urllib.request.Request(f"{OLLAMA_URL}/api/embed", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            v = np.array(json.load(res)["embeddings"], np.float32)[:, :EMBED_DIM]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)

    def _save(self):
        names = sorted(self.index, key=self.index.get)
        np.save(self.base + ".npy", self.matrix)
        with open(self.base + ".json", "w", encoding="utf-8") as f:
            json.dump({"model": EMBED_MODEL, "dim": EMBED_DIM, "names": names}, f, ensure_ascii=False)

    def ensure(self, names, progress=False):
        """未計算の概念のベクトルを問い合わせる。使えなければ False"""
        missing = [n for n in dict.fromkeys(names) if n not in self.index]
        if not missing:
            return True
        try:
            for i in range(0, len(missing), BATCH):
                batch = missing[i:i + BATCH]
                vecs = self._request(batch)
                start = len(self.matrix)
                self.matrix = np.vstack([self.matrix, vecs])
                for j, n in enumerate(batch):
                    self.index[n] = start + j
                if progress:
                    print(f"\r  埋め込み {min(i + BATCH, len(missing))}/{len(missing)}",
                          end="", file=sys.stderr, flush=True)
            if progress:
                print(file=sys.stderr)
            self._save()
            return True
        except OSError:
            self.available = False
            return False

    def query(self, text):
        """検索する側の文のベクトル(直前に使った文は使い回す)"""
        if self._last_query and self._last_query[0] == text:
            return self._last_query[1]
        try:
            q = self._request([QUERY_INSTRUCT + text])[0]
        except OSError:
            self.available = False
            return None
        self._last_query = (text, q)
        return q

    def similarities(self, text, names):
        """文と各概念の意味の近さ(コサイン類似度)。使えなければ None"""
        if not self.ensure(names):
            return None
        q = self.query(text)
        if q is None:
            return None
        rows = np.array([self.index[n] for n in names])
        return dict(zip(names, (self.matrix[rows] @ q).tolist()))
