"""言葉を意味のベクトルに変える(Ollama の埋め込みモデル)

脳で言えば、聞いた言葉の意味を捉えて、近い概念に結びつける入口の部分。
概念のベクトルは一度計算したらファイルに保存し、次からは使い回す。
"""
import array
import base64
import json
import math
import os
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:8b")
# qwen3-embedding は、検索する側の文に指示文を付けると精度が上がる
QUERY_INSTRUCT = ("Instruct: Given a user's request about their work, "
                  "retrieve the concept names it refers to\nQuery: ")


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _pack(v):
    return base64.b64encode(array.array("f", v).tobytes()).decode()


def _unpack(s):
    a = array.array("f")
    a.frombytes(base64.b64decode(s))
    return list(a)


class Embedder:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        self.cache = {}
        self.available = True
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("model") == EMBED_MODEL:
                self.cache = {k: _unpack(v) for k, v in data["vectors"].items()}

    def _request(self, texts):
        body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
        req = urllib.request.Request(f"{OLLAMA_URL}/api/embed", body,
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as res:
            return [_normalize(v) for v in json.load(res)["embeddings"]]

    def _save(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump({"model": EMBED_MODEL,
                       "vectors": {k: _pack(v) for k, v in self.cache.items()}}, f)

    def concepts(self, names):
        """概念のベクトル(未計算のものだけ問い合わせる)。使えなければ None"""
        missing = [n for n in names if n not in self.cache]
        if missing:
            try:
                for i in range(0, len(missing), 64):
                    batch = missing[i:i + 64]
                    self.cache.update(zip(batch, self._request(batch)))
                self._save()
            except OSError:
                self.available = False
                return None
        return {n: self.cache[n] for n in names}

    def query(self, text):
        try:
            return self._request([QUERY_INSTRUCT + text])[0]
        except OSError:
            self.available = False
            return None


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))
