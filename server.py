"""記憶モデルをブラウザから操作するためのローカルサーバー

  python server.py  →  http://127.0.0.1:8770

記憶は起動時に一度だけ読み込み、変更があるたびにファイルへ保存する。
ネットワーク図はサーバー側でタイル画像に描き、ブラウザは表示している部分だけを読み込む。
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from brain import Brain, is_place
from tiles import GraphMap

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))
PORT = int(os.environ.get("PORT", 8770))
AWAKE_HOURS = 16
LOCK = threading.Lock()   # 記憶の読み書きを1リクエストずつにする
HIPPO_SHOWN = 60          # 画面の一覧に出す海馬の記録(新しい順)
LINKS_SHOWN = 200         # 画面の一覧に出す皮質のリンク(強い順)

brain = None
mtime = None
# 図の作り直しが必要な変更があるたびに増える。起動時刻から始めて、再起動前のタイルをブラウザが使い回さないようにする
version = int(time.time())
graph = GraphMap(os.path.join(HERE, "layout.json"))


def load():
    """記憶を読み込む(ファイルが外から書き換えられていたら読み直す)"""
    global brain, mtime, version
    m = os.path.getmtime(STATE) if os.path.exists(STATE) else None
    if brain is None or m != mtime:
        brain = Brain(STATE)
        mtime = m
        version += 1
    return brain


def save():
    global mtime
    brain.save()
    mtime = os.path.getmtime(STATE)


def current_graph():
    """図を今の記憶に合わせる(想起による小さな強化では作り直さない。覚える・眠るなどで作り直す)"""
    with LOCK:
        b = load()
        v = version
        if graph.version != v:
            graph.update(b, v)
    return graph


def snapshot(b):
    concepts = b.known_concepts()
    links = sorted(b.cortex.items(), key=lambda x: -x[1]["w"])[:LINKS_SHOWN]
    return {
        "now": b.now,
        "version": version,
        "hippocampus": [{"concepts": ep["concepts"], "strength": b.hippo_strength(ep), "nights": ep["nights"]}
                        for ep in sorted(b.hippocampus, key=lambda e: -e["t"])[:HIPPO_SHOWN]],
        "hippocampus_total": len(b.hippocampus),
        "cortex": [{"a": k.split("\t")[0], "b": k.split("\t")[1], "w": link["w"],
                    "access": b.access(link), "eff": b.effective(link)} for k, link in links],
        "cortex_total": len(b.cortex),
        "concepts_total": len(concepts),
        "places_total": sum(1 for n in concepts if is_place(n)),
    }


def handle(path, body):
    with LOCK:
        return _handle(path, body)


def _handle(path, body):
    global version
    b = load()
    result = {}
    if path == "/api/state":
        pass
    elif path == "/api/learn":
        concepts = [c.strip() for c in body.get("concepts", []) if c.strip()]
        if len(concepts) < 2:
            return 400, {"error": "関連する概念を2つ以上入力してください"}
        b.learn(concepts)
        result = {"log": [f"海馬に記録: {' / '.join(concepts)}"]}
    elif path == "/api/recall":
        text = body.get("text", "").strip()
        if not text:
            return 400, {"error": "指示を入力してください"}
        cues, fragments, answer = b.recall(text)
        result = {"recall": {"text": text, "cues": cues, "cue_strength": b.last_cues,
                             "fragments": fragments[:30], "answer": answer,
                             "embedding": bool(b.embedder and b.embedder.available)}}
        if graph.version is not None:
            result["recall"]["overlay"] = graph.overlay(b.last_activation, cues)
    elif path == "/api/sleep":
        log = []
        for i in range(max(1, min(60, int(body.get("nights", 1))))):
            report = b.sleep()
            b.wait(AWAKE_HOURS)
            moved = sum(1 for r in report if r.startswith("皮質へ"))
            lost = sum(1 for r in report if r.startswith(("消失", "海馬から忘却")))
            log.append(f"{i + 1}晩目: 再生 {sum(1 for r in report if r.startswith('再生'))}件 / "
                       f"皮質へ移行 {moved}件 / 忘却 {lost}件")
        result = {"log": log}
    elif path == "/api/wait":
        hours = max(0.0, min(24 * 365, float(body.get("hours", 1))))
        b.wait(hours)
        result = {"log": [f"{hours:g}時間経過"]}
    elif path == "/api/reset":
        if os.path.exists(STATE):
            os.remove(STATE)
        global brain
        brain = b = Brain(STATE)
        result = {"log": ["記憶をすべて消去しました"]}
    else:
        return 404, {"error": "not found"}
    if path != "/api/state":
        save()
        if path != "/api/recall":
            version += 1
    result["state"] = snapshot(b)
    return 200, result


class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, ctype="application/json; charset=utf-8", cache=False):
        payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        # タイルは版(v)ごとに中身が決まるので、ブラウザにキャッシュさせてよい
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        elif url.path.startswith("/tiles/"):
            try:
                z, x, y = (int(p) for p in url.path[len("/tiles/"):].removesuffix(".png").split("/"))
            except ValueError:
                return self.send(404, {"error": "not found"})
            g = current_graph()
            self.send(200, g.tile(z, x, y, q.get("theme", "light")), "image/png", cache=True)
        elif url.path == "/api/graph":
            self.send(200, current_graph().meta())
        elif url.path == "/api/node_at":
            g = current_graph()
            self.send(200, {"node": g.node_at(float(q["x"]), float(q["y"]), int(q["z"]))})
        else:
            self.send(*handle(url.path, {}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            self.send(*handle(self.path, body))
        except (ValueError, TypeError) as e:
            self.send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        if not self.path.startswith("/tiles/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
