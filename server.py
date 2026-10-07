"""記憶モデルをブラウザから操作するためのローカルサーバー(標準ライブラリのみ)

  python server.py  →  http://127.0.0.1:8770

記憶は起動時に一度だけ読み込み、変更があるたびにファイルへ保存する。
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from brain import Brain, is_place

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))
PORT = int(os.environ.get("PORT", 8770))
AWAKE_HOURS = 16
LOCK = threading.Lock()   # 記憶の読み書きを1リクエストずつにする

brain = None
mtime = None


def load():
    """記憶を読み込む(ファイルが外から書き換えられていたら読み直す)"""
    global brain, mtime
    m = os.path.getmtime(STATE) if os.path.exists(STATE) else None
    if brain is None or m != mtime:
        brain = Brain(STATE)
        mtime = m
    return brain


def save():
    global mtime
    brain.save()
    mtime = os.path.getmtime(STATE)


def snapshot(b):
    """画面に送る記憶の中身(ネットワーク図に全部描くので、すべて送る)"""
    concepts = b.known_concepts()
    return {
        "now": b.now,
        "hippocampus": [{"concepts": ep["concepts"], "strength": b.hippo_strength(ep),
                         "nights": ep["nights"], "t": ep["t"]}
                        for ep in sorted(b.hippocampus, key=lambda e: -e["t"])],
        "hippocampus_total": len(b.hippocampus),
        "cortex": [{"a": k.split("\t")[0], "b": k.split("\t")[1], "w": link["w"],
                    "access": b.access(link), "eff": b.effective(link)} for k, link in b.cortex.items()],
        "cortex_total": len(b.cortex),
        "concepts_total": len(concepts),
        "places_total": sum(1 for n in concepts if is_place(n)),
        "context": list(b.context),
    }


def handle(path, body):
    with LOCK:
        return _handle(path, body)


def _handle(path, body):
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
                             "activation": b.last_activation,
                             "embedding": bool(b.embedder and b.embedder.available)}}
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
    result["state"] = snapshot(b)
    return 200, result


class Handler(BaseHTTPRequestHandler):
    def send(self, code, data, ctype="application/json; charset=utf-8"):
        payload = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                self.send(200, f.read(), "text/html; charset=utf-8")
        else:
            self.send(*handle(self.path, {}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            self.send(*handle(self.path, body))
        except (ValueError, TypeError) as e:
            self.send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.command, self.path))


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
