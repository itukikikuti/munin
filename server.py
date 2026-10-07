"""記憶モデルをブラウザから操作するためのローカルサーバー(標準ライブラリのみ)

  python server.py  →  http://127.0.0.1:8770
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
LOCK = threading.Lock()  # 状態ファイルの読み書きを1リクエストずつにする


def snapshot(brain):
    return {
        "now": brain.now,
        "hippocampus": [{"concepts": ep["concepts"], "strength": brain.hippo_strength(ep),
                         "nights": ep["nights"]} for ep in brain.hippocampus],
        "cortex": [{"a": k.split("\t")[0], "b": k.split("\t")[1], "w": link["w"],
                    "access": brain.access(link), "eff": brain.effective(link)}
                   for k, link in brain.cortex.items()],
        "context": list(brain.context),
        "places": sorted(n for n in brain.known_concepts() if is_place(n)),
    }


def handle(path, body):
    with LOCK:
        return _handle(path, body)


def _handle(path, body):
    brain = Brain(STATE)
    result = {}
    if path == "/api/state":
        pass
    elif path == "/api/learn":
        concepts = [c.strip() for c in body.get("concepts", []) if c.strip()]
        if len(concepts) < 2:
            return 400, {"error": "関連する概念を2つ以上入力してください"}
        brain.learn(concepts)
        result = {"log": [f"海馬に記録: {' / '.join(concepts)}"]}
    elif path == "/api/recall":
        text = body.get("text", "").strip()
        if not text:
            return 400, {"error": "指示を入力してください"}
        cues, fragments, answer = brain.recall(text)
        result = {"recall": {"text": text, "cues": cues, "fragments": fragments,
                             "answer": answer, "activation": brain.last_activation}}
    elif path == "/api/sleep":
        log = []
        for i in range(max(1, min(60, int(body.get("nights", 1))))):
            log.append(f"--- {i + 1}晩目 ---")
            log += brain.sleep() or ["(再生なし)"]
            brain.wait(AWAKE_HOURS)
        result = {"log": log}
    elif path == "/api/wait":
        hours = max(0.0, min(24 * 365, float(body.get("hours", 1))))
        brain.wait(hours)
        result = {"log": [f"{hours:g}時間経過"]}
    elif path == "/api/reset":
        if os.path.exists(STATE):
            os.remove(STATE)
        brain = Brain(STATE)
        result = {"log": ["記憶をすべて消去しました"]}
    else:
        return 404, {"error": "not found"}
    if path != "/api/state":
        brain.save()
    result["state"] = snapshot(brain)
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
