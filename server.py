"""記憶モデルをブラウザから操作するためのローカルサーバー

  python server.py  →  http://127.0.0.1:8770

記憶は起動時に一度だけ読み込み、変更があるたびにファイルへ保存する。
ネットワーク図はサーバー側でタイル画像に描き、ブラウザは表示している部分だけを読み込む。
"""
import json
import os
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import import_sessions as experience
from brain import Brain, is_place
from memory_detail import render
from tiles import GraphMap

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))
PORT = int(os.environ.get("PORT", 8770))
AWAKE_HOURS = 16
LOCK = threading.Lock()   # 記憶の読み書きを1リクエストずつにする
HIPPO_SHOWN = 60          # 画面の一覧に出す海馬の記録(新しい順)
LINKS_SHOWN = 200         # 画面の一覧に出す皮質のリンク(強い順)

LOG_DIR = os.path.join(HERE, "logs")
AUTOSAVE_SECONDS = 600

brain = None
mtime = None
live_session = None   # フックで最後に思い出した会話
life = None           # 使いながら覚えるための、体験の続きの状態(全件の取り込みで作られる)
LIFE = os.path.join(os.path.dirname(STATE), "life_state.json")
learn_queue = queue.Queue()   # 返答が終わった会話(覚える順番待ち)
CATCH_UP_SECONDS = 3600       # Codex の会話や、取りこぼした会話を拾い直す間隔
dirty = False         # フックでの想起による強化が、まだ保存されていない
# 図の作り直しが必要な変更があるたびに増える。起動時刻から始めて、再起動前のタイルをブラウザが使い回さないようにする
version = int(time.time())
graph = GraphMap(os.path.join(HERE, "layout.json"))


def load():
    """記憶を読み込む(ファイルが外から書き換えられていたら読み直す)"""
    global brain, mtime, version, life
    m = os.path.getmtime(STATE) if os.path.exists(STATE) else None
    if brain is None or m != mtime:
        brain = Brain(STATE)
        life = experience.Life.restore(brain, LIFE)   # 全件の取り込み直しのあとは、その続きから覚える
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


def hook_recall(b, body):
    """Claude Code のフック(メッセージ送信時)から呼ばれる: 思い出したことを返し、何を出したかを記録する

    記録は、しばらく使ったあとで「差し込んだ記憶を Claude が実際に使ったか」を分析するためのもの。
    """
    global live_session, dirty
    t0 = time.time()
    prompt = (body.get("prompt") or "").strip()
    sid = body.get("session_id") or ""
    session = f"{sid}.jsonl" if sid else None   # 記憶の中では、会話をセッション記録のファイル名で区別している
    if session != live_session:
        b.context = {}   # 新しい会話(または別の会話に切り替わった): 前の会話で考えていたことは持ち越さない
        live_session = session
    new_conversation = not b.context
    r = b.recollect(prompt, session=session, cwd=body.get("cwd") or None)
    dirty = True
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "session_id": sid, "cwd": body.get("cwd"),
        "transcript_path": body.get("transcript_path"), "prompt": prompt, "new_conversation": new_conversation,
        "episodes": [{"ref": e["ref"], "conf": round(e["conf"], 3), "prompt": e["prompt"], "when": e["when"]}
                     for e in r["episodes"]],
        "ranked": [{"ref": e["ref"], "conf": round(e["conf"], 3)} for e in r["ranked"]],
        "todos": [{"id": it["id"], "text": it["text"], "due": it["due"]} for it in r["todos"]],
        "recent": [e["ref"] for e in r["recent"]],
        "places": [p["name"] for p in r["places"]], "concepts": [c["name"] for c in r["concepts"]],
        "cues": r["cues"][:20], "text": r["text"], "ms": round((time.time() - t0) * 1000),
    }
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "recall_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"text": r["text"], "ms": entry["ms"]}


def autosave():
    """フックでの想起による強化を、ときどき保存する(1回ごとに 20MB を書くと遅いため)

    取り込み直しなどで記憶のファイルが外から書き換えられていたら、保存せずに読み直しに任せる。
    """
    global dirty
    while True:
        time.sleep(AUTOSAVE_SECONDS)
        with LOCK:
            m = os.path.getmtime(STATE) if os.path.exists(STATE) else None
            if dirty and brain is not None and m == mtime:
                save()
            dirty = False


def learn(files=(), codex=(), settle=False):
    """セッション記録の新しいやり取りを覚え、すぐ保存する(呼ぶ側で LOCK を取る)"""
    global dirty
    b = load()
    if life is None:
        return 0
    t0 = time.time()
    keep = (b.context, b.seq)   # 覚えるときの体験で、今の会話のワーキングメモリを崩さない
    n = experience.learn_new(life, files, codex, settle)
    b.context, b.seq = keep
    if n:
        save()
        life.save(LIFE)
        dirty = False
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "learn_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": len(files) + len(codex),
                            "learned": n, "ms": round((time.time() - t0) * 1000)}, ensure_ascii=False) + "\n")
    return n


def learner():
    """返答が終わった会話を、順番に覚える"""
    while True:
        path = learn_queue.get()
        try:
            with LOCK:
                if os.path.exists(path):
                    learn([path])
        except Exception:  # noqa: BLE001  覚えるのに失敗しても、思い出す方は止めない
            log_error(traceback.format_exc())


def catch_up():
    """前に読んだあとで書き足されたセッション記録を拾い直す(Codex の会話や、フックが動かなかった会話)"""
    while True:
        try:
            with LOCK:
                load()
                if life is not None:
                    changed = lambda ps: [p for p in ps if os.path.getmtime(p) > life.scanned.get(p, 0)]
                    files, codex = changed(experience.claude_files()), changed(experience.codex_files())
                    if files or codex:
                        learn(files, codex, settle=True)
                        for p in files + codex:   # 途中かもしれないファイルは、次の回にもう一度読む
                            m = os.path.getmtime(p)
                            if time.time() - m >= experience.SETTLE_SECONDS:
                                life.scanned[p] = m
                        life.save(LIFE)
        except Exception:  # noqa: BLE001
            log_error(traceback.format_exc())
        time.sleep(CATCH_UP_SECONDS)


def log_error(message):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "server_errors.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


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
        r = b.recollect(text, cwd=body.get("cwd") or None)   # 作業フォルダ(あれば、その場所の出来事を思い出しやすくする)
        r.pop("ranked", None)
        result = {"recall": {**r, "query": text, "cue_strength": b.last_cues,
                             "embedding": bool(b.embedder and b.embedder.available)}}
        if graph.version is not None:
            result["recall"]["overlay"] = graph.overlay(b.last_activation, r["cues"])
    elif path == "/api/hook":
        return 200, hook_recall(b, body)
    elif path == "/api/learn_session":   # 返答が終わった: その会話の新しいやり取りを、あとで(別のスレッドで)覚える
        if body.get("transcript_path"):
            learn_queue.put(body["transcript_path"])
        return 200, {"queued": learn_queue.qsize()}
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
    if path not in ("/api/state", "/api/hook", "/api/learn_session"):
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
        elif url.path == "/api/record":
            b = load()
            key = b.store.find(q.get("id", "")) if b.store else None
            record = b.store.record(key) if key else None
            self.send(200 if record else 404, {"key": key, "text": render(record, full="full" in q)} if record
                      else {"error": "記録が見つからない"})
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
            log_error(f"{self.path}: {traceback.format_exc()}")   # フックからの問い合わせの失敗に気づけるように
            self.send(400, {"error": str(e)})

    def log_message(self, fmt, *args):
        if not self.path.startswith("/tiles/"):
            sys.stderr.write("%s %s\n" % (self.command, self.path))


def warm_up():
    """最初のメッセージで待たせないよう、記憶と検索の準備を起動時に済ませておく"""
    with LOCK:
        b = load()
        b.recollect("起動", reinforce=False)
        b.context = {}


if __name__ == "__main__":
    print(f"http://127.0.0.1:{PORT}", flush=True)
    threading.Thread(target=warm_up, daemon=True).start()
    threading.Thread(target=autosave, daemon=True).start()
    threading.Thread(target=learner, daemon=True).start()
    threading.Thread(target=catch_up, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
