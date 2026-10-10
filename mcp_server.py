"""Claude が自分で使える記憶の道具(MCP サーバー、標準入出力)

  recall        : 言葉を手がかりに、関係する過去を思い出す(差し込まれた分では足りないときに、自分から探す)
  recall_detail : 思い出したことに付いている鍵 [xxxxxxxx] から、そのやり取りの記録を開く

記憶は常駐サーバー(server.py)が持っているので、ここはそこへ取り次ぐだけ。
サーバーが動いていなければ起動して待つ。外部のライブラリは使わない。
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hook  # noqa: E402  サーバーの起動のしかたを使い回す

BASE = f"http://127.0.0.1:{os.environ.get('PORT', 8770)}"
VERSION = "0.1.0"
TOOLS = [
    {"name": "recall",
     "description": "これまでの会話(Claude Code と Codex のすべてのセッション)の記憶から、言葉を手がかりに関係する過去を思い出す。"
                    "メッセージごとに自動で差し込まれる記憶では足りないとき、別の言い方や関係しそうな語で探し直すのに使う。"
                    "結果は手がかりと鍵の形で返るので、詳しく見たいものは recall_detail で開く。",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "思い出したいことの手がかり(依頼文のような文でも、語の並びでもよい)"},
         "cwd": {"type": "string", "description": "今の作業フォルダ(あれば、その場所での出来事を思い出しやすくなる)"}},
         "required": ["query"]}},
    {"name": "recall_detail",
     "description": "思い出したことに付いている鍵(例: [8351eb2f] の 8351eb2f)から、そのやり取りの記録を開く。"
                    "依頼の全文、返答、使ったツールとその結果が見られる。",
     "inputSchema": {"type": "object", "properties": {
         "key": {"type": "string", "description": "記録の鍵(先頭8文字でよい)"},
         "full": {"type": "boolean", "description": "長い返答やツールの結果も省略せずに見る", "default": False}},
         "required": ["key"]}},
]


def call_server(path, body=None, timeout=30):
    """常駐サーバーに問い合わせる(動いていなければ起動して、準備ができるまで待つ)"""
    url = BASE + path
    for attempt in range(2):
        try:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
            req = urllib.request.Request(url, data, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.load(res)
        except urllib.error.HTTPError as e:
            return json.loads(e.read() or b"{}")
        except urllib.error.URLError:
            if attempt:
                raise
            os.makedirs(hook.LOG_DIR, exist_ok=True)
            hook.start_server()
            for _ in range(60):   # 記憶を読み込むまで待つ
                time.sleep(1)
                try:
                    urllib.request.urlopen(BASE + "/api/status", timeout=2).read()
                    break
                except OSError:
                    pass
    return {}


def run_tool(name, args):
    if name == "recall":
        r = call_server("/api/search", {"query": args.get("query", ""), "cwd": args.get("cwd")})
        if r.get("importing"):
            return "まだ最初の記憶を作っている途中です。しばらくしてから使ってください。"
        return r.get("text") or "関係しそうな記憶は思い出せませんでした。"
    if name == "recall_detail":
        q = urllib.parse.urlencode({"id": args.get("key", "").strip("[] ")} | ({"full": 1} if args.get("full") else {}))
        r = call_server(f"/api/record?{q}")
        return r.get("text") or r.get("error") or "記録が見つかりませんでした。"
    raise ValueError(f"知らない道具: {name}")


def handle(msg):
    method, mid = msg.get("method"), msg.get("id")
    if mid is None:   # 通知(initialized など)には返事をしない
        return None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {}}, "serverInfo": {"name": "munin", "version": VERSION}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        params = msg.get("params") or {}
        try:
            text, error = run_tool(params.get("name"), params.get("arguments") or {}), False
        except Exception as e:  # noqa: BLE001  道具の失敗は、Claude に伝えて続ける
            text, error = f"失敗しました: {type(e).__name__}: {e}", True
        result = {"content": [{"type": "text", "text": text}], "isError": error}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main():
    stdin = sys.stdin.buffer
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            reply = handle(json.loads(line.decode("utf-8")))
        except ValueError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            sys.stdout.buffer.write(json.dumps(reply, ensure_ascii=False).encode("utf-8") + b"\n")
            sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
