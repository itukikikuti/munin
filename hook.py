"""Claude Code のフック: メッセージを送るたびに、関係しそうな記憶を思い出して差し込む

  ~/.claude/settings.json に登録する:
    UserPromptSubmit(メッセージを送ったとき): python C:\\dev\\brain-memory\\hook.py          思い出して差し込む
    Stop(返答が終わったとき)               : python C:\\dev\\brain-memory\\hook.py --learn  そのやり取りを覚える

記憶は常駐サーバー(server.py)が持っていて、このフックはそこへ問い合わせるだけ
(毎回記憶を読み込むと、数十秒かかるため)。サーバーが動いていなければ起動し、
今回は何も差し込まずに終わる(次のメッセージから差し込まれる)。

フックが失敗しても会話は止めない(何も差し込まずに終わる)。ただし、失敗に気づけるように
logs/hook_errors.log に書き残す。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
URL = f"http://127.0.0.1:{os.environ.get('PORT', 8770)}/api/hook"
LEARN_URL = f"http://127.0.0.1:{os.environ.get('PORT', 8770)}/api/learn"
LEARN = "--learn" in sys.argv   # 返答が終わったとき(Stop)に呼ばれる: 思い出すのではなく、覚える
TIMEOUT = 12          # 秒。フック全体の制限時間(settings.json の timeout)より短くする
LOG_DIR = os.path.join(HERE, "logs")


def log_error(message):
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "hook_errors.log"), "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def start_server():
    """常駐サーバーを、画面を出さずに起動する(このフックが終わっても動き続けるように切り離す)"""
    pythonw = os.path.join(os.path.dirname(sys.executable), "pythonw.exe")
    exe = pythonw if os.path.exists(pythonw) else sys.executable
    flags = 0x00000008 | 0x00000200 if os.name == "nt" else 0   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    with open(os.path.join(LOG_DIR, "server.log"), "ab") as out:
        subprocess.Popen([exe, os.path.join(HERE, "server.py")], cwd=HERE, stdout=out, stderr=out,
                         stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)


def main():
    # Windows では標準入力を一度に読むと空になることがあるので、最後まで読み切る
    raw = sys.stdin.buffer.read()
    try:
        event = json.loads(raw.decode("utf-8") or "{}")
    except ValueError as e:
        log_error(f"入力を読めない: {e}")
        return
    if LEARN:
        # 返答が終わった: その会話の新しいやり取りを覚えてもらう(サーバーが別のスレッドで覚えるので、すぐ戻る)
        if not event.get("transcript_path"):
            return
        url, body = LEARN_URL, {"transcript_path": event["transcript_path"]}
    else:
        prompt = (event.get("prompt") or "").strip()
        if not prompt or prompt.startswith("/"):   # コマンドには記憶は要らない
            return
        url, body = URL, {"prompt": prompt, "session_id": event.get("session_id"), "cwd": event.get("cwd"),
                          "transcript_path": event.get("transcript_path")}
    req = urllib.request.Request(url, json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as res:
            text = json.load(res).get("text", "")
    except urllib.error.HTTPError as e:
        log_error(f"問い合わせに失敗: {e} {e.read().decode('utf-8', 'replace')[:300]}")
        return
    except urllib.error.URLError as e:
        if isinstance(e.reason, ConnectionRefusedError) or "refused" in str(e.reason).lower() \
                or getattr(e.reason, "winerror", None) == 10061:
            os.makedirs(LOG_DIR, exist_ok=True)
            start_server()
            log_error("サーバーが動いていなかったので起動した(今回は差し込みなし)")
        else:
            log_error(f"問い合わせに失敗: {e}")
        return
    except Exception as e:  # noqa: BLE001  フックの失敗で会話を止めない
        log_error(f"問い合わせに失敗: {type(e).__name__}: {e}")
        return
    if text:
        out = {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": text}}
        sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))


if __name__ == "__main__":
    main()
