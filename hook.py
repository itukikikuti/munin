"""Claude Code のフック: メッセージを送るたびに、関係しそうな記憶を思い出して差し込む

  ~/.claude/settings.json に登録する:
    UserPromptSubmit(メッセージを送ったとき): python C:/dev/munin/hook.py          思い出して差し込む
    Stop(返答が終わったとき)               : python C:/dev/munin/hook.py --learn  そのやり取りを覚える
  パスは / で区切る(Windows でもフックは bash で実行され、\\ が消えてファイルが見つからなくなる。
  Python はファイルが見つからないと終了コード 2 で終わり、Claude Code はそれを「プロンプトを止める」と受け取る)

記憶は常駐サーバー(server.py)が持っていて、このフックはそこへ問い合わせるだけ
(毎回記憶を読み込むと、数十秒かかるため)。サーバーが動いていなければ起動し、
今回は何も差し込まずに終わる(次のメッセージから差し込まれる)。

フックが失敗しても会話は止めない(何も差し込まずに終わる)。ただし、失敗に気づけるように
logs/hook_errors.log に書き残す。
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
URL = f"http://127.0.0.1:{os.environ.get('PORT', 8770)}/api/hook"
LEARN_URL = f"http://127.0.0.1:{os.environ.get('PORT', 8770)}/api/learn_session"
LEARN = "--learn" in sys.argv   # 返答が終わったとき(Stop)に呼ばれる: 思い出すのではなく、覚える
TIMEOUT = 12          # 秒。フック全体の制限時間(settings.json の timeout)より短くする
LOG_DIR = paths.LOG_DIR


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


def problems():
    """動かすのに足りないもの(なければ空)"""
    import importlib.util
    out = []
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        out.append(f"Python のパッケージが入っていない: {', '.join(missing)}")
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as res:
            names = {m["name"] for m in json.load(res).get("models", [])}
        lacking = [m for m in MODELS if m not in names]
        if lacking:
            out.append(f"Ollama に埋め込みモデルがない: {', '.join(lacking)}")
    except (OSError, ValueError):
        out.append("Ollama が動いていない(Ollama のアプリを起動する)")
    return out


def session_start():
    """会話の始まり: 準備ができていればサーバーを起こしておき、できていなければ Claude に伝える"""
    lacking = problems()
    if lacking:
        message = ("munin(記憶のプラグイン)は、次の理由でまだ使えません。ユーザーに伝え、"
                   "/munin:setup の実行をすすめてください。\n- " + "\n- ".join(lacking))
    else:
        try:
            urllib.request.urlopen(STATUS_URL, timeout=2).read()
            return
        except OSError:
            os.makedirs(LOG_DIR, exist_ok=True)
            start_server()   # 最初のメッセージで思い出せるように、先に起こしておく
        if os.path.exists(paths.STATE):
            return
        message = ("munin(記憶のプラグイン)は、これまでの会話から最初の記憶を作っています"
                   "(会話の量によって数分〜数十分)。できるまでは思い出したことは差し込まれません。")
    out = {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": message}}
    sys.stdout.buffer.write(json.dumps(out, ensure_ascii=False).encode("utf-8"))


REQUIRED = ["janome", "wordfreq", "numpy", "PIL"]
MODELS = sorted({paths.config().get("embed_model", "qwen3-embedding:8b"), "qwen3-embedding:0.6b"})
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
STATUS_URL = f"http://127.0.0.1:{os.environ.get('PORT', 8770)}/api/status"


def main():
    # Windows では標準入力を一度に読むと空になることがあるので、最後まで読み切る
    raw = sys.stdin.buffer.read()
    try:
        event = json.loads(raw.decode("utf-8") or "{}")
    except ValueError as e:
        log_error(f"入力を読めない: {e}")
        return
    if "--session-start" in sys.argv:
        return session_start()
    if LEARN:
        # 返答が終わった: その会話の新しいやり取りを覚えてもらう(サーバーが別のスレッドで覚えるので、すぐ戻る)
        if not event.get("transcript_path"):
            return
        url, body = LEARN_URL, {"transcript_path": event["transcript_path"]}
    else:
        # アプリが依頼の前に付けるシステムの注意書きは、ユーザーの言葉ではないので除く
        prompt = re.sub(r"<system-reminder>.*?</system-reminder>", "", event.get("prompt") or "", flags=re.S).strip()
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
