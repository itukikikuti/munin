"""munin を使えるようにする(最初の1回。/munin:setup から呼ばれる。setup.py という名前は Python の配布の決まりと紛らわしいので避けた)

  python install.py            ※ GPU の有無を見て、使う埋め込みモデルを決める
  python install.py --full     ※ GPU がなくても大きいモデル(qwen3-embedding:8b)を使う
  python install.py --light    ※ GPU があっても軽いモデル(qwen3-embedding:0.6b)だけを使う

  1. Python のパッケージを入れる(janome・wordfreq・numpy・Pillow)
  2. Ollama に埋め込みモデルを入れる(8b は約4.7GB、0.6b は約0.6GB)
  3. 常駐サーバーを起動する。記憶がまだなければ、これまでのすべての会話から最初の記憶を作り始める
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402

PACKAGES = {"janome": "janome", "wordfreq": "wordfreq", "numpy": "numpy", "PIL": "Pillow"}
FULL = {"embed_model": "qwen3-embedding:8b", "embed_dim": 4096}
LIGHT = {"embed_model": "qwen3-embedding:0.6b", "embed_dim": 1024}
CHUNK_MODEL = "qwen3-embedding:0.6b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def step(text):
    print(f"\n== {text}", flush=True)


def has_gpu():
    """埋め込みを速く計算できる GPU があるか(NVIDIA だけを見る)"""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split()) >= 6000   # 8b を載せるには 6GB 程度は要る
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(paths.DATA, exist_ok=True)
    ok = True

    step("1. Python のパッケージ")
    missing = [pip for mod, pip in PACKAGES.items() if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"入れます: {', '.join(missing)}", flush=True)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--user", *missing])
        if r.returncode:
            print("失敗しました。上のエラーを確かめてください。")
            ok = False
    else:
        print("そろっています")

    step("2. 埋め込みモデル")
    config = paths.config()
    if "--full" in sys.argv or "--light" in sys.argv or "embed_model" not in config:
        light = "--light" in sys.argv or ("--full" not in sys.argv and not has_gpu())
        config.update(LIGHT if light else FULL)
        with open(paths.CONFIG, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("GPU が見つからないので、軽いモデルだけを使います(--full で大きいモデルにできます)" if light
              else "大きいモデルを使います(精度が高い)")
    print(f"概念と依頼の意味: {config['embed_model']} / 全文の部分: {CHUNK_MODEL}")
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as res:
            have = {m["name"] for m in json.load(res).get("models", [])}
    except OSError:
        print("Ollama が動いていません。https://ollama.com から入れて、アプリを起動してから、もう一度実行してください。")
        return 1
    for model in sorted({config["embed_model"], CHUNK_MODEL}):
        if model in have:
            print(f"{model}: 入っています")
            continue
        print(f"{model}: 入れます(数分かかります)", flush=True)
        if subprocess.run(["ollama", "pull", model]).returncode:
            print(f"{model} を入れられませんでした。")
            ok = False
    if not ok:
        return 1

    step("3. 記憶")
    import hook
    try:
        urllib.request.urlopen(hook.STATUS_URL, timeout=2).read()
        print("常駐サーバーはもう動いています")
    except OSError:
        os.makedirs(paths.LOG_DIR, exist_ok=True)
        hook.start_server()
        print("常駐サーバーを起動しました")
    if os.path.exists(paths.STATE):
        print(f"記憶があります({paths.DATA})。準備ができました。")
    else:
        print("これまでのすべての会話から、最初の記憶を作り始めました(会話の量によって数分〜数十分)。\n"
              f"進み具合は {os.path.join(paths.LOG_DIR, 'import.log')} で見られます。"
              "できるまでは、思い出したことは差し込まれません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
