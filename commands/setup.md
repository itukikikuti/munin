---
description: munin(記憶のプラグイン)を使えるようにする(Python のパッケージ・Ollama の埋め込みモデル・最初の記憶)
argument-hint: "[--full | --light]"
allowed-tools: Bash
---

munin の準備をします。次のコマンドを実行し、結果をユーザーに日本語で短く伝えてください。

```bash
python "${CLAUDE_PLUGIN_ROOT}/install.py" $ARGUMENTS
```

- 足りないものだけを入れます。Python のパッケージ(数十MB)と、Ollama の埋め込みモデル(大きいモデルは約4.7GB、軽いモデルは約0.6GB)をダウンロードすることがあります。
- Ollama が入っていない、または動いていないと言われたら、https://ollama.com から入れてアプリを起動するよう伝えてください。
- 最初の記憶づくりは裏で進みます。終わるまでは、思い出したことは差し込まれません。
