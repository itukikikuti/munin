---
description: munin(記憶のプラグイン)を使えるようにする(llama.cpp・日本語の辞書・埋め込みモデルを取ってきて、最初の記憶を作る)
argument-hint: "[--full | --light]"
allowed-tools: Bash
---

munin の準備をします。次のコマンドを実行し、結果をユーザーに日本語で短く伝えてください。

```bash
node --no-warnings "${CLAUDE_PLUGIN_ROOT}/src/cli.ts" setup $ARGUMENTS
```

- 足りないものだけを取ってきます。llama.cpp(約30MB)、日本語の辞書 kuromoji(約17MB)、埋め込みモデル(大きいモデルは約4.7GB、軽いモデルは約0.6GB。Ollama にすでにあればそれを使う)。
- Node.js が古いと言われたら、https://nodejs.org から 22.18 以上を入れるよう伝えてください。
- 最初の記憶づくりは裏で進みます。終わるまでは、思い出したことは差し込まれません。
