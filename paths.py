"""記憶のデータの置き場所

プログラム(プラグインとして入れると、更新のたびに入れ替わるフォルダに置かれる)とは分けて、
ユーザーのフォルダの下(~/.munin)に置く。MUNIN_HOME で変えられる。

  brain_state.json   記憶そのもの(皮質・海馬・痕跡・やること)
  life_state.json    続きから覚えるための、体験の途中の状態
  records.sqlite     やり取りの全文と検索用の索引
  embeddings.*       概念と依頼の意味のベクトル(作り直せるキャッシュ)
  records_vectors.*  全文の部分ごとの意味のベクトル(作り直せるキャッシュ)
  layout.json        ネットワーク図の配置
  logs/              使ったときの記録(差し込んだ内容・覚えた件数・エラー)
  config.json        設定(使う埋め込みモデルなど。install.py が作る)
"""
import json
import os

STATE = os.environ.get("BRAIN_STATE") or os.path.join(
    os.environ.get("MUNIN_HOME") or os.path.join(os.path.expanduser("~"), ".munin"), "brain_state.json")
DATA = os.path.dirname(os.path.abspath(STATE))
LOG_DIR = os.path.join(DATA, "logs")
CONFIG = os.path.join(DATA, "config.json")


def config():
    """設定(なければ空)"""
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}
