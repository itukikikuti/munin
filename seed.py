"""テスト用の記憶を入れる(約2か月分の生活を再現する)

  python seed.py   ※ 今の記憶は消えて置き換わる

昔覚えたもの・よく使うもの・最近覚えたばかりのもので、記憶の強さに差が出る。
"""
import os
import random
import sys

from brain import Brain

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))
AWAKE_HOURS = 16

# 時期ごとに覚える内容
OLD = [  # 1日目に覚えた古いプロジェクト
    ["マウスジグラー", "MouseJiggler", "C#", "常駐ツール", "自作", r"C:\dev\MouseJiggler"],
    ["業務カスタマイズ", "kintone", "JavaScript", "Vite", "業務", r"C:\dev\work"],
    ["JSEdit", "kintoneプラグイン", "Monaco Editor", "kintone", r"C:\dev\js-editor"],
    ["プラグイン集", "自動ルックアップ", "kintoneプラグイン", "kintone", r"C:\dev\kintone_plugin"],
    ["請求書リネーム", "PDF", "Python", "Googleドライブ", r"C:\dev\invoice-rename"],
    ["支払通知書リネーム", "PDF", "Python", "Googleドライブ", r"C:\dev\payment-notice-rename"],
    ["Garoonメール通知", "Garoon", "JavaScript", r"C:\dev\garoon-mail-notifier"],
    ["Difyワークフロー", "Dify", "kintone", "勘定科目", r"C:\dev\dify"],
    ["kintone", "Garoon", "サイボウズ"],
]
AI = [  # 20日目ごろ: AI関連
    ["画像生成", "FLUX.2", "Intel Arc", "XPU", "Python", r"C:\dev\ai-image"],
    ["文字起こし", "Whisper", "Python", "議事録", r"C:\dev\transcribe"],
    ["XPU文字起こし", "Whisper", "XPU", "Intel Arc", r"C:\dev\transcribe-xpu"],
    ["自作LLM", "LLM", "Python", "学習", r"C:\dev\llm"],
    ["AIテスト", "学習", "Python", r"C:\dev\ai-test"],
]
RECENT = [  # 30日目ごろ: 今の主な仕事
    ["渦電流探傷器", "オシロ", "DL1640", "3号機", "自作", r"C:\dev\ET"],
    ["MW2", "渦電流探傷器", "比較", r"C:\dev\MW2"],
    ["渦電流探傷器", "S/N", "フィルター", "開発会議"],
    ["選択読み上げ", "SelectionReader", "C#", "読み上げ", "常駐ツール", "自作", r"C:\dev\SelectionReader"],
    ["日誌アシスタント", "日誌", "C#", "常駐ツール", r"C:\dev\diary-assistant"],
    ["Markdown表示", "kintone", "marked", r"C:\dev\kintone-markdown-view"],
    ["溶接検査", "kintone", "Garoon", "JavaScript", "業務", r"C:\dev\yosetsukensa"],
    ["workflow2kintone", "Garoonプラグイン", "Garoon", "kintone", r"C:\dev\workflow2kintone"],
]
NEW = [  # 最近
    ["RAGデモ", "RAG", "Python", "検索", r"C:\dev\rag-agent-demo"],
]
TODAY = [  # 今日覚えた(まだ海馬にしかない)
    ["記憶モデル", "脳", "海馬", "Python", "自作", r"C:\dev\brain-memory"],
]

# 期間ごとに、どの作業をどのくらいの頻度で思い出すか(1日あたりの確率)
ROUTINE = {
    "1-19日目": {"業務カスタマイズの修正": .8, "プラグイン集を直して": .3, "請求書リネームが動かない": .2,
               "マウスジグラー": .05},
    "20-29日目": {"画像生成を試したい": .6, "文字起こしの精度": .5, "業務カスタマイズの修正": .4,
                "自作LLMを学習させる": .3},
    "30-59日目": {"渦電流探傷器の波形を見たい": .9, "MW2と比較": .3, "日誌アシスタントの不具合": .4,
                "溶接検査の画面を直して": .5, "業務カスタマイズの修正": .2, "画像生成を試したい": .15},
}


def live(brain, days, routine, rng):
    for _ in range(days):
        brain.wait(9)  # 朝
        for text, p in routine.items():
            if rng.random() < p:
                brain.recall(text)
                brain.wait(1)
        brain.now = brain.now - brain.now % 24 + AWAKE_HOURS  # その日の寝る時刻へ
        brain.sleep()


def learn_all(brain, episodes):
    for ep in episodes:
        brain.learn(ep)
        brain.wait(0.5)


def main():
    if os.path.exists(STATE):
        os.remove(STATE)
    rng = random.Random(42)
    b = Brain(STATE)
    learn_all(b, OLD)
    live(b, 19, ROUTINE["1-19日目"], rng)
    learn_all(b, AI)
    live(b, 10, ROUTINE["20-29日目"], rng)
    learn_all(b, RECENT)
    live(b, 28, ROUTINE["30-59日目"], rng)
    learn_all(b, NEW)
    live(b, 2, ROUTINE["30-59日目"], rng)
    b.wait(9)
    learn_all(b, TODAY)
    b.context = {}
    b.save()
    places = {k for key in b.cortex for k in key.split("\t") if k.startswith("C:")}
    print(f"{b.now / 24 + 1:.0f}日目まで再現しました: 皮質のリンク {len(b.cortex)}本 / "
          f"海馬の記録 {len(b.hippocampus)}件 / 場所 {len(places)}か所")


if __name__ == "__main__":
    main()
