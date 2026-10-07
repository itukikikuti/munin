"""想起の精度を測る(答えが分かっている指示を並べて、正しい場所を思い出せるか確かめる)

  python evaluate.py

記憶は変えない(毎回、保存された状態から思い出す)。
"""
import copy
import os
import sys
import time

from brain import Brain, is_place

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.environ.get("BRAIN_STATE", os.path.join(HERE, "brain_state.json"))

# (指示, 正しい場所の候補。None は「思い出さないのが正しい」)
CASES = [
    ("記憶モデルのサーバーを直して", [r"C:\dev\brain-memory"]),
    ("海馬の仕組み", [r"C:\dev\brain-memory"]),
    ("給与奉行のCSV", [r"C:\dev\work", r"C:\dev\work\batch\マネーフォワード連携"]),
    ("マネーフォワード連携", [r"C:\dev\work\batch\マネーフォワード連携", r"C:\dev\work"]),
    ("立替金の通知", [r"C:\dev\work", r"C:\dev\work\batch\マネーフォワード連携"]),
    ("Nucleoの波形", [r"C:\dev\ET"]),
    ("オシロの設定", [r"C:\dev\ET"]),
    ("RAGのチャンク", [r"C:\dev\rag-agent-demo"]),
    ("JavaScript研修の台本", [r"C:\Users\me\Documents\obsidian\20_案件\JavaScript研修",
                          r"C:\Users\me\Documents\obsidian"]),
    ("今日の天気", None),
    ("晩ご飯の献立", None),
]


# 文脈: 先に話題を出してから、曖昧な指示をする(最後の指示の答えを確かめる)
JS_DIR = r"C:\Users\me\Documents\obsidian\20_案件\JavaScript研修"
SEQUENCES = [
    (["RAGのチャンク", "このツールの設定を変えて"], [r"C:\dev\rag-agent-demo"]),
    (["JavaScript研修の台本", "それの続きを書いて"], [JS_DIR]),
    (["RAGのチャンク", "Nucleoの波形", "それの出力を見たい"], [r"C:\dev\ET"]),  # 話題が切り替わった
    (["Nucleoの波形", "今日の天気"], None),                                     # 文脈と無関係な話
]


def fresh(base):
    b = copy.copy(base)
    b.cortex = copy.deepcopy(base.cortex)
    b.hippocampus = copy.deepcopy(base.hippocampus)
    b.context = {}
    return b


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    base = Brain(STATE)
    ok = 0
    cases = [([t], e) for t, e in CASES] + SEQUENCES
    for texts, expected in cases:
        b = fresh(base)
        for before in texts[:-1]:
            b.recall(before)
        text = " → ".join(texts)
        t = time.time()
        _, _, ans = b.recall(texts[-1])
        dt = time.time() - t
        got = ans["place"] if not ans["uncertain"] else None
        hit = (got is None) if expected is None else (got in expected)
        ok += hit
        cues = ", ".join(f"{c}={v:.2f}" for c, v in sorted(b.last_cues.items(), key=lambda x: -x[1])[:6])
        places = sorted(((n, a) for n, a in b.last_activation.items() if is_place(n)), key=lambda x: -x[1])[:3]
        print(f"{'○' if hit else '×'} {text} → {ans['place']} (確信度 {ans['confidence']:.2f}"
              f"{' うろ覚え' if ans['uncertain'] else ''}) [{dt:.1f}s]\n    手がかり: {cues}\n"
              f"    場所の活性: " + ", ".join(f"{os.path.basename(n)}={a:.3f}" for n, a in places))
    print(f"\n正解 {ok}/{len(cases)}")


if __name__ == "__main__":
    main()
