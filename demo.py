"""覚える → 睡眠 → 思い出す → 強化 → 忘却 の一周を見せるデモ"""
import os
import sys
import tempfile

from brain import Brain, format_recall

sys.stdout.reconfigure(encoding="utf-8")


def title(text):
    print(f"\n==== {text} ====")


def sleep(brain, nights):
    for _ in range(nights):
        for line in brain.sleep():
            print("  " + line)
        brain.wait(16)


path = os.path.join(tempfile.mkdtemp(), "state.json")
b = Brain(path)

title("1日目: 2つのアプリのことを聞く")
b.learn(["家計簿アプリ", "アプリ", "自作", "C:\\dev\\kakeibo"])
b.learn(["渦電流探傷器", "アプリ", "自作", "C:\\dev\\ET"])
print(b.show())

title("その日のうちに思い出す(海馬だけで想起)")
print(format_recall(*b.recall("家計簿アプリのグラフを直して")))

title("3晩眠る(海馬 → 皮質へ固定化)")
sleep(b, 3)
print(b.show())

title("4日目: 渦電流探傷器の作業をしてから「アプリ」と言われる")
print(format_recall(*b.recall("渦電流探傷器の波形表示を見たい")))
b.wait(0.5)
print(format_recall(*b.recall("アプリのここを直して")))

title("翌日、文脈なしで「アプリ」とだけ言われる")
sleep(b, 1)
print(format_recall(*b.recall("アプリのここを直して")))

title("家計簿アプリだけ毎日思い出して、30日過ごす")
for _ in range(30):
    b.recall("家計簿アプリ")
    sleep(b, 1)
print(b.show())

title("30日後、それぞれ思い出す")
print(format_recall(*b.recall("家計簿アプリを直して")))
b.context = {}
print(format_recall(*b.recall("渦電流探傷器を直して")))
