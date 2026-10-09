"""記憶の鍵から、そのやり取りの記録を詳しく見る

  python memory_detail.py 鍵            ※ 長い返答やツールの結果は先頭だけ
  python memory_detail.py 鍵 --full     ※ 保存してあるものを全部

鍵は、思い出したことに [ ] で付いている記録の鍵(先頭8文字でよい)。
ツールの結果の全文が必要なときは、表示される「元の記録」(セッションの記録ファイルと発言のID)を開く。
"""
import os
import sys

from store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
SHORT = 800   # --full でないときに表示する長さ


def render(record, full=False):
    cut = (lambda t: t) if full else (lambda t: t if len(t) <= SHORT else t[:SHORT] + f"…(残り{len(t) - SHORT}文字)")
    src = record.get("src") or {}
    lines = [f"日時: {record.get('time', '')}", f"作業フォルダ: {record.get('cwd', '')}",
             f"元の記録: {src.get('file', '')} (発言のID {src.get('uuid', '')})", "", "## 依頼", cut(record["prompt"]), "",
             "## やり取り"]
    for step in record["steps"]:
        if step["kind"] == "text":
            lines += ["", "### 返答", cut(step["text"])]
        elif step["kind"] == "thinking":
            if full:   # 考えた過程は長いので、--full のときだけ
                lines += ["", "### 考えたこと", step["text"]]
        elif step["kind"] == "interrupt":
            lines += ["", "### (ユーザーがここで止めた)"]
        else:
            lines += ["", f"### {step['summary']}"]
            if step.get("said"):
                lines.append(f"ユーザーの言葉: {cut(step['said'])}")
            if step.get("result"):
                lines.append(("(エラー) " if step.get("error") else "") + cut(step["result"]))
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        print(__doc__)
        return
    store = Store(os.path.join(HERE, "records.sqlite"))
    key = store.find(sys.argv[1])
    if not key:
        print(f"鍵 {sys.argv[1]} の記録が見つからない(または1件に決まらない)")
        return
    print(render(store.record(key), full="--full" in sys.argv))


if __name__ == "__main__":
    main()
