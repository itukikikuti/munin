"""展望記憶: これからやること(やり残し・期限・返事待ち)を覚え、終わったら忘れる

過去の出来事を思い出す記憶(エピソード記憶)とは別に、人は「あとで〜する」「〜日までに〜する」という
未来の行動の記憶を持っている(展望記憶)。依頼の意味が近いかどうかでは思い出せない(「次なにする？」には
中身を示す語がない)ので、別の仕組みとして持つ。LLM は使わず、文の形から取り出す。

  取り出す : Claude の返答と依頼から、期限のある文・保留や未対応の文・人への連絡を頼む文
  完了     : 後のやり取りで、同じ件(概念が重なる)について「完了」「直しました」などが出てきたら完了にする
             (思い出したときに記憶が書き換わる、再固定化にあたる)
  思い出す : 新しい会話の最初と、「やり残し」「次なにする」「期限」のような依頼のときに、
             期限の近さ・依頼との関係・新しさで並べる
"""
import re
from datetime import date, datetime, timedelta

# 期限: 「10月12日までに」「10/15 まで」「期限は10月20日」など
DUE = re.compile(r"(?:(\d{4})[年/-])?(\d{1,2})(?:月|/)(\d{1,2})日?(?:\([^)]*\))?\s*(?:まで|迄|が期限|が締め切り|締め切り|〆)"
                 r"|(?:期限|締め切り|〆切)\s*(?:は|:|：)?\s*(?:(\d{4})[年/-])?(\d{1,2})(?:月|/)(\d{1,2})")
# 保留・未対応・あとでやること
# 「残っている」は「振幅が残っている」のように、やり残し以外の意味でもよく使われるので条件にしない
# 「宿題」は研修の受講生の宿題、「未検証」は性質の説明でもよく使われるので条件にしない
PENDING = re.compile(r"未対応|未完了|未着手|保留(?:に|の|中|して|します)|後回し|残タスク|やり残し|持ち越し|"
                     r"返事待ち|確認待ち|回答待ち|返信待ち|TODO|あとで(?:やる|対応|確認)|次回(?:やる|対応|確認)|"
                     r"まだ(?:終わって|できて|対応して|反映して|手を付けて|確認して)(?:い)?ません")
SUMMARY = re.compile(r"(?:決定|ToDo|保留)\s*\d+件")   # 議事録の件数のまとめ(「決定9件・保留14件」)は、やることそのものではない
# 人への連絡・確認を頼む文(誰に、が書かれているものだけ。「画面を確認してください」のような軽いものは除く)
ASK_PERSON = re.compile(r"[一-龥ぁ-んァ-ンA-Za-z]{1,8}(?:さん|課長|部長|係長|主任|先生|様)(?:に|へ)[^。]{0,40}"
                        r"(?:確認|連絡|相談|打診|依頼|報告|共有|提出|送付|聞いて|伝えて)")
# 完了を表す言い方
DONE = re.compile(r"完了|終わりました|終わった|直しました|直した|対応しました|対応済み|済みです|済んだ|反映しました|"
                  r"提出しました|送りました|送付しました|片付き|解決しました|できました|コミットしました|取り込みました")
# 「やり残し」「次なにする」「期限」のような、これからやることを尋ねる依頼
ASKS_TODO = re.compile(r"やり残|残って(?:る|い)|次(?:は|、)?(?:何|なに)|何(?:を)?(?:したら|すれば)|やること|やるべき|タスク|"
                       r"期限|締め切り|〆切|忘れて(?:る|い)|TODO|予定")
SENTENCE = re.compile(r"[^。\n!?！？]+[。!?！？]?")
MAX_PER_TURN = 2       # 1回のやり取りから取り出す数(多すぎると、どれも大事でなくなる)
MAX_CHARS = 150


def clean(text):
    text = re.sub(r"```.*?```", " ", text, flags=re.S)         # コードの部分は除く
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\*\*|__|^#+\s*|^\s*[-*]\s+|^\s*\d+\.\s+", "", text, flags=re.M)
    return text


def due_date(m, base):
    """期限の文字列から日付を作る(年がなければ、そのやり取りの日付から見て近い方の年)"""
    year, month, day = (m.group(1), m.group(2), m.group(3)) if m.group(2) else (m.group(4), m.group(5), m.group(6))
    try:
        d = date(int(year) if year else base.year, int(month), int(day))
    except ValueError:
        return None
    if not year and d < base - timedelta(days=60):   # 1月に「12月までに」とは言わないので、来年の話とみなす
        d = date(d.year + 1, d.month, d.day)
    return d


def extract(prompt, replies, when):
    """やり取りから、これからやることを取り出す。[{"text", "due", "kind"}]"""
    base = when.date() if isinstance(when, datetime) else when
    found = []
    for source, text in [("依頼", prompt)] + [("返答", r) for r in replies]:
        for s in SENTENCE.findall(clean(text)):
            s = " ".join(s.split())
            if len(s) < 8 or s.endswith(("?", "？")) or "ますか" in s:    # 質問や提案(〜しますか)は、やることではない
                continue
            m = DUE.search(s)
            if m:
                d = due_date(m, base)
                if d and d >= base - timedelta(days=1):                      # 過去の日付は、やることではない
                    found.append({"text": s[:MAX_CHARS], "due": d.isoformat(), "kind": "期限", "source": source})
                    continue
            if source == "返答" and PENDING.search(s) and not SUMMARY.search(s):
                found.append({"text": s[:MAX_CHARS], "due": None, "kind": "保留", "source": source})
            elif source == "返答" and ASK_PERSON.search(s):
                found.append({"text": s[:MAX_CHARS], "due": None, "kind": "連絡", "source": source})
    order = {"期限": 0, "保留": 1, "連絡": 2}
    unique = list({f["text"]: f for f in found}.values())
    return sorted(unique, key=lambda f: order[f["kind"]])[:MAX_PER_TURN]


def mentions_done(text):
    return bool(DONE.search(text))


def asks_todo(text):
    return bool(ASKS_TODO.search(text))
