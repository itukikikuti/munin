"""脳の記憶モデル(最小版)

  皮質   : 意味ネットワーク。知識は概念ノード間のリンクに断片として散らばる
  海馬   : 新しい出来事(エピソード)を一発で記録する。速く覚え、速く忘れる
  睡眠   : 海馬のエピソードを再生し、皮質のリンクを少しずつ強める(固定化)
  想起   : 手がかり → 海馬のパターン補完 + 皮質の活性化拡散 → 断片から再構成
  強化   : 思い出したリンクは使用履歴が増え、取り出しやすくなる
  忘却   : 使わないリンクは取り出しにくくなる(ACT-R の基本活性化の式)

時間は実時間ではなく、内部時計(時間単位)で進める。
"""
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from itertools import combinations

from embed import Embedder
from intentions import asks_todo

# ---- パラメータ ----
DECAY_D = 0.5                # 忘却の速さ(ACT-R の d)
ACCESS_TAU = -2.5            # 取り出しやすさの閾値
ACCESS_S = 0.5               # 閾値付近のなだらかさ
HIPPO_HALFLIFE = 48          # 海馬の記録の半減期(時間)
HIPPO_FORGET_BELOW = 0.05    # これ以下に薄れた海馬の記録は消える
NIGHTS_TO_INDEPENDENCE = 3   # 何晩再生されたら海馬なしで思い出せるか
REPLAYS_PER_NIGHT = 3
CONSOLIDATE_LR = 0.15        # 1回の再生で皮質リンクが強まる割合
INTERLEAVE_LR = 0.05         # 古い記憶を混ぜて再生するときの強化
DOWNSCALE = 0.97             # 睡眠中のシナプス整理(全体を少し弱める)
PRUNE_BELOW = 0.02           # これ以下のリンクは消える
SLEEP_HOURS = 8
ASSOC_FREQ_WEIGHT = 1.0      # 連合強度で「よく一緒に出てくるか」を「特に結びついているか」に対してどれだけ重視するか
CONTEXT_MIN = 0.3            # 指示が今の話題を指していないときの、ワーキングメモリの強さ(プライミング用)
PRIMING = 0.2                # 指示が今の話題を指していないとき、今の話題に近い記憶の活性を最大で何割増しにするか
GIST_SHARE = 0.5             # 手がかりに向ける注意のうち、出来事の要旨による照合に向ける割合
GIST_TOPK = 10               # 呼び起こすエピソード記憶の痕跡の数
GIST_TEMP = 0.03             # 意味の近さの差を、呼び起こしの強さの差にどれだけはっきり反映するか(小さいほど1位が独占)
GIST_DECAY = 0.2             # 古い痕跡ほど呼び起こされにくくする度合い(経過時間^-d)
GIST_Z_MIN = 1.5             # 一番近い痕跡が、他より何σ高ければ呼び起こし始めるか
GIST_Z_FULL = 3.5
GIST_ABS_MIN = 0.70          # 痕跡との類似度がこれ以下なら、呼び起こさない(無関係な依頼でも 0.7 程度にはなる)
GIST_ABS_FULL = 0.85         # 類似度がこれ以上なら、十分に近い
# 関連することを思い出す(出来事の活性 A = 要旨 + 概念 + 新しさ、確かさは ACT-R の検索確率)
EP_W_GIST = 1.0              # 要旨の意味の近さの重み
EP_W_CONCEPT = 1.0           # 出来事に含まれる概念の活性の重み
EP_W_BASE = 0.05             # 新しさ(基本活性化)の重み
EP_DECAY = 0.5               # 新しさの減り方(ACT-R の d)
EP_TAU = 1.55                # 出来事の検索閾値(自動評価の当たり外れから推定。evaluate_memory.py)
EP_S = 1.26                  # 出来事の検索確率の、閾値付近でのなだらかさ(同上)
CHOICE_T = 0.03              # 場所の選択規則の温度(ACT-R の t)。小さいほど1位が選ばれやすい
SHOW_CONF = 0.4              # この確かさ以上のものだけ差し込む(確かさ = 実際に関係している確率の目安)
PLACE_SHOW_CONF = 0.2
EPISODES_SHOWN = 5
ABSTAIN = False              # 確かな出来事を1つも思い出せないとき(会話の最初を除く)、語や場所も差し込まない。
                             # SHOW_CONF を上げて差し込みを短くするときに合わせて使う(0.5 と組み合わせると、記憶が要らないやり取りの半分で何も入れない)
CANDIDATES_KEPT = 50        # 評価用に残す、並べた出来事の上位の数(候補の段階での取りこぼしを測る)
EP_W_PLACE = 0.5             # 出来事が、今の作業フォルダ(またはそれを含むプロジェクト)で起きたかの重み
EP_W_KEYWORD = 0.5          # 出来事の記録に、依頼の語が文字どおり含まれるかの重み
KEYWORD_TERMS = 12           # キーワード検索に使う、依頼の語の数
EP_W_DETAIL = 1.0            # 出来事の記録の中に、依頼とよく合う部分があるかの重み
CHUNK_SIM_MIN = 0.45         # 記録の部分(チャンク)と依頼の類似度がこれ以下なら、関係なしとみなす
CHUNK_SIM_FULL = 0.70        # 類似度がこれ以上なら、十分に関係がある
DETAILS_SHOWN = 3            # 思い出した出来事ごとに、差し込む部分の数
PROMPT_HINT_CHARS = 60       # 差し込む文章に載せる、依頼の長さ
DETAIL_HINT_CHARS = 120      # 差し込む文章に載せる、関係する部分の長さ
HINTS_SHOWN = 6              # 思い出した出来事ごとに載せる、手がかりの語の数
# 展望記憶(これからやること)
DONE_OVERLAP = 3             # 後のやり取りと、何個の概念が重なれば同じ件とみなすか(完了の判定)
INTENT_OVERDUE_DAYS = 7      # 期限をこの日数より過ぎたものは、もう追わない
INTENT_FORGET_DAYS = 30      # 期限のないものは、この日数が経つと思い出さない
INTENT_RECENT_DAYS = 7       # 新しい会話の最初に出す、期限のないもの(最近の保留・連絡)の新しさ
INTENT_W_REL = 1.0           # 依頼との関係の重み
INTENT_W_URGENCY = 1.0       # 期限の近さの重み
INTENT_W_RECENT = 0.5        # 新しさの重み
INTENT_KIND_BONUS = {"期限": 0.3, "連絡": 0.15, "保留": 0.0}   # 取り出しの精度が高い種類ほど前に出す
INTENTS_SHOWN = 3            # 新しい会話の最初に出す数
INTENTS_ASKED = 6            # これからやることを尋ねられたときに出す数
TODO_CHARS = 100
STALE_PLACE = 0.2            # 今はもうない場所の活性を何倍にするか
SHOWN_PER_SESSION = 200      # 会話ごとに覚えておく、差し込んだ記憶の鍵の数
SHOWN_SESSIONS = 50          # 差し込んだ記憶の鍵を覚えておく会話の数(古い会話の分は忘れる)
SHORT_ID = 8                 # 記録の鍵の、差し込む文章に載せる長さ(先頭から)
CONCEPTS_SHOWN = 15
PLACES_SHOWN = 3
RECENT_SHOWN = 3
DEICTIC_CONTEXT = 0.6        # 指示語(「この」「それ」など)があるとき、ワーキングメモリに向ける注意
OMITTED_CONTEXT = 0.8        # 対象が省かれている(「直して」だけ)とき、ワーキングメモリに向ける注意
ZIPF_COMMON = 5.5            # この頻度(Zipf)以上の日常語は、手がかりとしてほぼ役に立たない
ZIPF_RARE = 2.5              # この頻度以下の語は、手がかりとして十分に役立つ
INFO_FLOOR = 0.1             # 日常語でも、手がかりとしての強さはこれ以上残す
CONTEXT_HALFLIFE = 1         # ワーキングメモリの半減期(時間)
WM_CAPACITY = 5              # ワーキングメモリに保てる概念の数(人間は 4±1 程度)
FOCUS_DECAY = 0.3            # 注意の焦点から1つ前の話題に戻るごとに、強さが何倍になるか
STRONG_CUE = 0.5             # これ以上の手がかりだけを、強化と文脈の対象にする
RECALL_LR = 0.05             # 想起による強化
EXACT_USES = 3               # 取り出しやすさの計算で、時刻を正確に覚えておく直近の使用回数
RECALL_THRESHOLD = 0.02       # これ未満の活性は、何も思い浮かばないのと同じ
RETRIEVAL_TAU = 0.05          # 検索閾値(ACT-R の τ)。活性がこれを超えると、思い出せる確率が5割を超える
RETRIEVAL_S = 0.015           # 検索確率の、閾値付近でのなだらかさ(ACT-R の s)
AMBIGUITY_RATIO = 0.8         # 2位がこの比率以上なら「迷う」
INNER_PLACE_RATIO = 0.2       # 中のフォルダの活性が親のこの比率以上(かつ検索閾値以上)なら、中のフォルダを答えにする
SEM_TOPK = 5                 # 意味が近い概念を最大いくつ手がかりにするか
SEM_MAX = 0.8                # 意味が近いだけの手がかりの最大強度(名前の一致は 1.0)
SEM_Z_MIN = 2.0              # 他の概念より何σ高ければ手がかりにし始めるか
SEM_Z_FULL = 4.0             # 何σ高ければ最大強度にするか
SEM_WITH_EXACT = 0.5         # 名前の一致があるときの、連想による手がかりの割合
SEM_ABS_MIN = 0.5            # 類似度がこれ以下なら、どれだけ際立っていても手がかりにしない
SEM_ABS_FULL = 0.65          # 類似度がこれ以上なら、絶対値の面では十分に近い

# 仕組みを1つずつ外して、それぞれの効果を切り分けるためのスイッチ(評価で使う)
ABLATIONS = {"sleep": "睡眠による固定化", "spread": "皮質の活性化拡散", "hippo": "海馬のパターン補完",
             "gist": "出来事の要旨の照合", "concept": "出来事の概念の活性", "recency": "新しさ",
             "detail": "記録の細部の照合", "context": "ワーキングメモリ(文脈)", "keyword": "キーワード検索"}
ABLATE = set()

PLACE_RE = re.compile(r"^([A-Za-z]:[\\/]|/|~[\\/])")


def is_place(name):
    return bool(PLACE_RE.match(name))


_tokenizer = None


def word_boundaries(text):
    """文を語に区切ったときの、語の境目の位置(形態素解析。使えなければ全位置を境目とみなす)"""
    global _tokenizer
    try:
        if _tokenizer is None:
            from janome.tokenizer import Tokenizer
            _tokenizer = Tokenizer()
    except ImportError:
        return set(range(len(text) + 1))
    bounds, pos = {0}, 0
    for tok in _tokenizer.tokenize(text, wakati=True):
        pos += len(tok)
        bounds.add(pos)
    return bounds


def informativeness(word):
    """手がかりとしての情報量(0〜1)

    日常的によく使う語(「設定」「アプリ」)は、どの記憶を指しているかの手がかりにならない。
    日本語での使用頻度(Zipf 尺度。1〜7、大きいほどよく使う)が低い語ほど強い手がかりになる。
    """
    if is_place(word):
        return 1.0
    try:
        from wordfreq import zipf_frequency
        z = zipf_frequency(word, "ja")
    except Exception:
        return 1.0
    return min(1.0, max(INFO_FLOOR, (ZIPF_COMMON - z) / (ZIPF_COMMON - ZIPF_RARE)))


DEICTIC = re.compile(r"この|その|あの|これ|それ|あれ|ここ|そこ|あそこ|こっち|そっち|さっき|例の|今の")


def content_words(text):
    """文の中の内容語(名詞。指示語などは除く)"""
    global _tokenizer
    try:
        if _tokenizer is None:
            from janome.tokenizer import Tokenizer
            _tokenizer = Tokenizer()
    except ImportError:
        return [text]
    words = []
    for tok in _tokenizer.tokenize(text):
        pos = tok.part_of_speech.split(",")
        if pos[0] == "名詞" and pos[1] not in ("代名詞", "非自立", "接尾", "数") and not DEICTIC.fullmatch(tok.surface):
            words.append(tok.surface)
    return words


def retrieval_probability(a, tau, s):
    """検索確率(ACT-R): 活性 a が閾値 tau を超えるほど、確かに思い出せる"""
    return 1 / (1 + math.exp(-(a - tau) / s))


def recollection_text(r):
    """思い出したことを、依頼に差し込む文章にする

    中身を全部渡すのではなく、記憶の手がかり(いつ・どこで・何の件か・関係しそうな語や部分)と、
    詳しく見るための鍵だけを渡す。相手(Claude)は、手がかりをもとに自分で調べたり、鍵で記録を開いたりできる。
    """
    def head(e):
        when = e["when"][5:].replace("-", "/") if e["when"] else ""
        return f"[{e['id']}] {when} {' '.join(e['places'])} 「{clip(e['prompt'], PROMPT_HINT_CHARS)}」".replace("  ", " ")

    lines = []
    if r.get("todos"):
        lines.append("未完了のこと(期限・保留・連絡):")
        for it in r["todos"]:
            due = f"{it['due'][5:].replace('-', '/')}まで " if it["due"] else ""
            lines.append(f"- [{it['id']}] {due}{' '.join(it['places'])} 「{clip(it['text'], TODO_CHARS)}」".replace("  ", " "))
    if r["recent"]:
        lines.append("直前の作業:")
        lines += [f"- {head(e)}" for e in r["recent"]]
    if r["episodes"]:
        lines.append("関係しそうな記憶(確かさ順):")
        for e in r["episodes"]:
            lines.append(f"- {head(e)} 確かさ{e['conf']:.2f}")
            hints = []
            if e["hints"]:
                hints.append("手がかり: " + ", ".join(e["hints"]))
            if e["details"]:  # そのやり取りの中で、今の依頼と最も関係する部分
                d = e["details"][0]
                hints.append(f"{d['kind']}「{clip(d['text'], DETAIL_HINT_CHARS)}」")
            if hints:
                lines.append("  " + " / ".join(hints))
    if r["places"]:
        lines.append("関係しそうな場所: " + ", ".join(p["name"] + ("(今はない)" if p.get("stale") else "")
                                                   for p in r["places"]))
    if r["concepts"]:
        lines.append("関連する語・ファイル: " + ", ".join(c["name"] for c in r["concepts"]))
    if not lines:
        return ""
    lines.append("([ ] は記録の鍵。詳しい中身は "
                 f"python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'memory_detail.py')} 鍵 で開ける)")
    return "<思い出したこと>\n" + "\n".join(lines) + "\n</思い出したこと>"


def clip(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def is_inside(inner, outer):
    """inner が outer の中のフォルダか"""
    i, o = inner.lower().rstrip("\\/"), outer.lower().rstrip("\\/")
    return i != o and (i.startswith(o + "\\") or i.startswith(o + "/"))


def at_place(cwd, place):
    """今の作業フォルダが、その場所そのものか、その中か(その場所で作業している)"""
    return cwd.lower().rstrip("\\/") == place.lower().rstrip("\\/") or is_inside(cwd, place)


def link_key(a, b):
    return "\t".join(sorted((a, b)))


class Brain:
    def __init__(self, path, embedder="auto", store="auto"):
        """store: やり取りの記録(全文)の保存先。痕跡はその記録への索引を持つ"""
        self.path = path
        here = os.path.dirname(os.path.abspath(path))
        if embedder == "auto":
            embedder = None
            if os.environ.get("BRAIN_EMBED", "1") != "0":
                embedder = Embedder(os.path.join(here, "embeddings.json"))
        self.embedder = embedder
        if store == "auto":
            from store import Store
            store = Store(os.path.join(here, "records.sqlite")) if embedder else None
        self.store = store
        self.last_cues = {}
        self.check_places = True   # 思い出した場所が今もあるか確かめる(評価では、過去の時点で思い出すので確かめない)
        self._place_cache = {}
        self.shown = {}            # 会話ごとに、すでに差し込んだ記憶の鍵(同じ会話で同じ記憶を繰り返し差し込まない)
        self.now = 0.0
        # link_key -> {"w": 保存強度, "recent": [直近の使用時刻], "n": 使用回数, "first": 最初の使用時刻}
        self.cortex = {}
        self.hippocampus = []  # {"concepts": [...], "t": 記録時刻, "nights": 再生された晩数}
        # エピソード記憶の痕跡: 出来事ごとの要旨(意味のベクトルの鍵)と、含まれていた概念。
        # 海馬から皮質へ移ったあとも残る(多重痕跡理論)。古いものほど呼び起こされにくい
        self.traces = []       # {"gist": 要旨の鍵, "concepts": [...], "t": 記録時刻}
        # 展望記憶: これからやること(期限・保留・連絡)。完了したら status が done になる
        self.intentions = []
        self.context = {}      # 概念 -> [最後に意識した時刻, 注意を向けた順番]
        self.seq = 0           # 注意を向けた回数(時刻が同じでも、どれが新しい話題か分かるように)
        self._rev = 0          # 皮質が変わるたびに増える(つながりの計算を使い回すため)
        self._assoc = None
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            self.now = s["now"]
            self.cortex = s["cortex"]
            for link in self.cortex.values():  # 古い形式(使用時刻の一覧)から変換
                if "uses" in link:
                    uses = link.pop("uses")
                    link.update(recent=uses[-EXACT_USES:], n=len(uses),
                                first=uses[0] if uses else None)
            self.hippocampus = s["hippocampus"]
            self.traces = s.get("traces", [])
            self.intentions = s.get("intentions", [])
            self.seq = s.get("seq", 0)
            self.shown = s.get("shown", {})
            self.context = {c: v if isinstance(v, list) else [v, 0]
                            for c, v in s["context"].items()}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"now": self.now, "seq": self.seq, "cortex": self.cortex,
                       "hippocampus": self.hippocampus, "traces": self.traces, "context": self.context,
                       "shown": self.shown, "intentions": self.intentions},
                      f, ensure_ascii=False, separators=(",", ":"))
        if self.embedder:
            self.embedder.save()

    # ---- 強度の計算 ----

    def use(self, link):
        """リンクを使ったことを記録する(直近の数回は時刻を、それより前は回数だけ残す)"""
        link["recent"] = (link["recent"] + [self.now])[-EXACT_USES:]
        link["n"] += 1
        if link["first"] is None:
            link["first"] = self.now
        self._rev += 1

    def access(self, link):
        """取り出しやすさ(0〜1)。使うほど高く、時間とともに下がる

        ACT-R の基本活性化 B = ln(Σ 経過時間^-d) を、Petrov(2006)のハイブリッド近似で求める。
        直近の数回は正確に足し、それより前の使用は、最初の使用から直近までの間に
        均等に散らばっていたとみなして積分で足す。使用履歴を捨てずに一定の計算量で済む。
        """
        if not link["n"]:
            return 0.0
        d = DECAY_D
        ages = [self.now - t + 1 for t in link["recent"]]
        total = sum(a ** -d for a in ages)
        rest = link["n"] - len(ages)
        if rest > 0:
            t_k = max(ages)                         # 正確に数えた中で一番古い使用
            t_n = self.now - link["first"] + 1      # 最初の使用
            if t_n > t_k:
                total += rest * (t_n ** (1 - d) - t_k ** (1 - d)) / ((1 - d) * (t_n - t_k))
            else:
                total += rest * t_n ** -d
        b = math.log(total)
        return 1 / (1 + math.exp(-(b - ACCESS_TAU) / ACCESS_S))

    def effective(self, link):
        return link["w"] * self.access(link)

    def hippo_strength(self, ep):
        return 0.5 ** ((self.now - ep["t"]) / HIPPO_HALFLIFE)

    def neighbors(self):
        return self.associations()[0]

    def associations(self):
        """皮質のつながり: (隣接, 概念ごとのつながりの強さの合計, 全体の合計)

        時刻と皮質が変わらないうちは、計算し直さずに使い回す。
        """
        key = (self._rev, self.now, len(self.cortex))
        if self._assoc and self._assoc[0] == key:
            return self._assoc[1]
        adj = {}
        for k, link in self.cortex.items():
            a, b = k.split("\t")
            eff = self.effective(link)
            adj.setdefault(a, []).append((b, eff))
            adj.setdefault(b, []).append((a, eff))
        degree = {n: sum(eff for _, eff in links) for n, links in adj.items()}
        result = (adj, degree, sum(degree.values()) or 1.0)
        self._assoc = (key, result)
        return result

    def known_concepts(self):
        names = set(self.context)
        for key in self.cortex:
            names.update(key.split("\t"))
        for ep in self.hippocampus:
            names.update(ep["concepts"])
        return names

    # ---- 覚える(海馬に記録) ----

    def learn(self, concepts, gist=None, note=None):
        """出来事を覚える

        gist: その出来事の要旨を表す文(依頼文など。なければ概念を並べたもの)
        note: 思い出したときに伝える中身(依頼・回答の抜粋、日時、会話の識別子など)
        """
        concepts = list(dict.fromkeys(concepts))
        self.hippocampus.append({"concepts": concepts, "t": self.now, "nights": 0})
        if self.embedder:
            key = self.embedder.remember_gist(gist or " ".join(concepts))
            if key:
                self.traces.append({"gist": key, "concepts": concepts, "t": self.now, **(note or {})})
        self.attend(concepts)

    # ---- 睡眠(固定化) ----

    def sleep(self, hours=SLEEP_HOURS):
        if "sleep" in ABLATE:  # 睡眠を外す: 再生も固定化もせず、海馬の記録は薄れて消えるだけ
            self.hippocampus = [ep for ep in self.hippocampus if self.hippo_strength(ep) >= HIPPO_FORGET_BELOW]
            self.now += hours
            self.context = {}
            return []
        report = []
        touched = set()
        # 睡眠が短いと、再生できる回数も減る
        replays = max(1, round(REPLAYS_PER_NIGHT * hours / SLEEP_HOURS))
        for ep in self.hippocampus:
            strength = self.hippo_strength(ep)
            if strength < HIPPO_FORGET_BELOW:
                continue
            # 海馬の記録を何度も再生し、皮質のリンクを強める
            for a, b in combinations(ep["concepts"], 2):
                key = link_key(a, b)
                link = self.cortex.setdefault(key, {"w": 0.0, "recent": [], "n": 0, "first": None})
                for _ in range(replays):
                    link["w"] += CONSOLIDATE_LR * strength * (1 - link["w"])
                self.use(link)
                touched.add(key)
            ep["nights"] += 1
            report.append(f"再生: {' / '.join(ep['concepts'])} ({ep['nights']}晩目)")

        # 古い記憶も混ぜて再生し、新しい知識で上書きされないようにする
        # 取り出しやすい記憶ほど再生されやすく、使わない記憶は再生されずに薄れていく
        for key, link in self.cortex.items():
            if key not in touched:
                link["w"] += INTERLEAVE_LR * self.access(link) * (1 - link["w"])

        # シナプス整理: 全体を少し弱め、弱すぎるリンクは消す
        for key in list(self.cortex):
            self.cortex[key]["w"] *= DOWNSCALE
            if self.cortex[key]["w"] < PRUNE_BELOW:
                del self.cortex[key]
                report.append(f"消失: {key.replace(chr(9), ' - ')}")

        # 十分に再生された記録・薄れた記録は海馬から離れる
        kept = []
        for ep in self.hippocampus:
            if ep["nights"] >= NIGHTS_TO_INDEPENDENCE:
                report.append(f"皮質へ移行完了: {' / '.join(ep['concepts'])}")
            elif self.hippo_strength(ep) < HIPPO_FORGET_BELOW:
                report.append(f"海馬から忘却: {' / '.join(ep['concepts'])}")
            else:
                kept.append(ep)
        self.hippocampus = kept
        self._rev += 1
        self.now += hours
        self.context = {}  # 起きたら作業中の文脈はリセット
        return report

    def wait(self, hours):
        self.now += hours

    # ---- 思い出す ----

    def perceive(self, text):
        """文から概念を見つける(言語野の代わり)。{概念: 手がかりの強さ} を返す

        1. 名前がそのまま含まれていれば、強さ 1.0
        2. 意味が近い概念も、埋め込みの類似度に応じて弱めに活性化する
           (類似度はその文の中での相対的な高さで判断する。無関係な文では何も選ばない)
        """
        found = {}
        low = text.lower()
        bounds = word_boundaries(text)
        used = [False] * len(low)
        known = self.known_concepts()
        for name in sorted(known, key=len, reverse=True):
            n = name.lower()
            i = low.find(n)
            while i >= 0:
                end = i + len(n)
                # 語の境目にそろっていて、まだ他の名前に使われていない部分だけ一致とみなす
                # (「マネーフォワード」の中の「ワード」は一致にしない)
                if i in bounds and end in bounds and not any(used[i:end]):
                    found[name] = informativeness(name)
                    used[i:end] = [True] * len(n)
                    break
                i = low.find(n, i + 1)

        if self.embedder and known:
            names = sorted(known)
            arr = self.embedder.similarities(text, names) if self.embedder.ensure(names) else None
            if arr is not None:
                mean = float(arr.mean())
                std = float(arr.std()) or 1.0
                top = [(names[i], float(arr[i])) for i in arr.argsort()[::-1][:SEM_TOPK]]
                z_top = (top[0][1] - mean) / std
                # 一番近い概念の強さは、どれだけ際立っているか(相対)と、
                # 類似度そのものの高さ(絶対)の両方で決める
                base = SEM_MAX * min(1.0, max(0.0, (z_top - SEM_Z_MIN) / (SEM_Z_FULL - SEM_Z_MIN)))
                base *= min(1.0, max(0.0, (top[0][1] - SEM_ABS_MIN) / (SEM_ABS_FULL - SEM_ABS_MIN)))
                if found:
                    base *= SEM_WITH_EXACT  # 名前が直接出ているなら、連想は控えめに
                for name, s in top:
                    z = (s - mean) / std
                    if base <= 0 or z <= SEM_Z_MIN:
                        break
                    # 2位以下は、1位との差が開くほど急に弱くなる
                    a = base * ((z - SEM_Z_MIN) / (z_top - SEM_Z_MIN)) ** 2 * informativeness(name)
                    if a > found.get(name, 0):
                        found[name] = a
        return found

    def recall(self, text, use_context=True, reinforce=True):
        """指示から場所を思い出す

        use_context=False: ワーキングメモリ(今の話題)を使わない(記憶そのものの力を測るとき)
        reinforce=False  : 思い出しても記憶を強めない・ワーキングメモリも変えない(試しに思い出すとき)
        """
        cue_strength = self.perceive(text)
        self.last_cues = cue_strength
        cues = sorted(cue_strength, key=lambda c: -cue_strength[c])
        # 注意の分配(ACT-R): 注意の総量 W=1 を、指示の手がかりとワーキングメモリ(今考えていること)で
        # 分け合う。人は、指示語(「この」「それ」)があるときや、「直して」「続けて」のように
        # 何をするかの対象が省かれているときに、今の話題を指していると解釈する
        total = sum(cue_strength.values())
        omitted = not content_words(text)
        referential = omitted or bool(DEICTIC.search(text))
        # 文脈を指していない指示では、文脈は注意を奪わず、思い出しやすさを少し変える(プライミング)だけ
        budget = (OMITTED_CONTEXT if omitted else DEICTIC_CONTEXT) if referential else CONTEXT_MIN
        cue_budget = 1 - budget if referential else 1.0

        # 手がかりに向ける注意を、語による手がかりと、出来事の要旨による照合で分け合う
        episodic, gate = self.recall_traces(text) if "gist" not in ABLATE else ({}, 0.0)
        gist_budget = cue_budget * GIST_SHARE * gate
        word_budget = cue_budget - gist_budget
        scale = word_budget / max(1.0, total) if total else 0.0
        attention = {c: s * scale for c, s in cue_strength.items()}
        reinstated = {c: gist_budget * a for c, a in episodic.items()}
        self.last_episodic = reinstated
        self.last_gate = gate
        wm = self.working_memory() if use_context and "context" not in ABLATE else {}
        context = {c: budget * w / max(1.0, sum(wm.values())) for c, w in wm.items() if c not in attention}

        act_cue, hits = self.activate(attention, reinstated)
        act_ctx, ctx_hits = self.activate(context)
        if referential:
            # 指示が今の話題を指している: 文脈から届いた記憶も、そのまま思い浮かぶ
            act = {n: min(1.0, act_cue.get(n, 0) + act_ctx.get(n, 0)) for n in {*act_cue, *act_ctx}}
            hits += ctx_hits
        else:
            # プライミング: 文脈は、手がかりから届いた記憶のうち今の話題に近いものを思い出しやすくするだけ
            # (天気の話をしているのに、直前の話題のフォルダを答えたりしない)
            peak = max(act_ctx.values(), default=0) or 1.0
            act = {n: min(1.0, a * (1 + PRIMING * act_ctx.get(n, 0) / peak)) for n, a in act_cue.items()}

        adj = self.neighbors()
        fragments = sorted(((n, a) for n, a in act.items()
                            if n not in cues and a >= RECALL_THRESHOLD),
                           key=lambda x: -x[1])
        answer = self.reconstruct(act, adj, cues)
        self.last_activation = act
        if reinforce:
            strong_cues = [c for c in cues if cue_strength[c] >= STRONG_CUE]
            self.reinforce(strong_cues, answer, hits)
        return cues, fragments, answer

    # ---- 展望記憶(これからやること) ----

    def intend(self, items, concepts, ref, when):
        """やり取りから取り出した「これからやること」を覚える"""
        for i, it in enumerate(items):
            self.intentions.append({"id": f"{(ref or '')[:SHORT_ID]}-{i + 1}", "ref": ref, "text": it["text"],
                                    "due": it["due"], "kind": it["kind"], "concepts": concepts, "time": when,
                                    "t": self.now, "status": "open"})

    def settle(self, concepts, done):
        """同じ件(概念が重なる)について完了したと分かったら、その「やること」を完了にする(再固定化)"""
        if not done:
            return
        mine = {c for c in concepts if not is_place(c)}
        for it in self.intentions:
            if it["status"] == "open" and it["t"] < self.now:
                if len(mine & {c for c in it["concepts"] if not is_place(c)}) >= DONE_OVERLAP:
                    it["status"] = "done"
                    it["done_t"] = self.now

    def today(self):
        """内部時計の今を、暦の日付にする(いちばん新しい出来事の日時から数える)"""
        dated = [tr for tr in self.traces if tr.get("time")]
        if not dated:
            return None
        latest = max(dated, key=lambda tr: tr["t"])
        return (datetime.fromisoformat(latest["time"]) + timedelta(hours=self.now - latest["t"])).date()

    def rank_intentions(self, act, asked):
        """未完了の「やること」を、期限の近さ・依頼との関係・新しさで並べる

        asked: 依頼が「やり残し」「次なにする」のように、これからやることを尋ねているか。
        尋ねていないとき(新しい会話の最初)は、期限のあるものと最近のものだけを出す。
        """
        today = self.today()
        open_items = [it for it in self.intentions if it["status"] == "open"]
        rel = [sum(act.get(c, 0) for c in it["concepts"]) for it in open_items]
        top = max(rel, default=0) or 1.0
        ranked = []
        for it, r in zip(open_items, rel):
            age = (self.now - it["t"]) / 24
            urgency = 0.0
            if it["due"] and today:
                days = (date.fromisoformat(it["due"]) - today).days
                if days < -INTENT_OVERDUE_DAYS:       # 期限を大きく過ぎたものは、もう追わない
                    continue
                urgency = 1.0 if days <= 3 else 0.7 if days <= 14 else 0.3
            elif age > INTENT_FORGET_DAYS:            # 期限のないものは、時間が経つと思い出さなくなる
                continue
            if not asked and not it["due"] and age > INTENT_RECENT_DAYS:
                continue
            score = (INTENT_W_REL * r / top + INTENT_W_URGENCY * urgency + INTENT_W_RECENT * (age + 1) ** -0.5
                     + INTENT_KIND_BONUS.get(it["kind"], 0))
            ranked.append((score, it))
        ranked.sort(key=lambda x: -x[0])
        n = INTENTS_ASKED if asked else INTENTS_SHOWN
        return [{"id": it["id"], "ref": it["ref"], "text": it["text"], "due": it["due"], "kind": it["kind"],
                 "places": [c for c in it["concepts"] if is_place(c)][:1]} for _, it in ranked[:n]]

    # ---- 関連することを思い出す(依頼に差し込む記憶) ----

    def recollect(self, text, use_context=True, reinforce=True, session=None, cwd=None):
        """依頼に関連することを思い出す

        記憶力の良い人が、依頼を聞いて関連することを思い出すのと同じように、
          出来事 : 関係のありそうな過去のやり取り(そのときの依頼と回答の抜粋)
          概念   : 関連する語やファイル
          場所   : 関連するフォルダ
          直近   : 新しい会話の最初なら、直前に何をしていたか
        を思い出し、それぞれに確かさ(ACT-R の検索確率)を付ける。text は差し込み用の文章。
        session: 今の会話の識別子。今の会話のやり取りは相手(Claude)がすでに覚えているので、思い出す対象から外す
        cwd    : 今の作業フォルダ。人が「今いる場所」で、その場所の出来事を思い出しやすくなる(文脈依存記憶)のと同じく、
                 同じ場所での出来事を思い出しやすくする
        """
        new_conversation = not self.context
        cues, _, answer = self.recall(text, use_context, reinforce=False)
        act = self.last_activation
        cue_set = set(cues)

        episodes = self.rank_episodes(act, text, exclude_session=session, cwd=cwd)
        concepts = [{"name": n, "conf": retrieval_probability(a, RETRIEVAL_TAU, RETRIEVAL_S)}
                    for n, a in sorted(act.items(), key=lambda x: -x[1])
                    if not is_place(n) and n not in cue_set and a >= RECALL_THRESHOLD][:CONCEPTS_SHOWN]
        places = self.rank_places(act)
        recent = []
        if new_conversation:
            # 直前と同じ: 新しい会話の最初は、直前に何をしていたかを思い出しておく
            others = [tr for tr in self.traces if session is None or tr.get("session") != session]
            recent = [self.episode_view(tr, None) for tr in sorted(others, key=lambda tr: -tr["t"])[:RECENT_SHOWN]]

        shown = [e for e in episodes if e["conf"] >= SHOW_CONF]
        if session and reinforce:
            # 同じ会話ですでに差し込んだ記憶は、相手(Claude)がもう知っているので繰り返さない
            already = set(self.shown.get(session, []))
            shown = [e for e in shown if e["ref"] not in already]
        # 展望記憶: 新しい会話の最初と、これからやることを尋ねられたときに思い出す
        asked = asks_todo(text)
        todos = self.rank_intentions(act, asked) if new_conversation or asked else []
        # 確かな出来事を1つも思い出せないなら、語や場所だけを出しても当て推量になる(何も入れない判断)
        abstain = ABSTAIN and not shown and not new_conversation
        result = {"ranked": episodes[:EPISODES_SHOWN * 2], "todos": todos,
                  "episodes": shown[:EPISODES_SHOWN],
                  "concepts": [] if abstain else [c for c in concepts if c["conf"] >= SHOW_CONF],
                  "places": [] if abstain else [p for p in places if p["conf"] >= PLACE_SHOW_CONF][:PLACES_SHOWN],
                  "recent": recent, "cues": cues, "answer": answer}
        result["text"] = recollection_text(result)
        if reinforce:
            strong_cues = [c for c in cues if self.last_cues[c] >= STRONG_CUE]
            self.reinforce(strong_cues, answer, [])
            for e in result["episodes"]:  # 思い出した出来事は、次も思い出しやすくなる
                tr = e["trace"]
                tr["uses"] = (tr.get("uses", []) + [self.now])[-EXACT_USES:]
            if session:
                refs = self.shown.pop(session, []) + [e["ref"] for e in result["episodes"] + result["recent"] if e["ref"]]
                self.shown[session] = refs[-SHOWN_PER_SESSION:]
                for old in list(self.shown)[:-SHOWN_SESSIONS]:  # 古い会話の分は忘れる
                    del self.shown[old]
        for e in result["ranked"] + result["episodes"] + result["recent"]:
            e.pop("trace", None)
        return result

    def rank_episodes(self, act, text, exclude_session=None, cwd=None):
        """出来事(エピソード記憶の痕跡)を、ACT-R の活性 A = 基本活性化 + 拡散活性化 で並べる

          要旨   : 依頼と出来事の要旨の意味の近さ(他の出来事と比べてどれだけ際立つか)
          概念   : 出来事に含まれる概念が、今どれだけ活性化しているか(語のつながり・文脈を経由)
          新しさ : 基本活性化 ln(Σ 経過時間^-d)。最近の出来事、最近思い出した出来事ほど高い
          細部   : 出来事の記録(返答の段落・コマンドなど)の中に、依頼とよく合う部分があるか
        """
        sim = getattr(self, "last_trace_sim", {})
        traces = [tr for tr in self.traces if exclude_session is None or tr.get("session") != exclude_session]
        refs = {tr["ref"] for tr in traces if tr.get("ref")}
        details = self.store.search(text, refs) if self.store else {}
        # キーワード: 依頼の語が、やり取りの記録に文字どおり含まれるか(型番・ファイル名・エラーのような手がかり)
        keyword = self.store.keyword_search(self.query_terms(text), refs) if self.store and "keyword" not in ABLATE else {}
        top_k = max(keyword.values(), default=0) or 1.0
        concept_score = [sum(act.get(c, 0) for c in tr["concepts"]) for tr in traces]
        top_c = max(concept_score, default=0) or 1.0
        ranked = []
        features = {}
        for tr, cs in zip(traces, concept_score):
            # 依頼との類似度そのもので判断する(無関係な依頼でも、どれかは「他より際立って」近くなるため)
            gist = max(0.0, (sim.get(id(tr), 0.0) - GIST_ABS_MIN) / (GIST_ABS_FULL - GIST_ABS_MIN))
            base = math.log(sum((self.now - u + 1) ** -EP_DECAY for u in [tr["t"]] + tr.get("uses", [])))
            parts = [d for d in details.get(tr.get("ref"), []) if d[0] >= CHUNK_SIM_MIN]
            detail = max(0.0, (parts[0][0] - CHUNK_SIM_MIN) / (CHUNK_SIM_FULL - CHUNK_SIM_MIN)) if parts else 0.0
            on = lambda name: 0.0 if name in ABLATE else 1.0
            here = 1.0 if cwd and any(is_place(c) and at_place(cwd, c) for c in tr["concepts"]) else 0.0
            a = (on("gist") * EP_W_GIST * gist + on("concept") * EP_W_CONCEPT * cs / top_c
                 + on("recency") * EP_W_BASE * base + on("detail") * EP_W_DETAIL * detail
                 + EP_W_KEYWORD * keyword.get(tr.get("ref"), 0.0) / top_k + EP_W_PLACE * here)
            ranked.append((a, tr, parts))
            features[id(tr)] = {"gist": gist, "concept": cs / top_c, "base": base, "detail": detail,
                                "keyword": keyword.get(tr.get("ref"), 0.0) / top_k, "place": here}
        ranked.sort(key=lambda x: -x[0])
        self.last_features = features   # 評価用: 並べ方の重みをデータから推定するため
        self.last_candidates = [tr for _, tr, _ in ranked[:CANDIDATES_KEPT]]   # 評価用: 候補の段階で何が拾えていたか
        out = []
        for a, tr, parts in ranked[:EPISODES_SHOWN * 4]:
            view = self.episode_view(tr, retrieval_probability(a, EP_TAU, EP_S))
            view["activation"] = a
            # 依頼と関係する部分(依頼文そのものは見出しに出すので除く)
            view["details"] = [{"kind": k, "text": t, "sim": s} for s, k, t in parts if k != "依頼"][:DETAILS_SHOWN]
            # 手がかり: その出来事の概念のうち、今思い浮かんでいるもの(場所は別に出すので除く)
            view["hints"] = sorted((c for c in tr["concepts"] if not is_place(c) and act.get(c, 0) > 0),
                                   key=lambda c: -act[c])[:HINTS_SHOWN]
            out.append(view)
        return out

    def query_terms(self, text):
        """キーワード検索に使う語: 依頼の名詞、英数字の語(型番・ファイル名など)、名前が一致した概念"""
        terms = content_words(text) + re.findall(r"[A-Za-z0-9_.\-]{3,}", text)
        terms += [c for c, s in self.last_cues.items() if s >= STRONG_CUE and not is_place(c)]
        return [t for t in dict.fromkeys(terms) if len(t) >= 2 and not DEICTIC.fullmatch(t)][:KEYWORD_TERMS]

    def episode_view(self, tr, conf):
        ref = tr.get("ref") or ""
        places = [c + ("(今はない)" if not self.place_exists(c) else "") for c in tr["concepts"] if is_place(c)][:2]
        return {"trace": tr, "conf": conf, "when": (tr.get("time") or "")[:16].replace("T", " "), "details": [],
                "id": ref[:SHORT_ID], "ref": ref, "hints": [],
                "prompt": tr.get("prompt", " / ".join(tr["concepts"])), "reply": tr.get("reply", ""),
                "places": places, "concepts": tr["concepts"]}

    def place_exists(self, place):
        """その場所(フォルダ)が今もあるか。このPCのパスでなければ、あるとみなす。確かめた結果は1分間使い回す

        別のPCで覚えた場所(このPCにないユーザーのフォルダの中)は、なくなったのではなく確かめられないだけなので、あるとみなす。
        """
        if not self.check_places or not re.match(r"^[A-Za-z]:", place):
            return True
        hit = self._place_cache.get(place)
        if hit is None or time.time() - hit[1] > 60:
            user = re.match(r"(?i)^([a-z]:\\users\\[^\\]+)", place)
            other_pc = bool(user) and not os.path.isdir(user.group(1))
            hit = (other_pc or os.path.isdir(place), time.time())
            self._place_cache[place] = hit
        return hit[0]

    def rank_places(self, act):
        """関連する場所の確かさ = 思い出せる確率 × ほかの場所との競合で選ばれる確率(ACT-R の選択規則)

        競合はプロジェクト単位で考える(あるフォルダと、その中のフォルダは競合しない)。
        """
        # 今はもうない場所(移動・削除された)は、思い出しても役に立たないので弱める(干渉による忘却)
        places = sorted(((n, a if self.place_exists(n) else a * STALE_PLACE) for n, a in act.items()
                         if is_place(n) and a > 0), key=lambda x: -x[1])
        top = {}  # プロジェクト(他のどの場所の中でもない場所) -> 活性
        group = {}
        for n, a in places:
            root = next((r for r in top if is_inside(n, r)), None)
            if root is None:
                top[n] = a
                root = n
            group[n] = root
        z = sum(math.exp(a / CHOICE_T) for a in top.values()) or 1.0
        choice = {r: math.exp(a / CHOICE_T) / z for r, a in top.items()}
        return [{"name": n, "conf": choice[group[n]] * retrieval_probability(a, RETRIEVAL_TAU, RETRIEVAL_S),
                 "stale": not self.place_exists(n)} for n, a in places]

    def recall_traces(self, text):
        """エピソード記憶の痕跡を、指示との意味の近さで呼び起こす

        要旨が指示と近い出来事ほど、また新しい出来事ほど強く呼び起こされ、その出来事に含まれていた
        概念(場所を含む)が思い浮かぶ。どの痕跡も指示と際立って近くなければ、何も呼び起こさない。
        戻り値: ({概念: 呼び起こされた強さ(合計1)}, 呼び起こしの確かさ 0〜1)
        """
        self.last_trace_z = {}
        self.last_trace_sim = {}
        traces = [tr for tr in self.traces if tr["gist"] in self.embedder.index] if self.embedder else []
        if not traces:
            return {}, 0.0
        sims = self.embedder.similarities(text, [tr["gist"] for tr in traces])
        if sims is None:
            return {}, 0.0
        zs = (sims - sims.mean()) / (float(sims.std()) or 1.0)
        self.last_trace_z = {id(tr): float(z) for tr, z in zip(traces, zs)}
        self.last_trace_sim = {id(tr): float(x) for tr, x in zip(traces, sims)}
        order = sims.argsort()[::-1]
        best = float(sims[order[0]])
        z = (best - float(sims.mean())) / (float(sims.std()) or 1.0)
        gate = (min(1.0, max(0.0, (z - GIST_Z_MIN) / (GIST_Z_FULL - GIST_Z_MIN)))
                * min(1.0, max(0.0, (best - GIST_ABS_MIN) / (GIST_ABS_FULL - GIST_ABS_MIN))))
        if gate <= 0:
            return {}, 0.0
        weights = {}
        for k in order[:GIST_TOPK]:
            tr = traces[k]
            recency = (self.now - tr["t"] + 1) ** -GIST_DECAY
            weights[k] = math.exp((float(sims[k]) - best) / GIST_TEMP) * recency
        norm = sum(weights.values())
        out = {}
        for k, w in weights.items():
            for c in traces[k]["concepts"]:
                out[c] = out.get(c, 0) + w / norm
        return out, gate

    def activate(self, senders, reinstated=None):
        """活性の発生源から、海馬のパターン補完と皮質の拡散で、各概念の活性を求める

        reinstated: エピソード記憶の痕跡から、すでに呼び起こされている概念
        """
        # 海馬のパターン補完: 手がかりが一部でも合えば、記録全体が呼び起こされる
        # 多くの記録に共通する語(「アプリ」など)は、どれか1つを呼び起こす力が弱い
        count = {}
        for ep in self.hippocampus:
            for c in ep["concepts"]:
                count[c] = count.get(c, 0) + 1
        # 同じ概念が複数の記録で呼び起こされたら、その分だけ強く思い浮かぶ(足し合わせる)。
        # 手がかりの力は記録の数で割ってあるので、足しても手がかり1つ分を超えない
        sources = dict(senders)
        for c, a in (reinstated or {}).items():
            sources[c] = sources.get(c, 0) + a
        hits = []
        reinstated = {}
        for ep in (self.hippocampus if "hippo" not in ABLATE else []):
            match = min(1.0, sum(senders[c] / count[c] for c in ep["concepts"]
                                 if c in senders))
            if match <= 0:
                continue
            a = match * self.hippo_strength(ep)
            hits.append(ep)
            for c in ep["concepts"]:
                reinstated[c] = reinstated.get(c, 0) + a * 0.9
        for c, a in reinstated.items():
            sources[c] = max(sources.get(c, 0), min(1.0, a))

        # 皮質の活性化拡散(ACT-R): 手がかりから、つながった概念へ1段だけ活性が流れる。
        # 流れる量は連合強度 S_ji = ln(P(i|j) / P(i))(Anderson の合理的分析)で決まる:
        #   P(i|j) … 手がかり j があるときに概念 i が必要になる確率(j のつながりの中での i の割合)
        #   P(i)   … 普段から i が必要になる確率(i のつながりの多さ)
        # つながりの多い手がかりは P(i|j) が小さく(ファン効果)、何にでも出てくる概念は
        # P(i) が大きいので、どちらも強度が下がる。手がかりと特に結びついた概念にだけ活性が届く。
        # ただし比率だけだと、一度たまたま一緒に出てきた珍しい概念が過大評価されるので、
        # 実際によく一緒に出てくるか(P(i|j))でも重み付けする
        adj, degree, total = self.associations()
        scale = math.log(max(2, len(adj)))  # ln(lift) の最大値の目安で 0〜1 にそろえる
        act = dict(sources)
        if "spread" in ABLATE:
            return act, hits
        for j, a in {**senders, **sources}.items():
            if degree.get(j, 0) <= 0:
                continue
            for i, eff in adj[j]:
                if i in sources or i in senders or eff <= 0:
                    continue
                p_ij = eff / degree[j]
                s = p_ij ** ASSOC_FREQ_WEIGHT * math.log(p_ij / (degree[i] / total)) / scale
                if s > 0:
                    act[i] = min(1.0, act.get(i, 0) + a * s)
        return act, hits

    def reconstruct(self, act, adj, cues=()):
        """活性の高い断片から答えを組み立てる"""
        active = {n: a for n, a in act.items() if a > 0}
        places = sorted(((n, a) for n, a in active.items() if is_place(n)), key=lambda x: -x[1])
        if not places or places[0][1] < RECALL_THRESHOLD:
            return {"place": None, "about": [], "confidence": 0, "ambiguous": [], "uncertain": False}
        best, level = places[0]
        # 迷うかどうかは、どのプロジェクト(場所)かの段階で決める
        ambiguous = [n for n, a in places[1:] if not is_inside(n, best) and a >= level * AMBIGUITY_RATIO]
        # 中のフォルダまで具体的に思い出せるなら(検索閾値を超えていれば)、そちらを答えにする
        # (親フォルダの活性は、中のフォルダでの出来事から受け継いだ分を含むので高く出やすい)
        inner = [(n, a) for n, a in places if is_inside(n, best)
                 and a >= max(RETRIEVAL_TAU, level * INNER_PLACE_RATIO)]
        if inner:
            best, level = inner[0]
        # 確信度 = 思い出せる確率 × ほかの場所との競合で選ばれる確率(ACT-R)。2位と僅差なら下がる
        conf = next(p["conf"] for p in self.rank_places(act) if p["name"] == best)

        # 断片どうしのつながり(皮質のリンク + 海馬の記録)
        linked = {n: set() for n in active}
        for a, nbs in adj.items():
            for b, _ in nbs:
                linked.setdefault(a, set()).add(b)
        for ep in self.hippocampus:
            for a, b in combinations(ep["concepts"], 2):
                linked.setdefault(a, set()).add(b)
                linked.setdefault(b, set()).add(a)

        # 「何の場所か」: 指示に含まれていた固有に近い名前を優先し、なければ
        # その場所とつながる概念のうち、他の場所とはつながらない固有のもの
        def specificity(n):
            places_linked = sum(1 for m in linked.get(n, ()) if is_place(m))
            return (places_linked > 2, n not in cues, places_linked, -active[n])
        about = sorted((n for n in linked.get(best, ()) if n in active and not is_place(n)),
                       key=specificity)[:1]
        return {"place": best, "about": about, "confidence": conf, "activation": level,
                "ambiguous": ambiguous, "uncertain": conf < 0.5}

    def reinforce(self, cues, answer, hippo_hits):
        """想起による強化: 実際に使ったつながりが強まる"""
        used = set(cues) | set(answer["about"])
        if answer["place"]:
            used.add(answer["place"])
        for a, b in combinations(used, 2):
            link = self.cortex.get(link_key(a, b))
            if link:
                link["w"] += RECALL_LR * (1 - link["w"])
                self.use(link)
        for ep in hippo_hits:
            if used & set(ep["concepts"]):
                ep["t"] = self.now  # 思い出した海馬の記録は鮮明さを取り戻す
        self.attend(used)

    def working_memory(self):
        """ワーキングメモリの各概念の強さ(0〜1)

        いちばん新しい話題が注意の焦点。それより前の話題は、1つさかのぼるごとに弱まる。
        時間が経つと全体が薄れる。
        """
        order = sorted({seq for _, seq in self.context.values()}, reverse=True)
        return {c: FOCUS_DECAY ** order.index(seq) * 0.5 ** ((self.now - t) / CONTEXT_HALFLIFE)
                for c, (t, seq) in self.context.items()}

    def attend(self, concepts):
        """ワーキングメモリに入れる。容量を超えたら古いものから押し出される"""
        self.seq += 1
        for c in concepts:
            self.context[c] = [self.now, self.seq]
        recent = sorted(self.context.items(), key=lambda x: -x[1][1])[:WM_CAPACITY]
        self.context = dict(recent)

    # ---- 中身を見る ----

    def show(self):
        lines = [f"時刻: {self.now:.0f}時間目", "", "[海馬]"]
        for ep in self.hippocampus:
            lines.append(f"  {' / '.join(ep['concepts'])}  "
                         f"鮮明さ={self.hippo_strength(ep):.2f} 再生={ep['nights']}晩")
        if not self.hippocampus:
            lines.append("  (空)")
        lines.append("")
        lines.append("[皮質]  保存強度 × 取り出しやすさ = 実効強度")
        for key, link in sorted(self.cortex.items(), key=lambda x: -self.effective(x[1])):
            lines.append(f"  {key.replace(chr(9), ' - '):40} "
                         f"{link['w']:.2f} × {self.access(link):.2f} = {self.effective(link):.2f}")
        if not self.cortex:
            lines.append("  (空)")
        return "\n".join(lines)


def format_recall(cues, fragments, answer):
    lines = [f"手がかり: {', '.join(cues) or '(なし)'}"]
    lines.append("活性化: " + ", ".join(f"{n}({a:.2f})" for n, a in fragments[:6]))
    if answer["place"]:
        what = f"{answer['about'][0]} の場所は" if answer["about"] else "場所は"
        line = f"想起: {what} {answer['place']} (確信度 {answer['confidence']:.2f})"
        if answer["ambiguous"]:
            line += f"  ※ {', '.join(answer['ambiguous'])} とも迷う"
        if answer["uncertain"]:
            line += "  ※ うろ覚え"
        lines.append(line)
    else:
        lines.append("想起: 場所は思い出せない")
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    state = os.environ.get("BRAIN_STATE",
                           os.path.join(os.path.dirname(os.path.abspath(__file__)), "brain_state.json"))
    args = sys.argv[1:]
    if not args:
        print("使い方:\n"
              "  python brain.py learn 家計簿アプリ アプリ 自作 C:\\dev\\kakeibo\n"
              "  python brain.py recall \"アプリのここを直して\"\n"
              "  python brain.py sleep [晩数]\n"
              "  python brain.py wait 時間\n"
              "  python brain.py show")
        return
    brain = Brain(state)
    cmd = args[0]
    if cmd == "learn":
        brain.learn(args[1:])
        print(f"海馬に記録: {' / '.join(args[1:])}")
    elif cmd == "recall":
        print(format_recall(*brain.recall(" ".join(args[1:]))))
    elif cmd == "sleep":
        for i in range(int(args[1]) if len(args) > 1 else 1):
            print(f"--- {i + 1}晩目 ---")
            print("\n".join(brain.sleep()) or "(再生なし)")
            brain.wait(16)  # 起きている時間
    elif cmd == "wait":
        brain.wait(float(args[1]))
    elif cmd == "show":
        print(brain.show())
    else:
        print(f"不明なコマンド: {cmd}")
        return
    brain.save()


if __name__ == "__main__":
    main()
