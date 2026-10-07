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
from itertools import combinations

from embed import Embedder

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
CONTEXT_MIN = 0.3            # 指示が今の話題を指していないときに、ワーキングメモリに向ける注意
PRIMING = 0.5                # 今の話題に近い記憶の活性を、最大で何割増しにするか
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
INNER_PLACE_RATIO = 0.6       # 中のフォルダの活性が親のこの比率以上なら、中のフォルダを答えにする
SEM_TOPK = 5                 # 意味が近い概念を最大いくつ手がかりにするか
SEM_MAX = 0.8                # 意味が近いだけの手がかりの最大強度(名前の一致は 1.0)
SEM_Z_MIN = 2.0              # 他の概念より何σ高ければ手がかりにし始めるか
SEM_Z_FULL = 4.0             # 何σ高ければ最大強度にするか
SEM_WITH_EXACT = 0.5         # 名前の一致があるときの、連想による手がかりの割合
SEM_ABS_MIN = 0.5            # 類似度がこれ以下なら、どれだけ際立っていても手がかりにしない
SEM_ABS_FULL = 0.65          # 類似度がこれ以上なら、絶対値の面では十分に近い

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


def is_inside(inner, outer):
    """inner が outer の中のフォルダか"""
    i, o = inner.lower().rstrip("\\/"), outer.lower().rstrip("\\/")
    return i != o and (i.startswith(o + "\\") or i.startswith(o + "/"))


def link_key(a, b):
    return "\t".join(sorted((a, b)))


class Brain:
    def __init__(self, path, embedder="auto"):
        self.path = path
        if embedder == "auto":
            embedder = None
            if os.environ.get("BRAIN_EMBED", "1") != "0":
                cache = os.path.join(os.path.dirname(os.path.abspath(path)), "embeddings.json")
                embedder = Embedder(cache)
        self.embedder = embedder
        self.last_cues = {}
        self.now = 0.0
        # link_key -> {"w": 保存強度, "recent": [直近の使用時刻], "n": 使用回数, "first": 最初の使用時刻}
        self.cortex = {}
        self.hippocampus = []  # {"concepts": [...], "t": 記録時刻, "nights": 再生された晩数}
        self.context = {}      # 概念 -> [最後に意識した時刻, 注意を向けた順番]
        self.seq = 0           # 注意を向けた回数(時刻が同じでも、どれが新しい話題か分かるように)
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
            self.seq = s.get("seq", 0)
            self.context = {c: v if isinstance(v, list) else [v, 0]
                            for c, v in s["context"].items()}

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"now": self.now, "seq": self.seq, "cortex": self.cortex,
                       "hippocampus": self.hippocampus, "context": self.context},
                      f, ensure_ascii=False, separators=(",", ":"))

    # ---- 強度の計算 ----

    def use(self, link):
        """リンクを使ったことを記録する(直近の数回は時刻を、それより前は回数だけ残す)"""
        link["recent"] = (link["recent"] + [self.now])[-EXACT_USES:]
        link["n"] += 1
        if link["first"] is None:
            link["first"] = self.now

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
        adj = {}
        for key, link in self.cortex.items():
            a, b = key.split("\t")
            eff = self.effective(link)
            adj.setdefault(a, []).append((b, eff))
            adj.setdefault(b, []).append((a, eff))
        return adj

    def known_concepts(self):
        names = set(self.context)
        for key in self.cortex:
            names.update(key.split("\t"))
        for ep in self.hippocampus:
            names.update(ep["concepts"])
        return names

    # ---- 覚える(海馬に記録) ----

    def learn(self, concepts):
        concepts = list(dict.fromkeys(concepts))
        self.hippocampus.append({"concepts": concepts, "t": self.now, "nights": 0})
        self.attend(concepts)

    # ---- 睡眠(固定化) ----

    def sleep(self, hours=SLEEP_HOURS):
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
            sims = self.embedder.similarities(text, sorted(known))
            if sims:
                mean = sum(sims.values()) / len(sims)
                std = math.sqrt(sum((s - mean) ** 2 for s in sims.values()) / len(sims)) or 1.0
                top = sorted(sims.items(), key=lambda x: -x[1])[:SEM_TOPK]
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

    def recall(self, text):
        cue_strength = self.perceive(text)
        self.last_cues = cue_strength
        cues = sorted(cue_strength, key=lambda c: -cue_strength[c])
        # 注意の分配(ACT-R): 注意の総量 W=1 を、指示の手がかりとワーキングメモリ(今考えていること)で
        # 分け合う。人は、指示語(「この」「それ」)があるときや、「直して」「続けて」のように
        # 何をするかの対象が省かれているときに、今の話題を指していると解釈する
        total = sum(cue_strength.values())
        omitted = not content_words(text)
        referential = omitted or bool(DEICTIC.search(text))
        budget = (OMITTED_CONTEXT if omitted else DEICTIC_CONTEXT) if referential else CONTEXT_MIN
        scale = (1 - budget) / max(1.0, total) if total else 0.0
        attention = {c: s * scale for c, s in cue_strength.items()}
        wm = self.working_memory()
        context = {c: budget * w / max(1.0, sum(wm.values())) for c, w in wm.items() if c not in attention}

        act_cue, hits = self.activate(attention)
        # 文脈そのものを思い出したことにはせず、そこから結びついた記憶へ活性を送るだけ
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
        strong_cues = [c for c in cues if cue_strength[c] >= STRONG_CUE]
        self.reinforce(strong_cues, answer, hits)
        return cues, fragments, answer

    def activate(self, senders):
        """活性の発生源から、海馬のパターン補完と皮質の拡散で、各概念の活性を求める"""
        # 海馬のパターン補完: 手がかりが一部でも合えば、記録全体が呼び起こされる
        # 多くの記録に共通する語(「アプリ」など)は、どれか1つを呼び起こす力が弱い
        count = {}
        for ep in self.hippocampus:
            for c in ep["concepts"]:
                count[c] = count.get(c, 0) + 1
        # 同じ概念が複数の記録で呼び起こされたら、その分だけ強く思い浮かぶ(足し合わせる)。
        # 手がかりの力は記録の数で割ってあるので、足しても手がかり1つ分を超えない
        sources = dict(senders)
        hits = []
        reinstated = {}
        for ep in self.hippocampus:
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
        adj = self.neighbors()
        degree = {n: sum(eff for _, eff in links) for n, links in adj.items()}
        total = sum(degree.values()) or 1.0
        scale = math.log(max(2, len(adj)))  # ln(lift) の最大値の目安で 0〜1 にそろえる
        act = dict(sources)
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
        # フォルダとその中のフォルダが両方思い浮かんだら、具体的な中のフォルダを答えにする
        # (親フォルダの活性は、中のフォルダでの出来事から受け継いだ分であることが多い)
        inner = [(n, a) for n, a in places if is_inside(n, best) and a >= level * INNER_PLACE_RATIO]
        if inner:
            best, level = inner[0]
        rivals = [(n, a) for n, a in places if n != best and not is_inside(best, n)]
        ambiguous = [n for n, a in rivals if a >= level * AMBIGUITY_RATIO]
        # 検索確率(ACT-R): 活性が検索閾値を超えるほど、確かに思い出せる
        conf = 1 / (1 + math.exp(-(level - RETRIEVAL_TAU) / RETRIEVAL_S))

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
