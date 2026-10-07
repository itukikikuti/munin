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
INTERLEAVE_COUNT = 10
DOWNSCALE = 0.97             # 睡眠中のシナプス整理(全体を少し弱める)
PRUNE_BELOW = 0.02           # これ以下のリンクは消える
SLEEP_HOURS = 8
SPREAD_STEPS = 3
SPREAD_DECAY = 0.6
CONTEXT_WEIGHT = 0.5         # ワーキングメモリ(直前の文脈)の初期活性
CONTEXT_HALFLIFE = 2         # ワーキングメモリの半減期(時間)
RECALL_LR = 0.05             # 想起による強化
MAX_USES = 50
RECALL_THRESHOLD = 0.05
AMBIGUITY_RATIO = 0.8        # 2位がこの比率以上なら「曖昧」

PLACE_RE = re.compile(r"^([A-Za-z]:[\\/]|/|~[\\/])")


def is_place(name):
    return bool(PLACE_RE.match(name))


def link_key(a, b):
    return "\t".join(sorted((a, b)))


class Brain:
    def __init__(self, path):
        self.path = path
        self.now = 0.0
        self.cortex = {}       # link_key -> {"w": 保存強度, "uses": [使用時刻]}
        self.hippocampus = []  # {"concepts": [...], "t": 記録時刻, "nights": 再生された晩数}
        self.context = {}      # 概念 -> 最後に意識した時刻
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
            self.now = s["now"]
            self.cortex = s["cortex"]
            self.hippocampus = s["hippocampus"]
            self.context = s["context"]

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"now": self.now, "cortex": self.cortex,
                       "hippocampus": self.hippocampus, "context": self.context},
                      f, ensure_ascii=False, indent=1)

    # ---- 強度の計算 ----

    def access(self, link):
        """取り出しやすさ(0〜1)。使うほど高く、時間とともに下がる"""
        total = sum((self.now - t + 1) ** -DECAY_D for t in link["uses"])
        if total <= 0:
            return 0.0
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
        for c in concepts:
            self.context[c] = self.now

    # ---- 睡眠(固定化) ----

    def sleep(self):
        report = []
        touched = set()
        for ep in self.hippocampus:
            strength = self.hippo_strength(ep)
            if strength < HIPPO_FORGET_BELOW:
                continue
            # 海馬の記録を何度も再生し、皮質のリンクを強める
            for a, b in combinations(ep["concepts"], 2):
                key = link_key(a, b)
                link = self.cortex.setdefault(key, {"w": 0.0, "uses": []})
                for _ in range(REPLAYS_PER_NIGHT):
                    link["w"] += CONSOLIDATE_LR * strength * (1 - link["w"])
                link["uses"] = (link["uses"] + [self.now])[-MAX_USES:]
                touched.add(key)
            ep["nights"] += 1
            report.append(f"再生: {' / '.join(ep['concepts'])} ({ep['nights']}晩目)")

        # 古い記憶も混ぜて再生し、新しい知識で上書きされないようにする
        old = sorted((k for k in self.cortex if k not in touched),
                     key=lambda k: -self.effective(self.cortex[k]))[:INTERLEAVE_COUNT]
        for key in old:
            link = self.cortex[key]
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
        self.now += SLEEP_HOURS
        self.context = {}  # 起きたら作業中の文脈はリセット
        return report

    def wait(self, hours):
        self.now += hours

    # ---- 思い出す ----

    def perceive(self, text):
        """文から既知の概念を見つける(言語野の代わり。長い名前を優先)"""
        found = []
        rest = text.lower()
        for name in sorted(self.known_concepts(), key=len, reverse=True):
            if name.lower() in rest:
                found.append(name)
                rest = rest.replace(name.lower(), " ")
        return found

    def recall(self, text):
        cues = self.perceive(text)
        sources = {c: 1.0 for c in cues}

        # ワーキングメモリ: 直前まで意識していた概念は少し活性が残っている
        for c, t in self.context.items():
            a = CONTEXT_WEIGHT * 0.5 ** ((self.now - t) / CONTEXT_HALFLIFE)
            sources[c] = max(sources.get(c, 0), a)

        # 海馬のパターン補完: 手がかりが一部でも合えば、記録全体が呼び起こされる
        hippo_hits = []
        for ep in self.hippocampus:
            match = min(1.0, sum(sources.get(c, 0) for c in ep["concepts"]))
            if match <= 0:
                continue
            a = match * self.hippo_strength(ep)
            hippo_hits.append(ep)
            for c in ep["concepts"]:
                sources[c] = max(sources.get(c, 0), a * 0.9)

        # 皮質の活性化拡散(つながりの多い概念ほど、1本あたりに流れる量は減る)
        adj = self.neighbors()
        act = dict(sources)
        for _ in range(SPREAD_STEPS):
            nxt = dict(sources)
            for node, a in act.items():
                links = adj.get(node, [])
                fan = math.sqrt(max(1, len(links)))
                for nb, eff in links:
                    nxt[nb] = nxt.get(nb, 0) + a * eff * SPREAD_DECAY / fan
            act = {n: min(1.0, a) for n, a in nxt.items()}

        fragments = sorted(((n, a) for n, a in act.items()
                            if n not in cues and a >= RECALL_THRESHOLD),
                           key=lambda x: -x[1])
        answer = self.reconstruct(fragments, adj)
        self.reinforce(cues, answer, hippo_hits)
        return cues, fragments, answer

    def reconstruct(self, fragments, adj):
        """活性の高い断片から答えを組み立てる"""
        places = [(n, a) for n, a in fragments if is_place(n)]
        if not places:
            return {"place": None, "about": [f for f, _ in fragments[:3]], "confidence": 0,
                    "ambiguous": []}
        best, conf = places[0]
        ambiguous = [n for n, a in places[1:] if a >= conf * AMBIGUITY_RATIO]
        # その場所と強く結びついている、活性の高い概念を「何の場所か」とする
        frag = dict(fragments)
        about = sorted((nb for nb, _ in adj.get(best, []) if nb in frag),
                       key=lambda n: -frag[n])[:3]
        return {"place": best, "about": about, "confidence": conf, "ambiguous": ambiguous}

    def reinforce(self, cues, answer, hippo_hits):
        """想起による強化: 実際に使ったつながりが強まる"""
        used = set(cues) | set(answer["about"])
        if answer["place"]:
            used.add(answer["place"])
        for a, b in combinations(used, 2):
            link = self.cortex.get(link_key(a, b))
            if link:
                link["w"] += RECALL_LR * (1 - link["w"])
                link["uses"] = (link["uses"] + [self.now])[-MAX_USES:]
        for ep in hippo_hits:
            if used & set(ep["concepts"]):
                ep["t"] = self.now  # 思い出した海馬の記録は鮮明さを取り戻す
        for c in used:
            self.context[c] = self.now

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
        line = f"想起: {' / '.join(answer['about'])} の場所は {answer['place']} (確信度 {answer['confidence']:.2f})"
        if answer["ambiguous"]:
            line += f"  ※ {', '.join(answer['ambiguous'])} とも迷う"
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
