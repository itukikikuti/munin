"""記憶のネットワーク図をサーバー側で描く(地図と同じタイル方式)

全体を 256px 四方の「世界」に配置し、ズーム z では 256×2^z px の大きさになる。
それを 256px のタイルに分け、ブラウザが表示している部分のタイルだけを描いて返す。
ズームが上がるほど、ラベルを付ける概念を増やす(地図で拡大すると細かい地名が出るのと同じ)。
"""
import io
import json
import math
import os
import threading
from collections import OrderedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brain import is_place

TILE = 256
WORLD = 256.0           # ズーム0での全体の大きさ(px)
MAX_ZOOM = 7
SUPERSAMPLE = 2         # 2倍の大きさで描いて縮小し、線や文字をなめらかにする
CACHE_TILES = 3000
FONT = r"C:\Windows\Fonts\YuGothM.ttc"
FONT_BOLD = r"C:\Windows\Fonts\YuGothB.ttc"
# 線は半透明(不透明度 0.3)にして、重なるほど濃くなるようにする。濃淡で線の密集具合が分かる。
# 半透明を積み重ねるには下地が要るので、タイルは背景色で塗ってから描く
# ただし線が数万本あると、どのズームでも1ピクセルに何本も重なって真っ黒になるので、
# 混んでいるところでも真っ黒にならないよう、ズームごとに不透明度を下げる(上限 0.3)
LINE_ALPHA_MAX = 0.3
DENSE_DARKNESS = 0.85
LABEL_MARGIN_PX = 160   # タイルの外にある点のラベルがはみ出してくる分も描く(タイルの境目でラベルが切れないように)

THEMES = {
    "light": {"bg": (246, 245, 241), "line": (0, 0, 0), "hippo": (30, 100, 170), "ink": (29, 29, 27),
              "node": (255, 255, 255), "edge": (90, 88, 80), "place": (47, 125, 91), "place_bg": (224, 240, 231),
              "halo": (246, 245, 241)},
    "dark": {"bg": (22, 22, 20), "line": (255, 255, 255), "hippo": (120, 180, 235), "ink": (236, 235, 230),
             "node": (42, 41, 37), "edge": (160, 157, 145), "place": (95, 193, 147), "place_bg": (23, 56, 42),
             "halo": (22, 22, 20)},
}


def text_px(z):
    """ズーム z でのラベルの文字の大きさ(px)。拡大するほど少しずつ大きくする"""
    return round(10 + 1.2 * z)


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


class GraphMap:
    def __init__(self, layout_path):
        self.layout_path = layout_path
        self.lock = threading.Lock()
        self.version = None
        self.cache = OrderedDict()
        self.pos = {}        # id -> [x, y](配置計算の座標。ファイルに保存して次回も使う)
        if os.path.exists(layout_path):
            with open(layout_path, encoding="utf-8") as f:
                self.pos = json.load(f)
        self.fonts = {}

    # ---- 図の中身と配置 ----

    def update(self, brain, version):
        """記憶が変わっていたら、図の中身と配置を作り直す"""
        with self.lock:
            if version == self.version:
                return
            ids, kinds, labels, index = [], [], [], {}

            def add(nid, kind, label):
                if nid not in index:
                    index[nid] = len(ids)
                    ids.append(nid)
                    kinds.append(kind)
                    labels.append(label)
                return index[nid]

            concept = lambda n: add(n, "place" if is_place(n) else "concept", n)
            edges = []  # (a, b, 強さ, 種類 0=皮質 1=海馬)
            for key, link in brain.cortex.items():
                a, b = key.split("\t")
                edges.append((concept(a), concept(b), brain.effective(link), 0))
            self.episodes = {}
            for ep in brain.hippocampus:
                eid = f"ep:{ep['t']:.6f}:{'|'.join(ep['concepts'])}"
                h = add(eid, "episode", " / ".join(ep["concepts"]))
                self.episodes[h] = ep
                for c in ep["concepts"]:
                    edges.append((h, concept(c), brain.hippo_strength(ep), 1))

            self.ids, self.kinds, self.labels, self.index = ids, kinds, labels, index
            self.edges = np.array(edges, dtype=np.float64).reshape(-1, 4)
            n = len(ids)
            self.degree = np.zeros(n)
            if len(edges):
                np.add.at(self.degree, self.edges[:, 0].astype(int), 1)
                np.add.at(self.degree, self.edges[:, 1].astype(int), 1)
            self._layout()
            self._to_world()
            self._label_order()
            self.version = version
            self._density = None
            self.cache.clear()

    def _layout(self):
        """力学モデルで配置する(すでに配置済みの点はそのまま使い、新しい点だけ近くに置いて整える)"""
        n = len(self.ids)
        if n == 0:
            return
        X = np.zeros((n, 2))
        known = np.zeros(n, bool)
        for i, nid in enumerate(self.ids):
            if nid in self.pos:
                X[i] = self.pos[nid]
                known[i] = True
        a, b = self.edges[:, 0].astype(int), self.edges[:, 1].astype(int)
        rng = np.random.default_rng(0)
        spread = math.sqrt(n) * 40
        for i in np.where(~known)[0]:
            nb = np.concatenate([b[a == i], a[b == i]])
            nb = nb[known[nb]]
            X[i] = (X[nb[0]] if len(nb) else 0) + rng.uniform(-1, 1, 2) * (80 if len(nb) else spread)
        fresh = (~known).sum()
        if fresh == 0:
            return
        iters = 300 if fresh > n * 0.2 else 80
        w = self.edges[:, 2]
        length = np.where(self.edges[:, 3] == 1, 60.0, 110.0)
        chunk = max(1, int(2e7 // n))
        X = X.astype(np.float32)
        for it in range(iters):
            t = 1 - it / iters
            F = np.zeros_like(X)
            x, y = X[:, 0], X[:, 1]
            for s in range(0, n, chunk):  # 反発(全ての点の組)。大きいときはメモリに収まるよう分けて計算する
                dx = x[s:s + chunk, None] - x[None, :]
                dy = y[s:s + chunk, None] - y[None, :]
                d2 = dx * dx + dy * dy + np.float32(0.01)
                k = np.float32(6000) / (d2 * np.sqrt(d2))
                F[s:s + chunk, 0] += (dx * k).sum(1)
                F[s:s + chunk, 1] += (dy * k).sum(1)
            if len(a):  # ばね(つながっている点の組)
                d = X[b] - X[a]
                dist = np.sqrt((d ** 2).sum(1)) + 0.01
                f = (d * ((dist - length) * 0.03 * (0.3 + w) / dist)[:, None]).astype(np.float32)
                np.add.at(F, a, f)
                np.add.at(F, b, -f)
            F -= X * np.float32(0.004)  # 中心へ
            m = np.sqrt((F ** 2).sum(1))
            lim = 25 * t + 1
            F *= np.minimum(1, lim / np.maximum(m, 1e-9))[:, None].astype(np.float32)
            X += F
        self.pos = {nid: X[i].tolist() for i, nid in enumerate(self.ids)}
        with open(self.layout_path, "w", encoding="utf-8") as f:
            json.dump(self.pos, f)

    def _to_world(self):
        """配置座標を 0〜256 の世界座標にそろえる(縦横比は保つ)"""
        n = len(self.ids)
        if n == 0:
            self.world = np.zeros((0, 2))
            return
        X = np.array([self.pos[nid] for nid in self.ids])
        lo, hi = X.min(0), X.max(0)
        span = max((hi - lo).max(), 1e-9)
        pad = WORLD * 0.04
        self.world = (X - lo) / span * (WORLD - 2 * pad) + pad
        self.world += ((WORLD - 2 * pad) - (hi - lo) / span * (WORLD - 2 * pad)) / 2  # 中央に寄せる

    def _label_order(self):
        """ラベルを付ける優先順位(場所 → つながりの多い概念)"""
        score = self.degree.copy()
        kinds = np.array(self.kinds)
        score[kinds == "place"] += 1e6
        score[kinds == "episode"] = -1
        self.label_priority = np.argsort(-score)
        self.label_rank = np.empty(len(score), int)
        self.label_rank[self.label_priority] = np.arange(len(score))
        self.label_sets = {}

    def label_text(self, i):
        text = self.labels[i]
        if self.kinds[i] == "place":
            text = text if len(text) <= 34 else "…" + text[-33:]
            text = text.replace("\\", "＼")  # 日本語フォントは \ を ¥ の形で描くので、全角の＼にする
        return text

    def label_box(self, i, z):
        """ズーム z の画面座標(px)でのラベルの範囲"""
        S = SUPERSAMPLE
        kind = self.kinds[i]
        font = self.font(text_px(z) * S, bold=kind == "place")
        l, t, r, b = font.getbbox(self.label_text(i))
        w, h = (r - l) / S, (b - t) / S
        cx, cy = self.world[i] * 2 ** z
        if kind == "place":
            return cx - w / 2 - 6, cy - h / 2 - 4, cx + w / 2 + 6, cy + h / 2 + 4
        r0 = 2 + min(5, math.sqrt(self.degree[i]))
        return cx - w / 2, cy + r0 + 2, cx + w / 2, cy + r0 + 4 + h

    def labels_at(self, z):
        """ズーム z でラベルを出す点(重要なものから順に置き、既に置いたラベルと重なるものは出さない)

        画面全体の座標で決めるので、タイルの境目でラベルの有無が食い違わない(地図のラベル配置と同じ)。
        """
        if z in self.label_sets:
            return self.label_sets[z]
        cell = 64
        grid = {}
        shown = set()
        for i in self.label_priority:
            if self.kinds[i] == "episode":
                break
            box = self.label_box(i, z)
            cells = [(cx, cy) for cx in range(int(box[0] // cell), int(box[2] // cell) + 1)
                     for cy in range(int(box[1] // cell), int(box[3] // cell) + 1)]
            if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3])
                   for c in cells for o in grid.get(c, ())):
                continue
            for c in cells:
                grid.setdefault(c, []).append(box)
            shown.add(int(i))
        self.label_sets[z] = shown
        return shown

    # ---- タイル ----

    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = _font(FONT_BOLD if bold else FONT, size)
        return self.fonts[key]

    def tile(self, z, x, y, theme="light"):
        key = (self.version, theme, z, x, y)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        png = self._render(z, x, y, THEMES.get(theme, THEMES["light"]))
        with self.lock:
            self.cache[key] = png
            while len(self.cache) > CACHE_TILES:
                self.cache.popitem(last=False)
        return png

    def _render(self, z, tx, ty, th):
        S = SUPERSAMPLE
        scale = 2 ** z                      # 世界座標 → px
        size = TILE * S
        img = Image.new("RGB", (size, size), th["bg"])  # 不透明の画像に描くと、半透明の線が下の色と正しく混ざる
        draw = ImageDraw.Draw(img, "RGBA")
        n = len(self.ids)
        if n == 0:
            return _png(img.resize((TILE, TILE)))
        x0, y0 = tx * TILE / scale, ty * TILE / scale          # このタイルが写す世界の範囲
        x1, y1 = (tx + 1) * TILE / scale, (ty + 1) * TILE / scale
        to_px = lambda p: ((p[..., 0] - x0) * scale * S, (p[..., 1] - y0) * scale * S)
        grow = min(1.0, 2 ** (z - 3))       # 引いて見ているときは点や線を細くする

        # 線(タイルの範囲にかかるものだけ)
        if len(self.edges):
            a, b = self.edges[:, 0].astype(int), self.edges[:, 1].astype(int)
            P, Q = self.world[a], self.world[b]
            alpha = self.line_alpha(z)
            hit = ((np.minimum(P[:, 0], Q[:, 0]) <= x1) & (np.maximum(P[:, 0], Q[:, 0]) >= x0) &
                   (np.minimum(P[:, 1], Q[:, 1]) <= y1) & (np.maximum(P[:, 1], Q[:, 1]) >= y0))
            for k in np.where(hit)[0]:
                w, kind = self.edges[k, 2], self.edges[k, 3]
                px, py = to_px(P[k])
                qx, qy = to_px(Q[k])
                if kind == 1:
                    col = th["hippo"] + (alpha,)
                    width = S
                else:
                    col = th["line"] + (alpha,)
                    width = S
                draw.line([(px, py), (qx, qy)], fill=col, width=width)

        # 点とラベル(ラベルがはみ出してくる分も含めて、少し広い範囲の点を描く)
        m = LABEL_MARGIN_PX / scale
        W = self.world
        near = np.where((W[:, 0] >= x0 - m) & (W[:, 0] <= x1 + m) & (W[:, 1] >= y0 - m) & (W[:, 1] <= y1 + m))[0]
        shown = self.labels_at(z)
        font = self.font(text_px(z) * S)
        font_place = self.font(text_px(z) * S, bold=True)
        for i in sorted(near, key=lambda i: -self.label_rank[i]):  # 重要なものを後に描いて上に出す
            px, py = to_px(W[i])
            kind = self.kinds[i]
            if kind == "episode":
                r = max(1, 2.5 * S * grow)
                draw.ellipse([px - r, py - r, px + r, py + r], fill=th["hippo"] + (200,))
                continue
            label = int(i) in shown
            if kind == "place":
                if label:
                    text = self.label_text(i)
                    l, t, r_, b_ = draw.textbbox((0, 0), text, font=font_place)
                    w, h = r_ - l, b_ - t
                    draw.rounded_rectangle([px - w / 2 - 6 * S, py - h / 2 - 4 * S, px + w / 2 + 6 * S, py + h / 2 + 4 * S],
                                           radius=4 * S, fill=th["place_bg"] + (240,), outline=th["place"] + (255,), width=S)
                    draw.text((px - w / 2 - l, py - h / 2 - t), text, font=font_place, fill=th["place"] + (255,))
                else:
                    r = max(1.5, 4 * S * grow)
                    draw.rectangle([px - r, py - r, px + r, py + r], fill=th["place"] + (255,))
                continue
            r = max(1.0, (2 + min(5, math.sqrt(self.degree[i]))) * S * grow)
            draw.ellipse([px - r, py - r, px + r, py + r], fill=th["node"] + (255,), outline=th["edge"] + (255,), width=S)
            if label:
                text = self.label_text(i)
                l, t, r_, b_ = draw.textbbox((0, 0), text, font=font)
                draw.text((px - (r_ - l) / 2 - l, py + r + 2 * S - t), text, font=font, fill=th["ink"] + (230,),
                          stroke_width=2 * S // 2, stroke_fill=th["halo"] + (200,))
        return _png(img.resize((TILE, TILE), Image.LANCZOS))

    def line_alpha(self, z):
        """ズーム z での線の不透明度(0〜255)

        不透明度 a の線を k 本重ねた濃さは 1-(1-a)^k。図を細かいマス目に分けて線の密度を数えておき、
        いちばん混んでいるあたり(上位1%)のマスでも濃さが DENSE_DARKNESS で止まる a を選ぶ。
        そうすると真っ黒につぶれず、混み具合が濃淡として見える。マス目は図全体で共通なので、
        タイルの境目で濃さが食い違わない。
        """
        if self._density is None:
            self._density = self._line_density()
        cell, p99 = self._density
        k = p99 / (cell ** 2 * 2 ** z)       # 混んだマスで、1ピクセルに重なる線の本数(線幅1px)
        alpha = 1 - (1 - DENSE_DARKNESS) ** (1 / max(k, 1e-9))
        return max(2, int(255 * min(LINE_ALPHA_MAX, alpha)))

    def _line_density(self, grid=512):
        """マス目ごとの線の長さの合計(世界座標)。返り値: (マスの大きさ, 上位1%のマスの値)"""
        cell = WORLD / grid
        a, b = self.edges[:, 0].astype(int), self.edges[:, 1].astype(int)
        P, Q = self.world[a], self.world[b]
        length = np.sqrt(((Q - P) ** 2).sum(1))
        n = np.maximum(1, np.ceil(length / (cell / 2)).astype(int))   # 線に沿って細かく点を打つ
        idx = np.repeat(np.arange(len(n)), n)
        t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / np.repeat(n, n)
        pts = P[idx] + (Q[idx] - P[idx]) * t[:, None]
        cells = np.clip((pts / cell).astype(int), 0, grid - 1)
        G = np.zeros((grid, grid))
        np.add.at(G, (cells[:, 1], cells[:, 0]), (length / n)[idx])
        nonzero = G[G > 0]
        return cell, float(np.percentile(nonzero, 99)) if len(nonzero) else 1.0

    # ---- 問い合わせ ----

    def meta(self):
        return {"world": WORLD, "max_zoom": MAX_ZOOM, "tile": TILE, "version": self.version,
                "nodes": len(self.ids), "edges": int(len(self.edges))}

    def node_at(self, wx, wy, z):
        """クリックした位置にある点(近いものから、画面上で 12px 以内)"""
        if not len(self.ids):
            return None
        d = np.sqrt(((self.world - [wx, wy]) ** 2).sum(1)) * 2 ** z
        i = int(d.argmin())
        if d[i] > 12:
            return None
        info = {"id": self.ids[i], "kind": self.kinds[i], "label": self.labels[i],
                "world": self.world[i].tolist(), "degree": int(self.degree[i])}
        if self.kinds[i] == "episode":
            ep = self.episodes[i]
            info.update(concepts=ep["concepts"], nights=ep["nights"])
        else:
            a, b = self.edges[:, 0].astype(int), self.edges[:, 1].astype(int)
            mine = np.where(((a == i) | (b == i)) & (self.edges[:, 3] == 0))[0]
            mine = mine[np.argsort(-self.edges[mine, 2])][:12]
            info["links"] = [[self.labels[b[k] if a[k] == i else a[k]], round(float(self.edges[k, 2]), 3)] for k in mine]
            info["episodes"] = int(((a == i) | (b == i)).sum() - len(np.where(((a == i) | (b == i)) & (self.edges[:, 3] == 0))[0]))
        return info

    def overlay(self, activation, cues, limit=40):
        """想起で活性化した概念を、タイルの上に重ねて描くための位置とつながり"""
        if not len(self.ids):
            return {"nodes": [], "edges": []}
        top = sorted(((n, a) for n, a in activation.items() if n in self.index and a > 0.02),
                     key=lambda x: -x[1])[:limit]
        chosen = {n for n, _ in top} | {c for c in cues if c in self.index}
        nodes = [{"id": n, "kind": self.kinds[self.index[n]], "act": round(activation.get(n, 0), 3),
                  "cue": n in cues, "world": self.world[self.index[n]].tolist()} for n in chosen]
        idx = {self.index[n] for n in chosen}
        edges = []
        for a, b, w, kind in self.edges:
            if kind == 0 and int(a) in idx and int(b) in idx:
                edges.append([self.ids[int(a)], self.ids[int(b)], round(float(w), 3)])
        return {"nodes": nodes, "edges": edges}


def _png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=False)
    return buf.getvalue()
