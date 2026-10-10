// 記憶のネットワーク図(地図と同じタイル方式)
//
// 全体を 256px 四方の「世界」に配置し、ズーム z では 256×2^z px の大きさになる。
// それを 256px のタイルに分け、ブラウザが表示している部分のタイルの中身(線・点・ラベル)だけを返す。
// 絵はブラウザが描く。ズームが上がるほど、ラベルを付ける概念を増やす(地図で拡大すると細かい地名が出るのと同じ)。
//
// 線は半透明にして、重なるほど濃くなるようにする。濃淡で線の密集具合が分かる。
// ただし線が数万本あると、どのズームでも真っ黒になるので、混んでいるところでも真っ黒にならないよう、
// ズームごとに不透明度を下げる(上限 0.3)。

import fs from 'node:fs';
import { Worker } from 'node:worker_threads';
import { isPlace, type Brain, type Episode } from './brain.ts';

const TILE = 256;
const WORLD = 256;             // ズーム0での全体の大きさ(px)
const MAX_ZOOM = 7;
const LINE_ALPHA_MAX = 0.3;
const DENSE_DARKNESS = 0.85;
const LABEL_MARGIN_PX = 160;   // タイルの外にある点のラベルがはみ出してくる分も含める

/** ズーム z でのラベルの文字の大きさ(px)。拡大するほど少しずつ大きくする */
export const textPx = (z: number) => Math.round(10 + 1.2 * z);

/** ラベルの幅の見積もり(全角は文字の大きさ、半角はその約半分) */
function textWidth(text: string, px: number): number {
  let w = 0;
  for (const ch of text) w += /[\u0000-ÿ]/.test(ch) ? 0.56 * px : px;
  return w;
}

type Kind = 'concept' | 'place' | 'episode';

export class GraphMap {
  readonly layoutPath: string;
  version: number | null = null;
  ids: string[] = [];
  kinds: Kind[] = [];
  labels: string[] = [];
  index = new Map<string, number>();
  edges: Float64Array = new Float64Array(0);   // [a, b, 強さ, 種類(0=皮質 1=海馬)] の並び
  degree: Float64Array = new Float64Array(0);
  world: Float64Array = new Float64Array(0);   // [x, y] の並び(0〜256)
  episodes = new Map<number, Episode>();
  pos: Record<string, [number, number]> = {};
  private labelRank: Int32Array = new Int32Array(0);
  private labelPriority: number[] = [];
  private labelSets = new Map<number, Set<number>>();
  private density: [number, number] | null = null;
  private updating: Promise<void> | null = null;

  constructor(layoutPath: string) {
    this.layoutPath = layoutPath;
    try {
      this.pos = JSON.parse(fs.readFileSync(layoutPath, 'utf8'));
    } catch {
      // まだない
    }
  }

  /** 記憶が変わっていたら、図の中身と配置を作り直す(配置の計算は重いので、別のスレッドで行う) */
  update(brain: Brain, version: number): Promise<void> {
    if (version === this.version) return Promise.resolve();
    if (this.updating) return this.updating;
    this.updating = this.rebuild(brain, version).finally(() => (this.updating = null));
    return this.updating;
  }

  private async rebuild(brain: Brain, version: number): Promise<void> {
    const ids: string[] = [], kinds: Kind[] = [], labels: string[] = [];
    const index = new Map<string, number>();
    const add = (id: string, kind: Kind, label: string) => {
      if (!index.has(id)) {
        index.set(id, ids.length);
        ids.push(id);
        kinds.push(kind);
        labels.push(label);
      }
      return index.get(id)!;
    };
    const concept = (n: string) => add(n, isPlace(n) ? 'place' : 'concept', n);
    const edges: number[] = [];
    for (const [key, link] of Object.entries(brain.cortex)) {
      const [a, b] = key.split('\t');
      edges.push(concept(a), concept(b), brain.effective(link), 0);
    }
    const episodes = new Map<number, Episode>();
    for (const ep of brain.hippocampus) {
      const h = add(`ep:${ep.t.toFixed(6)}:${ep.concepts.join('|')}`, 'episode', ep.concepts.join(' / '));
      episodes.set(h, ep);
      for (const c of ep.concepts) edges.push(h, concept(c), brain.hippoStrength(ep), 1);
    }
    const E = Float64Array.from(edges);
    const degree = new Float64Array(ids.length);
    for (let k = 0; k < E.length; k += 4) {
      degree[E[k]]++;
      degree[E[k + 1]]++;
    }
    const known = ids.map((id) => this.pos[id] ?? null);
    const X = await layoutInWorker(ids.length, E, known);
    this.pos = Object.fromEntries(ids.map((id, i) => [id, [X[2 * i], X[2 * i + 1]] as [number, number]]));
    fs.writeFileSync(this.layoutPath, JSON.stringify(this.pos));
    Object.assign(this, { ids, kinds, labels, index, edges: E, degree, episodes });
    this.toWorld(X);
    this.labelOrder();
    this.density = null;
    this.version = version;
  }

  /** 配置座標を 0〜256 の世界座標にそろえる(縦横比は保つ) */
  private toWorld(X: Float64Array): void {
    const n = this.ids.length;
    this.world = new Float64Array(2 * n);
    if (!n) return;
    let [x0, y0, x1, y1] = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
      x0 = Math.min(x0, X[2 * i]); x1 = Math.max(x1, X[2 * i]);
      y0 = Math.min(y0, X[2 * i + 1]); y1 = Math.max(y1, X[2 * i + 1]);
    }
    const span = Math.max(x1 - x0, y1 - y0, 1e-9);
    const pad = WORLD * 0.04;
    const inner = WORLD - 2 * pad;
    const offX = (inner - ((x1 - x0) / span) * inner) / 2;   // 中央に寄せる
    const offY = (inner - ((y1 - y0) / span) * inner) / 2;
    for (let i = 0; i < n; i++) {
      this.world[2 * i] = ((X[2 * i] - x0) / span) * inner + pad + offX;
      this.world[2 * i + 1] = ((X[2 * i + 1] - y0) / span) * inner + pad + offY;
    }
  }

  /** ラベルを付ける優先順位(場所 → つながりの多い概念) */
  private labelOrder(): void {
    const n = this.ids.length;
    const score = Array.from(this.degree, (d, i) => (this.kinds[i] === 'place' ? d + 1e6 : this.kinds[i] === 'episode' ? -1 : d));
    this.labelPriority = Array.from({ length: n }, (_, i) => i).sort((a, b) => score[b] - score[a]);
    this.labelRank = new Int32Array(n);
    this.labelPriority.forEach((i, r) => (this.labelRank[i] = r));
    this.labelSets.clear();
  }

  labelText(i: number): string {
    const text = this.labels[i];
    if (this.kinds[i] !== 'place') return text;
    return text.length <= 34 ? text : '…' + text.slice(-33);
  }

  /** ズーム z の画面座標(px)でのラベルの範囲 */
  private labelBox(i: number, z: number): [number, number, number, number] {
    const px = textPx(z);
    const w = textWidth(this.labelText(i), px);
    const h = px * 1.2;
    const cx = this.world[2 * i] * 2 ** z;
    const cy = this.world[2 * i + 1] * 2 ** z;
    if (this.kinds[i] === 'place') return [cx - w / 2 - 6, cy - h / 2 - 4, cx + w / 2 + 6, cy + h / 2 + 4];
    const r0 = 2 + Math.min(5, Math.sqrt(this.degree[i]));
    return [cx - w / 2, cy + r0 + 2, cx + w / 2, cy + r0 + 4 + h];
  }

  /** ズーム z でラベルを出す点(重要なものから順に置き、既に置いたラベルと重なるものは出さない)
   *  画面全体の座標で決めるので、タイルの境目でラベルの有無が食い違わない(地図のラベル配置と同じ)。 */
  labelsAt(z: number): Set<number> {
    const hit = this.labelSets.get(z);
    if (hit) return hit;
    const cell = 64;
    const grid = new Map<string, [number, number, number, number][]>();
    const shown = new Set<number>();
    for (const i of this.labelPriority) {
      if (this.kinds[i] === 'episode') break;
      const box = this.labelBox(i, z);
      const cells: string[] = [];
      for (let cx = Math.floor(box[0] / cell); cx <= Math.floor(box[2] / cell); cx++)
        for (let cy = Math.floor(box[1] / cell); cy <= Math.floor(box[3] / cell); cy++) cells.push(`${cx},${cy}`);
      const clash = cells.some((c) => (grid.get(c) ?? []).some((o) => !(box[2] < o[0] || box[0] > o[2] || box[3] < o[1] || box[1] > o[3])));
      if (clash) continue;
      for (const c of cells) {
        if (!grid.has(c)) grid.set(c, []);
        grid.get(c)!.push(box);
      }
      shown.add(i);
    }
    this.labelSets.set(z, shown);
    return shown;
  }

  /** ズーム z での線の不透明度(0〜1)
   *  不透明度 a の線を k 本重ねた濃さは 1-(1-a)^k。図を細かいマス目に分けて線の密度を数えておき、
   *  いちばん混んでいるあたり(上位1%)のマスでも濃さが DENSE_DARKNESS で止まる a を選ぶ。 */
  lineAlpha(z: number): number {
    this.density ??= this.lineDensity();
    const [cell, p99] = this.density;
    const k = p99 / (cell ** 2 * 2 ** z);   // 混んだマスで、1ピクセルに重なる線の本数(線幅1px)
    const alpha = 1 - (1 - DENSE_DARKNESS) ** (1 / Math.max(k, 1e-9));
    return Math.max(2 / 255, Math.min(LINE_ALPHA_MAX, alpha));
  }

  /** マス目ごとの線の長さの合計。戻り値: [マスの大きさ, 上位1%のマスの値] */
  private lineDensity(grid = 512): [number, number] {
    const cell = WORLD / grid;
    const G = new Float64Array(grid * grid);
    const W = this.world;
    for (let k = 0; k < this.edges.length; k += 4) {
      const a = this.edges[k], b = this.edges[k + 1];
      const px = W[2 * a], py = W[2 * a + 1], qx = W[2 * b], qy = W[2 * b + 1];
      const len = Math.hypot(qx - px, qy - py);
      const n = Math.max(1, Math.ceil(len / (cell / 2)));   // 線に沿って細かく点を打つ
      for (let s = 0; s < n; s++) {
        const t = (s + 0.5) / n;
        const cx = Math.min(grid - 1, Math.max(0, Math.floor((px + (qx - px) * t) / cell)));
        const cy = Math.min(grid - 1, Math.max(0, Math.floor((py + (qy - py) * t) / cell)));
        G[cy * grid + cx] += len / n;
      }
    }
    const nz = Array.from(G).filter((v) => v > 0).sort((a, b) => a - b);
    return [cell, nz.length ? nz[Math.min(nz.length - 1, Math.floor(0.99 * (nz.length - 1)))] : 1];
  }

  /** タイル1枚の中身(タイルの左上を原点にした px 座標) */
  tile(z: number, tx: number, ty: number) {
    const scale = 2 ** z;
    const x0 = (tx * TILE) / scale, y0 = (ty * TILE) / scale, x1 = ((tx + 1) * TILE) / scale, y1 = ((ty + 1) * TILE) / scale;
    const W = this.world;
    const toPx = (i: number): [number, number] => [Math.round((W[2 * i] - x0) * scale * 10) / 10, Math.round((W[2 * i + 1] - y0) * scale * 10) / 10];
    const lines: number[] = [];
    for (let k = 0; k < this.edges.length; k += 4) {
      const a = this.edges[k], b = this.edges[k + 1];
      const px = W[2 * a], py = W[2 * a + 1], qx = W[2 * b], qy = W[2 * b + 1];
      if (Math.min(px, qx) > x1 || Math.max(px, qx) < x0 || Math.min(py, qy) > y1 || Math.max(py, qy) < y0) continue;
      lines.push(...toPx(a), ...toPx(b), this.edges[k + 3]);
    }
    const m = LABEL_MARGIN_PX / scale;
    const shown = this.labelsAt(z);
    const grow = Math.min(1, 2 ** (z - 3));   // 引いて見ているときは点や線を細くする
    const nodes: any[] = [];
    for (let i = 0; i < this.ids.length; i++) {
      const wx = W[2 * i], wy = W[2 * i + 1];
      if (wx < x0 - m || wx > x1 + m || wy < y0 - m || wy > y1 + m) continue;
      const [px, py] = toPx(i);
      const r = this.kinds[i] === 'episode' ? Math.max(0.5, 2.5 * grow)
        : this.kinds[i] === 'place' ? Math.max(0.75, 4 * grow) : Math.max(0.5, (2 + Math.min(5, Math.sqrt(this.degree[i]))) * grow);
      nodes.push({ x: px, y: py, k: this.kinds[i][0], r: Math.round(r * 10) / 10, l: shown.has(i) ? this.labelText(i) : undefined, rank: this.labelRank[i] });
    }
    nodes.sort((a, b) => b.rank - a.rank);   // 重要なものを後に描いて上に出す
    for (const n of nodes) delete n.rank;
    return { alpha: this.lineAlpha(z), px: textPx(z), lines, nodes };
  }

  meta() {
    return { world: WORLD, max_zoom: MAX_ZOOM, tile: TILE, version: this.version, nodes: this.ids.length, edges: this.edges.length / 4 };
  }

  /** クリックした位置にある点(近いものから、画面上で 12px 以内) */
  nodeAt(wx: number, wy: number, z: number) {
    if (!this.ids.length) return null;
    let best = -1, bd = Infinity;
    for (let i = 0; i < this.ids.length; i++) {
      const d = Math.hypot(this.world[2 * i] - wx, this.world[2 * i + 1] - wy) * 2 ** z;
      if (d < bd) [best, bd] = [i, d];
    }
    if (bd > 12) return null;
    const i = best;
    const info: any = { id: this.ids[i], kind: this.kinds[i], label: this.labels[i], world: [this.world[2 * i], this.world[2 * i + 1]], degree: this.degree[i] };
    if (this.kinds[i] === 'episode') {
      const ep = this.episodes.get(i)!;
      Object.assign(info, { concepts: ep.concepts, nights: ep.nights });
    } else {
      const links: [string, number][] = [];
      let eps = 0;
      for (let k = 0; k < this.edges.length; k += 4) {
        const a = this.edges[k], b = this.edges[k + 1];
        if (a !== i && b !== i) continue;
        if (this.edges[k + 3] === 1) eps++;
        else links.push([this.labels[a === i ? b : a], Math.round(this.edges[k + 2] * 1000) / 1000]);
      }
      info.links = links.sort((x, y) => y[1] - x[1]).slice(0, 12);
      info.episodes = eps;
    }
    return info;
  }

  /** 想起で活性化した概念を、タイルの上に重ねて描くための位置とつながり */
  overlay(activation: Map<string, number>, cues: string[], limit = 40) {
    if (!this.ids.length) return { nodes: [], edges: [] };
    const top = [...activation].filter(([n, a]) => this.index.has(n) && a > 0.02).sort((a, b) => b[1] - a[1]).slice(0, limit);
    const chosen = new Set([...top.map(([n]) => n), ...cues.filter((c) => this.index.has(c))]);
    const cueSet = new Set(cues);
    const nodes = [...chosen].map((n) => {
      const i = this.index.get(n)!;
      return { id: n, kind: this.kinds[i], act: Math.round((activation.get(n) ?? 0) * 1000) / 1000, cue: cueSet.has(n), world: [this.world[2 * i], this.world[2 * i + 1]] };
    });
    const idx = new Set([...chosen].map((n) => this.index.get(n)!));
    const edges: [string, string, number][] = [];
    for (let k = 0; k < this.edges.length; k += 4) {
      const a = this.edges[k], b = this.edges[k + 1];
      if (this.edges[k + 3] === 0 && idx.has(a) && idx.has(b)) edges.push([this.ids[a], this.ids[b], Math.round(this.edges[k + 2] * 1000) / 1000]);
    }
    return { nodes, edges };
  }
}

/** 力学モデルで配置する(すでに配置済みの点はそのまま使い、新しい点だけ近くに置いて整える)。別のスレッドで計算する */
function layoutInWorker(n: number, edges: Float64Array, known: ([number, number] | null)[]): Promise<Float64Array> {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./layout.ts', import.meta.url), { workerData: { n, edges, known } });
    w.once('message', (X: Float64Array) => resolve(X));
    w.once('error', reject);
  });
}
