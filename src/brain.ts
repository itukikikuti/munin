// munin: 脳の記憶モデル
//
//   皮質   : 意味ネットワーク。知識は概念ノード間のリンクに断片として散らばる
//   海馬   : 新しい出来事(エピソード)を一発で記録する。速く覚え、速く忘れる
//   睡眠   : 海馬のエピソードを再生し、皮質のリンクを少しずつ強める(固定化)
//   想起   : 手がかり → 海馬のパターン補完 + 皮質の活性化拡散 → 断片から再構成
//   強化   : 思い出したリンクは使用履歴が増え、取り出しやすくなる
//   忘却   : 使わないリンクは取り出しにくくなる(ACT-R の基本活性化の式)
//
// 時間は実時間ではなく、内部時計(時間単位)で進める。

import fs from 'node:fs';
import path from 'node:path';
import { Embedder } from './embed.ts';
import { asksTodo, daysBetween, type Intent } from './intentions.ts';
import type { Detail, Store } from './store.ts';
import { clip, contentWords, DEICTIC, wordBoundaries } from './text.ts';

// ---- パラメータ(評価で変えられるように、1つのオブジェクトにまとめる) ----
export const P = {
  DECAY_D: 0.5,                // 忘却の速さ(ACT-R の d)
  ACCESS_TAU: -2.5,            // 取り出しやすさの閾値
  ACCESS_S: 0.5,               // 閾値付近のなだらかさ
  HIPPO_HALFLIFE: 48,          // 海馬の記録の半減期(時間)
  HIPPO_FORGET_BELOW: 0.05,    // これ以下に薄れた海馬の記録は消える
  NIGHTS_TO_INDEPENDENCE: 3,   // 何晩再生されたら海馬なしで思い出せるか
  REPLAYS_PER_NIGHT: 3,
  CONSOLIDATE_LR: 0.15,        // 1回の再生で皮質リンクが強まる割合
  INTERLEAVE_LR: 0.05,         // 古い記憶を混ぜて再生するときの強化
  DOWNSCALE: 0.97,             // 睡眠中のシナプス整理(全体を少し弱める)
  PRUNE_BELOW: 0.02,           // これ以下のリンクは消える
  SLEEP_HOURS: 8,
  ASSOC_FREQ_WEIGHT: 1.0,      // 連合強度で「よく一緒に出てくるか」を「特に結びついているか」に対してどれだけ重視するか
  CONTEXT_MIN: 0.3,            // 指示が今の話題を指していないときの、ワーキングメモリの強さ(プライミング用)
  PRIMING: 0.2,                // 指示が今の話題を指していないとき、今の話題に近い記憶の活性を最大で何割増しにするか
  GIST_SHARE: 0.5,             // 手がかりに向ける注意のうち、出来事の要旨による照合に向ける割合
  GIST_TOPK: 10,               // 呼び起こすエピソード記憶の痕跡の数
  GIST_TEMP: 0.03,             // 意味の近さの差を、呼び起こしの強さの差にどれだけはっきり反映するか(小さいほど1位が独占)
  GIST_DECAY: 0.2,             // 古い痕跡ほど呼び起こされにくくする度合い(経過時間^-d)
  GIST_Z_MIN: 1.5,             // 一番近い痕跡が、他より何σ高ければ呼び起こし始めるか
  GIST_Z_FULL: 3.5,
  GIST_ABS_MIN: 0.70,          // 痕跡との類似度がこれ以下なら、呼び起こさない(無関係な依頼でも 0.7 程度にはなる)
  GIST_ABS_FULL: 0.85,         // 類似度がこれ以上なら、十分に近い
  // 関連することを思い出す(出来事の活性 A = 要旨 + 概念 + 新しさ + 細部 + キーワード + 場所、確かさは ACT-R の検索確率)
  EP_W_GIST: 1.0,              // 要旨の意味の近さの重み
  EP_W_CONCEPT: 1.0,           // 出来事に含まれる概念の活性の重み
  EP_W_BASE: 0.05,             // 新しさ(基本活性化)の重み
  EP_DECAY: 0.5,               // 新しさの減り方(ACT-R の d)
  EP_TAU: 1.55,                // 出来事の検索閾値(自動評価の当たり外れから推定)
  EP_S: 1.26,                  // 出来事の検索確率の、閾値付近でのなだらかさ(同上)
  EP_W_PLACE: 0.5,             // 出来事が、今の作業フォルダ(またはそれを含むプロジェクト)で起きたかの重み
  EP_W_KEYWORD: 0.5,           // 出来事の記録に、依頼の語が文字どおり含まれるかの重み
  EP_W_DETAIL: 1.0,            // 出来事の記録の中に、依頼とよく合う部分があるかの重み
  KEYWORD_TERMS: 12,           // キーワード検索に使う、依頼の語の数
  CHUNK_SIM_MIN: 0.45,         // 記録の部分(チャンク)と依頼の類似度がこれ以下なら、関係なしとみなす
  CHUNK_SIM_FULL: 0.70,        // 類似度がこれ以上なら、十分に関係がある
  CHOICE_T: 0.03,              // 場所の選択規則の温度(ACT-R の t)。小さいほど1位が選ばれやすい
  SHOW_CONF: 0.3,              // この確かさ以上のものだけ差し込む(確かさ = 実際に関係している確率の目安)。
                               // 作業フォルダの手がかりがない会話(アプリのチャット)では確かさが低めに出るので、外れを少し許して取りこぼしを防ぐ
  PLACE_SHOW_CONF: 0.2,
  EPISODES_SHOWN: 5,
  ABSTAIN: false,              // 確かな出来事を1つも思い出せないとき(会話の最初を除く)、語や場所も差し込まない
  CANDIDATES_KEPT: 50,         // 評価用に残す、並べた出来事の上位の数(候補の段階での取りこぼしを測る)
  DETAILS_SHOWN: 3,            // 思い出した出来事ごとに、差し込む部分の数
  PROMPT_HINT_CHARS: 60,       // 差し込む文章に載せる、依頼の長さ
  DETAIL_HINT_CHARS: 120,      // 差し込む文章に載せる、関係する部分の長さ
  HINTS_SHOWN: 6,              // 思い出した出来事ごとに載せる、手がかりの語の数
  // 展望記憶(これからやること)
  DONE_OVERLAP: 3,             // 後のやり取りと、何個の概念が重なれば同じ件とみなすか(完了の判定)
  INTENT_OVERDUE_DAYS: 7,      // 期限をこの日数より過ぎたものは、もう追わない
  INTENT_FORGET_DAYS: 30,      // 期限のないものは、この日数が経つと思い出さない
  INTENT_RECENT_DAYS: 7,       // 新しい会話の最初に出す、期限のないもの(最近の保留・連絡)の新しさ
  INTENT_W_REL: 1.0,           // 依頼との関係の重み
  INTENT_W_URGENCY: 1.0,       // 期限の近さの重み
  INTENT_W_RECENT: 0.5,        // 新しさの重み
  INTENTS_SHOWN: 3,            // 新しい会話の最初に出す数
  INTENTS_ASKED: 6,            // これからやることを尋ねられたときに出す数
  TODO_CHARS: 100,
  STALE_PLACE: 0.2,            // 今はもうない場所の活性を何倍にするか
  SHOWN_PER_SESSION: 200,      // 会話ごとに覚えておく、差し込んだ記憶の鍵の数
  SHOWN_SESSIONS: 50,          // 差し込んだ記憶の鍵を覚えておく会話の数(古い会話の分は忘れる)
  SHORT_ID: 8,                 // 記録の鍵の、差し込む文章に載せる長さ(先頭から)
  CONCEPTS_SHOWN: 15,
  PLACES_SHOWN: 3,
  RECENT_SHOWN: 3,
  DEICTIC_CONTEXT: 0.6,        // 指示語(「この」「それ」など)があるとき、ワーキングメモリに向ける注意
  OMITTED_CONTEXT: 0.8,        // 対象が省かれている(「直して」だけ)とき、ワーキングメモリに向ける注意
  // 手がかりとしての語の強さ: 自分の経験(出来事)の中で珍しい語ほど強い手がかりになる。
  // その語が出てきた出来事の割合が INFO_COMMON 以上ならほぼ役に立たず、INFO_RARE 以下なら十分に役立つ
  INFO_COMMON: 0.1,
  INFO_RARE: 0.002,
  INFO_FLOOR: 0.1,             // ありふれた語でも、手がかりとしての強さはこれ以上残す
  CONTEXT_HALFLIFE: 1,         // ワーキングメモリの半減期(時間)
  WM_CAPACITY: 5,              // ワーキングメモリに保てる概念の数(人間は 4±1 程度)
  FOCUS_DECAY: 0.3,            // 注意の焦点から1つ前の話題に戻るごとに、強さが何倍になるか
  STRONG_CUE: 0.5,             // これ以上の手がかりだけを、強化と文脈の対象にする
  RECALL_LR: 0.05,             // 想起による強化
  EXACT_USES: 3,               // 取り出しやすさの計算で、時刻を正確に覚えておく直近の使用回数
  RECALL_THRESHOLD: 0.02,      // これ未満の活性は、何も思い浮かばないのと同じ
  RETRIEVAL_TAU: 0.05,         // 検索閾値(ACT-R の τ)。活性がこれを超えると、思い出せる確率が5割を超える
  RETRIEVAL_S: 0.015,          // 検索確率の、閾値付近でのなだらかさ(ACT-R の s)
  AMBIGUITY_RATIO: 0.8,        // 2位がこの比率以上なら「迷う」
  INNER_PLACE_RATIO: 0.2,      // 中のフォルダの活性が親のこの比率以上(かつ検索閾値以上)なら、中のフォルダを答えにする
  SEM_TOPK: 5,                 // 意味が近い概念を最大いくつ手がかりにするか
  SEM_MAX: 0.8,                // 意味が近いだけの手がかりの最大強度(名前の一致は 1.0)
  SEM_Z_MIN: 2.0,              // 他の概念より何σ高ければ手がかりにし始めるか
  SEM_Z_FULL: 4.0,             // 何σ高ければ最大強度にするか
  SEM_WITH_EXACT: 0.5,         // 名前の一致があるときの、連想による手がかりの割合
  SEM_ABS_MIN: 0.5,            // 類似度がこれ以下なら、どれだけ際立っていても手がかりにしない
  SEM_ABS_FULL: 0.65,          // 類似度がこれ以上なら、絶対値の面では十分に近い
  INTENT_KIND_BONUS: { 期限: 0.3, 連絡: 0.15, 保留: 0.0 } as Record<string, number>,   // 取り出しの精度が高い種類ほど前に出す
};

// 仕組みを1つずつ外して、それぞれの効果を切り分けるためのスイッチ(評価で使う)
export const ABLATIONS: Record<string, string> = {
  sleep: '睡眠による固定化', spread: '皮質の活性化拡散', hippo: '海馬のパターン補完',
  gist: '出来事の要旨の照合', concept: '出来事の概念の活性', recency: '新しさ',
  detail: '記録の細部の照合', context: 'ワーキングメモリ(文脈)', keyword: 'キーワード検索',
};
export const ABLATE = new Set<string>();

// ---- 型 ----
export interface Link { w: number; recent: number[]; n: number; first: number | null }
export interface Episode { concepts: string[]; t: number; nights: number }
export interface Trace {
  gist: string; concepts: string[]; t: number;
  prompt?: string; reply?: string; time?: string; session?: string; ref?: string; uses?: number[];
}
export interface Intention {
  id: string; ref: string | null; text: string; due: string | null; kind: string; concepts: string[];
  time: string; t: number; status: 'open' | 'done'; done_t?: number;
}
export interface Answer { place: string | null; about: string[]; confidence: number; activation?: number; ambiguous: string[]; uncertain: boolean }
export interface EpisodeView {
  trace?: Trace; conf: number | null; when: string; details: { kind: string; text: string; sim: number }[];
  id: string; ref: string; hints: string[]; prompt: string; reply: string; places: string[]; concepts: string[]; activation?: number;
}
export interface Recollection {
  ranked: EpisodeView[]; todos: TodoView[]; episodes: EpisodeView[];
  concepts: { name: string; conf: number }[]; places: PlaceView[]; recent: EpisodeView[];
  cues: string[]; answer: Answer; text: string;
}
export interface TodoView { id: string; ref: string | null; text: string; due: string | null; kind: string; places: string[] }
export interface PlaceView { name: string; conf: number; stale: boolean }
export type Features = { gist: number; concept: number; base: number; detail: number; keyword: number; place: number };

const PLACE_RE = /^([A-Za-z]:[\\/]|\/|~[\\/])/;
export const isPlace = (name: string) => PLACE_RE.test(name);

export function retrievalProbability(a: number, tau: number, s: number): number {
  return 1 / (1 + Math.exp(-(a - tau) / s));
}

/** inner が outer の中のフォルダか */
export function isInside(inner: string, outer: string): boolean {
  const i = inner.toLowerCase().replace(/[\\/]+$/, '');
  const o = outer.toLowerCase().replace(/[\\/]+$/, '');
  return i !== o && (i.startsWith(o + '\\') || i.startsWith(o + '/'));
}

/** 今の作業フォルダが、その場所そのものか、その中か(その場所で作業している) */
export function atPlace(cwd: string, place: string): boolean {
  return cwd.toLowerCase().replace(/[\\/]+$/, '') === place.toLowerCase().replace(/[\\/]+$/, '') || isInside(cwd, place);
}

export const linkKey = (a: string, b: string) => (a < b ? `${a}\t${b}` : `${b}\t${a}`);

/** 組み合わせ(Python の itertools.combinations(xs, 2) と同じ順) */
function* pairs<T>(xs: T[]): Generator<[T, T]> {
  for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) yield [xs[i], xs[j]];
}

const uniq = <T>(xs: Iterable<T>) => [...new Set(xs)];

function meanStd(xs: Float32Array): [number, number] {
  let m = 0;
  for (const x of xs) m += x;
  m /= xs.length || 1;
  let v = 0;
  for (const x of xs) v += (x - m) ** 2;
  return [m, Math.sqrt(v / (xs.length || 1))];
}

/** 大きい順の並び(添字) */
function argsortDesc(xs: Float32Array): number[] {
  return Array.from(xs.keys()).sort((a, b) => xs[b] - xs[a]);
}

/** 思い出したことを、依頼に差し込む文章にする
 *
 * 中身を全部渡すのではなく、記憶の手がかり(いつ・どこで・何の件か・関係しそうな語や部分)と、
 * 詳しく見るための鍵だけを渡す。相手(Claude)は、手がかりをもとに自分で調べたり、鍵で記録を開いたりできる。 */
export function recollectionText(r: Pick<Recollection, 'todos' | 'recent' | 'episodes' | 'places' | 'concepts'>): string {
  const head = (e: EpisodeView) => {
    const when = e.when ? e.when.slice(5).replace(/-/g, '/') : '';
    return `[${e.id}] ${when} ${e.places.join(' ')} 「${clip(e.prompt, P.PROMPT_HINT_CHARS)}」`.replace(/  /g, ' ');
  };
  const lines: string[] = [];
  if (r.todos?.length) {
    lines.push('未完了のこと(期限・保留・連絡):');
    for (const it of r.todos) {
      const due = it.due ? `${it.due.slice(5).replace(/-/g, '/')}まで ` : '';
      lines.push(`- [${it.id}] ${due}${it.places.join(' ')} 「${clip(it.text, P.TODO_CHARS)}」`.replace(/  /g, ' '));
    }
  }
  if (r.recent.length) {
    lines.push('直前の作業:');
    for (const e of r.recent) lines.push(`- ${head(e)}`);
  }
  if (r.episodes.length) {
    lines.push('関係しそうな記憶(確かさ順):');
    for (const e of r.episodes) {
      lines.push(`- ${head(e)} 確かさ${(e.conf ?? 0).toFixed(2)}`);
      const hints: string[] = [];
      if (e.hints.length) hints.push('手がかり: ' + e.hints.join(', '));
      if (e.details.length) {   // そのやり取りの中で、今の依頼と最も関係する部分
        const d = e.details[0];
        hints.push(`${d.kind}「${clip(d.text, P.DETAIL_HINT_CHARS)}」`);
      }
      if (hints.length) lines.push('  ' + hints.join(' / '));
    }
  }
  if (r.places.length) lines.push('関係しそうな場所: ' + r.places.map((p) => p.name + (p.stale ? '(今はない)' : '')).join(', '));
  if (r.concepts.length) lines.push('関連する語・ファイル: ' + r.concepts.map((c) => c.name).join(', '));
  if (!lines.length) return '';
  lines.push('([ ] は記録の鍵。詳しい中身は munin の recall_detail で鍵を開ける。足りなければ recall で探し直せる)');
  return '<思い出したこと>\n' + lines.join('\n') + '\n</思い出したこと>';
}

export class Brain {
  readonly path: string;
  embedder: Embedder | null;
  store: Store | null;
  now = 0;
  seq = 0;               // 注意を向けた回数(時刻が同じでも、どれが新しい話題か分かるように)
  /** link_key -> {w: 保存強度, recent: 直近の使用時刻, n: 使用回数, first: 最初の使用時刻} */
  cortex: Record<string, Link> = {};
  hippocampus: Episode[] = [];
  /** エピソード記憶の痕跡: 出来事ごとの要旨(意味のベクトルの鍵)と、含まれていた概念。
   *  海馬から皮質へ移ったあとも残る(多重痕跡理論)。古いものほど呼び起こされにくい */
  traces: Trace[] = [];
  /** 展望記憶: これからやること(期限・保留・連絡)。完了したら status が done になる */
  intentions: Intention[] = [];
  /** 概念 -> [最後に意識した時刻, 注意を向けた順番] */
  context: Record<string, [number, number]> = {};
  /** 会話ごとに、すでに差し込んだ記憶の鍵(同じ会話で同じ記憶を繰り返し差し込まない) */
  shown: Record<string, string[]> = {};
  checkPlaces = true;    // 思い出した場所が今もあるか確かめる(評価では、過去の時点で思い出すので確かめない)
  lastCues: Record<string, number> = {};
  lastActivation: Map<string, number> = new Map();
  lastTraceSim: Map<Trace, number> = new Map();
  lastFeatures: Map<Trace, Features> = new Map();
  lastCandidates: Trace[] = [];
  private rev = 0;       // 皮質が変わるたびに増える(つながりの計算を使い回すため)
  private assoc: { key: string; adj: Map<string, [string, number][]>; degree: Map<string, number>; total: number } | null = null;
  private placeCache = new Map<string, [boolean, number]>();
  private dfCache: { n: number; df: Map<string, number> } | null = null;
  private knownCache: { key: string; names: Set<string> } | null = null;

  constructor(file: string, embedder: Embedder | null, store: Store | null) {
    this.path = file;
    this.embedder = embedder;
    this.store = store;
    if (fs.existsSync(file)) {
      const s = JSON.parse(fs.readFileSync(file, 'utf8'));
      this.now = s.now;
      this.cortex = s.cortex;
      this.hippocampus = s.hippocampus;
      this.traces = s.traces ?? [];
      this.intentions = s.intentions ?? [];
      this.seq = s.seq ?? 0;
      this.shown = s.shown ?? {};
      this.context = Object.fromEntries(Object.entries(s.context ?? {}).map(([c, v]) => [c, Array.isArray(v) ? v : [v, 0]])) as any;
    }
  }

  save(): void {
    fs.mkdirSync(path.dirname(this.path), { recursive: true });
    const tmp = this.path + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify({ now: this.now, seq: this.seq, cortex: this.cortex, hippocampus: this.hippocampus,
      traces: this.traces, context: this.context, shown: this.shown, intentions: this.intentions }));
    fs.renameSync(tmp, this.path);   // 書いている途中で止まっても、前の記憶が壊れないように
    this.embedder?.save();
  }

  // ---- 強度の計算 ----

  /** リンクを使ったことを記録する(直近の数回は時刻を、それより前は回数だけ残す) */
  use(link: Link): void {
    link.recent = [...link.recent, this.now].slice(-P.EXACT_USES);
    link.n += 1;
    if (link.first === null) link.first = this.now;
    this.rev++;
  }

  /** 取り出しやすさ(0〜1)。使うほど高く、時間とともに下がる
   *
   * ACT-R の基本活性化 B = ln(Σ 経過時間^-d) を、Petrov(2006)のハイブリッド近似で求める。
   * 直近の数回は正確に足し、それより前の使用は、最初の使用から直近までの間に
   * 均等に散らばっていたとみなして積分で足す。使用履歴を捨てずに一定の計算量で済む。 */
  access(link: Link): number {
    if (!link.n) return 0;
    const d = P.DECAY_D;
    const ages = link.recent.map((t) => this.now - t + 1);
    let total = ages.reduce((s, a) => s + a ** -d, 0);
    const rest = link.n - ages.length;
    if (rest > 0) {
      const tk = Math.max(...ages);                 // 正確に数えた中で一番古い使用
      const tn = this.now - (link.first ?? this.now) + 1;   // 最初の使用
      total += tn > tk ? (rest * (tn ** (1 - d) - tk ** (1 - d))) / ((1 - d) * (tn - tk)) : rest * tn ** -d;
    }
    const b = Math.log(total);
    return 1 / (1 + Math.exp(-(b - P.ACCESS_TAU) / P.ACCESS_S));
  }

  effective(link: Link): number {
    return link.w * this.access(link);
  }

  hippoStrength(ep: Episode): number {
    return 0.5 ** ((this.now - ep.t) / P.HIPPO_HALFLIFE);
  }

  /** 皮質のつながり: 隣接、概念ごとのつながりの強さの合計、全体の合計(時刻と皮質が変わらないうちは使い回す) */
  associations() {
    const key = `${this.rev}|${this.now}`;   // 皮質が変わると rev が増える
    if (this.assoc && this.assoc.key === key) return this.assoc;
    const adj = new Map<string, [string, number][]>();
    for (const [k, link] of Object.entries(this.cortex)) {
      const [a, b] = k.split('\t');
      const eff = this.effective(link);
      if (!adj.has(a)) adj.set(a, []);
      if (!adj.has(b)) adj.set(b, []);
      adj.get(a)!.push([b, eff]);
      adj.get(b)!.push([a, eff]);
    }
    const degree = new Map<string, number>();
    let total = 0;
    for (const [n, links] of adj) {
      const s = links.reduce((x, [, e]) => x + e, 0);
      degree.set(n, s);
      total += s;
    }
    this.assoc = { key, adj, degree, total: total || 1 };
    return this.assoc;
  }

  knownConcepts(): Set<string> {
    const key = `${this.rev}|${this.hippocampus.length}|${Object.keys(this.context).join('\u0001')}`;
    if (this.knownCache && this.knownCache.key === key) return this.knownCache.names;
    const names = new Set(Object.keys(this.context));
    for (const k of Object.keys(this.cortex)) for (const n of k.split('\t')) names.add(n);
    for (const ep of this.hippocampus) for (const c of ep.concepts) names.add(c);
    this.knownCache = { key, names };
    return names;
  }

  /** 手がかりとしての情報量(0〜1)
   *
   * 自分の経験の中でよく出てくる語(「設定」「アプリ」)は、どの記憶を指しているかの手がかりにならない。
   * その語が出てきた出来事の割合が小さいほど強い手がかりになる(情報量 -log p)。 */
  informativeness(word: string): number {
    if (isPlace(word)) return 1;
    if (!this.dfCache || this.dfCache.n !== this.traces.length) {
      const df = new Map<string, number>();
      for (const tr of this.traces) for (const c of new Set(tr.concepts)) df.set(c, (df.get(c) ?? 0) + 1);
      this.dfCache = { n: this.traces.length, df };
    }
    const n = this.dfCache.n;
    if (!n) return 1;
    const p = (this.dfCache.df.get(word) ?? 0) / n;
    const lo = -Math.log(P.INFO_COMMON);
    const hi = -Math.log(P.INFO_RARE);
    const info = p > 0 ? -Math.log(p) : hi;
    return Math.min(1, Math.max(P.INFO_FLOOR, (info - lo) / (hi - lo)));
  }

  // ---- 覚える(海馬に記録) ----

  /** 出来事を覚える。gist: その出来事の要旨を表す文(依頼文)。note: 思い出したときに伝える中身 */
  async learn(concepts: string[], gist: string | null = null, note: Partial<Trace> = {}): Promise<void> {
    concepts = uniq(concepts);
    this.hippocampus.push({ concepts, t: this.now, nights: 0 });
    if (this.embedder) {
      const key = await this.embedder.rememberGist(gist || concepts.join(' '));
      if (key) this.traces.push({ gist: key, concepts, t: this.now, ...note });
    }
    this.attend(concepts);
  }

  // ---- 睡眠(固定化) ----

  sleep(hours = P.SLEEP_HOURS): string[] {
    if (ABLATE.has('sleep')) {   // 睡眠を外す: 再生も固定化もせず、海馬の記録は薄れて消えるだけ
      this.hippocampus = this.hippocampus.filter((ep) => this.hippoStrength(ep) >= P.HIPPO_FORGET_BELOW);
      this.now += hours;
      this.context = {};
      return [];
    }
    const report: string[] = [];
    const touched = new Set<string>();
    const replays = Math.max(1, Math.round((P.REPLAYS_PER_NIGHT * hours) / P.SLEEP_HOURS));   // 睡眠が短いと、再生できる回数も減る
    for (const ep of this.hippocampus) {
      const strength = this.hippoStrength(ep);
      if (strength < P.HIPPO_FORGET_BELOW) continue;
      // 海馬の記録を何度も再生し、皮質のリンクを強める
      for (const [a, b] of pairs(ep.concepts)) {
        const key = linkKey(a, b);
        const link = (this.cortex[key] ??= { w: 0, recent: [], n: 0, first: null });
        for (let i = 0; i < replays; i++) link.w += P.CONSOLIDATE_LR * strength * (1 - link.w);
        this.use(link);
        touched.add(key);
      }
      ep.nights += 1;
      report.push(`再生: ${ep.concepts.join(' / ')} (${ep.nights}晩目)`);
    }
    // 古い記憶も混ぜて再生し、新しい知識で上書きされないようにする
    // 取り出しやすい記憶ほど再生されやすく、使わない記憶は再生されずに薄れていく
    for (const [key, link] of Object.entries(this.cortex)) {
      if (!touched.has(key)) link.w += P.INTERLEAVE_LR * this.access(link) * (1 - link.w);
    }
    // シナプス整理: 全体を少し弱め、弱すぎるリンクは消す
    for (const key of Object.keys(this.cortex)) {
      this.cortex[key].w *= P.DOWNSCALE;
      if (this.cortex[key].w < P.PRUNE_BELOW) {
        delete this.cortex[key];
        report.push(`消失: ${key.replace('\t', ' - ')}`);
      }
    }
    // 十分に再生された記録・薄れた記録は海馬から離れる
    const kept: Episode[] = [];
    for (const ep of this.hippocampus) {
      if (ep.nights >= P.NIGHTS_TO_INDEPENDENCE) report.push(`皮質へ移行完了: ${ep.concepts.join(' / ')}`);
      else if (this.hippoStrength(ep) < P.HIPPO_FORGET_BELOW) report.push(`海馬から忘却: ${ep.concepts.join(' / ')}`);
      else kept.push(ep);
    }
    this.hippocampus = kept;
    this.rev++;
    this.now += hours;
    this.context = {};   // 起きたら作業中の文脈はリセット
    return report;
  }

  wait(hours: number): void {
    this.now += hours;
  }

  // ---- 思い出す ----

  /** 文から概念を見つける(言語野の代わり)。概念 → 手がかりの強さ
   *
   * 1. 名前がそのまま含まれていれば、強さ 1.0(× 情報量)
   * 2. 意味が近い概念も、埋め込みの類似度に応じて弱めに活性化する
   *    (類似度はその文の中での相対的な高さで判断する。無関係な文では何も選ばない) */
  async perceive(text: string): Promise<Record<string, number>> {
    const found: Record<string, number> = {};
    const low = text.toLowerCase();
    const bounds = wordBoundaries(text);
    const used = new Uint8Array(low.length);
    const known = this.knownConcepts();
    for (const name of [...known].sort((a, b) => b.length - a.length)) {
      const n = name.toLowerCase();
      if (!n) continue;
      let i = low.indexOf(n);
      while (i >= 0) {
        const end = i + n.length;
        // 語の境目にそろっていて、まだ他の名前に使われていない部分だけ一致とみなす
        // (「マネーフォワード」の中の「ワード」は一致にしない)
        if (bounds.has(i) && bounds.has(end) && !used.subarray(i, end).some((u) => u)) {
          found[name] = this.informativeness(name);
          used.fill(1, i, end);
          break;
        }
        i = low.indexOf(n, i + 1);
      }
    }
    if (this.embedder && known.size) {
      const names = [...known].sort();
      const arr = (await this.embedder.ensure(names)) ? await this.embedder.similarities(text, names) : null;
      if (arr && arr.length) {
        const [mean, std0] = meanStd(arr);
        const std = std0 || 1;
        const top = argsortDesc(arr).slice(0, P.SEM_TOPK).map((i) => [names[i], arr[i]] as [string, number]);
        const zTop = (top[0][1] - mean) / std;
        // 一番近い概念の強さは、どれだけ際立っているか(相対)と、類似度そのものの高さ(絶対)の両方で決める
        let base = P.SEM_MAX * Math.min(1, Math.max(0, (zTop - P.SEM_Z_MIN) / (P.SEM_Z_FULL - P.SEM_Z_MIN)));
        base *= Math.min(1, Math.max(0, (top[0][1] - P.SEM_ABS_MIN) / (P.SEM_ABS_FULL - P.SEM_ABS_MIN)));
        if (Object.keys(found).length) base *= P.SEM_WITH_EXACT;   // 名前が直接出ているなら、連想は控えめに
        for (const [name, s] of top) {
          const z = (s - mean) / std;
          if (base <= 0 || z <= P.SEM_Z_MIN) break;
          // 2位以下は、1位との差が開くほど急に弱くなる
          const a = base * ((z - P.SEM_Z_MIN) / (zTop - P.SEM_Z_MIN)) ** 2 * this.informativeness(name);
          if (a > (found[name] ?? 0)) found[name] = a;
        }
      }
    }
    return found;
  }

  /** 指示から思い出す(場所の答えと、活性化した概念)
   *
   * useContext=false: ワーキングメモリ(今の話題)を使わない(記憶そのものの力を測るとき)
   * reinforce=false : 思い出しても記憶を強めない・ワーキングメモリも変えない(試しに思い出すとき) */
  async recall(text: string, useContext = true, reinforce = true): Promise<{ cues: string[]; answer: Answer }> {
    const cueStrength = await this.perceive(text);
    this.lastCues = cueStrength;
    const cues = Object.keys(cueStrength).sort((a, b) => cueStrength[b] - cueStrength[a]);
    // 注意の分配(ACT-R): 注意の総量 W=1 を、指示の手がかりとワーキングメモリ(今考えていること)で分け合う。
    // 人は、指示語(「この」「それ」)があるときや、「直して」「続けて」のように
    // 何をするかの対象が省かれているときに、今の話題を指していると解釈する
    const total = Object.values(cueStrength).reduce((s, x) => s + x, 0);
    const omitted = contentWords(text).length === 0;
    const referential = omitted || DEICTIC.test(text);
    // 文脈を指していない指示では、文脈は注意を奪わず、思い出しやすさを少し変える(プライミング)だけ
    const budget = referential ? (omitted ? P.OMITTED_CONTEXT : P.DEICTIC_CONTEXT) : P.CONTEXT_MIN;
    const cueBudget = referential ? 1 - budget : 1;

    // 手がかりに向ける注意を、語による手がかりと、出来事の要旨による照合で分け合う
    const [episodic, gate] = ABLATE.has('gist') ? [new Map<string, number>(), 0] : await this.recallTraces(text);
    const gistBudget = cueBudget * P.GIST_SHARE * gate;
    const wordBudget = cueBudget - gistBudget;
    const scale = total ? wordBudget / Math.max(1, total) : 0;
    const attention = new Map(Object.entries(cueStrength).map(([c, s]) => [c, s * scale]));
    const reinstated = new Map([...episodic].map(([c, a]) => [c, gistBudget * a]));
    const wm = useContext && !ABLATE.has('context') ? this.workingMemory() : new Map<string, number>();
    const wmSum = Math.max(1, [...wm.values()].reduce((s, x) => s + x, 0));
    const context = new Map([...wm].filter(([c]) => !attention.has(c)).map(([c, w]) => [c, (budget * w) / wmSum]));

    const [actCue, hits] = this.activate(attention, reinstated);
    const [actCtx, ctxHits] = this.activate(context);
    let act: Map<string, number>;
    if (referential) {
      // 指示が今の話題を指している: 文脈から届いた記憶も、そのまま思い浮かぶ
      act = new Map();
      for (const n of new Set([...actCue.keys(), ...actCtx.keys()])) act.set(n, Math.min(1, (actCue.get(n) ?? 0) + (actCtx.get(n) ?? 0)));
      hits.push(...ctxHits);
    } else {
      // プライミング: 文脈は、手がかりから届いた記憶のうち今の話題に近いものを思い出しやすくするだけ
      // (天気の話をしているのに、直前の話題のフォルダを答えたりしない)
      const peak = Math.max(0, ...actCtx.values()) || 1;
      act = new Map([...actCue].map(([n, a]) => [n, Math.min(1, a * (1 + (P.PRIMING * (actCtx.get(n) ?? 0)) / peak))]));
    }
    const answer = this.reconstruct(act, cues);
    this.lastActivation = act;
    if (reinforce) this.reinforce(cues.filter((c) => cueStrength[c] >= P.STRONG_CUE), answer, hits);
    return { cues, answer };
  }

  // ---- 展望記憶(これからやること) ----

  /** やり取りから取り出した「これからやること」を覚える */
  intend(items: Intent[], concepts: string[], ref: string | null, when: string): void {
    items.forEach((it, i) => {
      this.intentions.push({ id: `${(ref ?? '').slice(0, P.SHORT_ID)}-${i + 1}`, ref, text: it.text, due: it.due, kind: it.kind,
        concepts, time: when, t: this.now, status: 'open' });
    });
  }

  /** 同じ件(概念が重なる)について完了したと分かったら、その「やること」を完了にする(再固定化) */
  settle(concepts: string[], done: boolean): void {
    if (!done) return;
    const mine = new Set(concepts.filter((c) => !isPlace(c)));
    for (const it of this.intentions) {
      if (it.status === 'open' && it.t < this.now) {
        const overlap = it.concepts.filter((c) => !isPlace(c) && mine.has(c)).length;
        if (overlap >= P.DONE_OVERLAP) {
          it.status = 'done';
          it.done_t = this.now;
        }
      }
    }
  }

  /** 内部時計の今を、暦の日付(YYYY-MM-DD)にする(いちばん新しい出来事の日時から数える) */
  today(): string | null {
    let latest: Trace | null = null;
    for (const tr of this.traces) if (tr.time && (!latest || tr.t > latest.t)) latest = tr;
    if (!latest?.time) return null;
    const t = Date.parse(latest.time) + (this.now - latest.t) * 3600_000;
    // 記録の日時はその場所の時刻(+09:00 など)で書いてあるので、同じずれで日付にする
    const off = latest.time.match(/([+-])(\d{2}):?(\d{2})$/);
    const shift = off ? (off[1] === '-' ? -1 : 1) * (Number(off[2]) * 60 + Number(off[3])) * 60_000 : 0;
    return new Date(t + shift).toISOString().slice(0, 10);
  }

  /** 未完了の「やること」を、期限の近さ・依頼との関係・新しさで並べる
   *
   * asked: 依頼が「やり残し」「次なにする」のように、これからやることを尋ねているか。
   * 尋ねていないとき(新しい会話の最初)は、期限のあるものと最近のものだけを出す。 */
  rankIntentions(act: Map<string, number>, asked: boolean): TodoView[] {
    const today = this.today();
    const open = this.intentions.filter((it) => it.status === 'open');
    const rel = open.map((it) => it.concepts.reduce((s, c) => s + (act.get(c) ?? 0), 0));
    const top = Math.max(0, ...rel) || 1;
    const ranked: [number, Intention][] = [];
    open.forEach((it, k) => {
      const age = (this.now - it.t) / 24;
      let urgency = 0;
      if (it.due && today) {
        const days = daysBetween(it.due, today);
        if (days < -P.INTENT_OVERDUE_DAYS) return;   // 期限を大きく過ぎたものは、もう追わない
        urgency = days <= 3 ? 1 : days <= 14 ? 0.7 : 0.3;
      } else if (age > P.INTENT_FORGET_DAYS) return;   // 期限のないものは、時間が経つと思い出さなくなる
      if (!asked && !it.due && age > P.INTENT_RECENT_DAYS) return;
      const score = P.INTENT_W_REL * rel[k] / top + P.INTENT_W_URGENCY * urgency + P.INTENT_W_RECENT * (age + 1) ** -0.5
        + (P.INTENT_KIND_BONUS[it.kind] ?? 0);
      ranked.push([score, it]);
    });
    ranked.sort((a, b) => b[0] - a[0]);
    const n = asked ? P.INTENTS_ASKED : P.INTENTS_SHOWN;
    return ranked.slice(0, n).map(([, it]) => ({ id: it.id, ref: it.ref, text: it.text, due: it.due, kind: it.kind,
      places: it.concepts.filter(isPlace).slice(0, 1) }));
  }

  // ---- 関連することを思い出す(依頼に差し込む記憶) ----

  /** 依頼に関連することを思い出す
   *
   * 記憶力の良い人が、依頼を聞いて関連することを思い出すのと同じように、
   *   出来事 : 関係のありそうな過去のやり取り(そのときの依頼と回答の抜粋)
   *   概念   : 関連する語やファイル
   *   場所   : 関連するフォルダ
   *   直近   : 新しい会話の最初なら、直前に何をしていたか
   * を思い出し、それぞれに確かさ(ACT-R の検索確率)を付ける。text は差し込み用の文章。
   * session: 今の会話の識別子。今の会話のやり取りは相手(Claude)がすでに覚えているので、思い出す対象から外す
   * cwd    : 今の作業フォルダ。人が「今いる場所」で、その場所の出来事を思い出しやすくなる(文脈依存記憶)のと同じく、
   *          同じ場所での出来事を思い出しやすくする */
  async recollect(text: string, opts: { useContext?: boolean; reinforce?: boolean; session?: string | null; cwd?: string | null } = {}): Promise<Recollection> {
    const { useContext = true, reinforce = true, session = null, cwd = null } = opts;
    const newConversation = Object.keys(this.context).length === 0;
    const { cues, answer } = await this.recall(text, useContext, false);
    const act = this.lastActivation;
    const cueSet = new Set(cues);

    const episodes = await this.rankEpisodes(act, text, session, cwd);
    const concepts = [...act].sort((a, b) => b[1] - a[1])
      .filter(([n, a]) => !isPlace(n) && !cueSet.has(n) && a >= P.RECALL_THRESHOLD).slice(0, P.CONCEPTS_SHOWN)
      .map(([n, a]) => ({ name: n, conf: retrievalProbability(a, P.RETRIEVAL_TAU, P.RETRIEVAL_S) }));
    const places = this.rankPlaces(act);
    let recent: EpisodeView[] = [];
    if (newConversation) {
      // 新しい会話の最初は、直前に何をしていたかを思い出しておく
      const others = this.traces.filter((tr) => session === null || tr.session !== session);
      recent = [...others].sort((a, b) => b.t - a.t).slice(0, P.RECENT_SHOWN).map((tr) => this.episodeView(tr, null));
    }
    let shown = episodes.filter((e) => (e.conf ?? 0) >= P.SHOW_CONF);
    if (session && reinforce) {
      // 同じ会話ですでに差し込んだ記憶は、相手(Claude)がもう知っているので繰り返さない
      const already = new Set(this.shown[session] ?? []);
      shown = shown.filter((e) => !already.has(e.ref));
    }
    // 展望記憶: 新しい会話の最初と、これからやることを尋ねられたときに思い出す
    const asked = asksTodo(text);
    const todos = newConversation || asked ? this.rankIntentions(act, asked) : [];
    // 確かな出来事を1つも思い出せないなら、語や場所だけを出しても当て推量になる(何も入れない判断)
    const abstain = P.ABSTAIN && !shown.length && !newConversation;
    const result: Recollection = {
      ranked: episodes.slice(0, P.EPISODES_SHOWN * 2), todos, episodes: shown.slice(0, P.EPISODES_SHOWN),
      concepts: abstain ? [] : concepts.filter((c) => c.conf >= P.SHOW_CONF),
      places: abstain ? [] : places.filter((p) => p.conf >= P.PLACE_SHOW_CONF).slice(0, P.PLACES_SHOWN),
      recent, cues, answer, text: '',
    };
    result.text = recollectionText(result);
    if (reinforce) {
      this.reinforce(cues.filter((c) => this.lastCues[c] >= P.STRONG_CUE), answer, []);
      for (const e of result.episodes) {   // 思い出した出来事は、次も思い出しやすくなる
        const tr = e.trace!;
        tr.uses = [...(tr.uses ?? []), this.now].slice(-P.EXACT_USES);
      }
      if (session) {
        const refs = [...(this.shown[session] ?? []), ...[...result.episodes, ...result.recent].map((e) => e.ref).filter(Boolean)];
        delete this.shown[session];
        this.shown[session] = refs.slice(-P.SHOWN_PER_SESSION);
        const sessions = Object.keys(this.shown);
        for (const old of sessions.slice(0, Math.max(0, sessions.length - P.SHOWN_SESSIONS))) delete this.shown[old];   // 古い会話の分は忘れる
      }
    }
    for (const e of [...result.ranked, ...result.episodes, ...result.recent]) delete e.trace;
    return result;
  }

  /** 出来事(エピソード記憶の痕跡)を、ACT-R の活性 A = 基本活性化 + 拡散活性化 で並べる
   *
   *   要旨   : 依頼と出来事の要旨の意味の近さ
   *   概念   : 出来事に含まれる概念が、今どれだけ活性化しているか(語のつながり・文脈を経由)
   *   新しさ : 基本活性化 ln(Σ 経過時間^-d)。最近の出来事、最近思い出した出来事ほど高い
   *   細部   : 出来事の記録(返答の段落・コマンドなど)の中に、依頼とよく合う部分があるか
   *   キーワード・場所: 依頼の語が記録に文字どおりあるか、今の作業フォルダでの出来事か */
  async rankEpisodes(act: Map<string, number>, text: string, excludeSession: string | null = null, cwd: string | null = null): Promise<EpisodeView[]> {
    const sim = this.lastTraceSim;
    const traces = this.traces.filter((tr) => excludeSession === null || tr.session !== excludeSession);
    const refs = new Set(traces.map((tr) => tr.ref).filter((r): r is string => !!r));
    const details = this.store ? await this.store.search(text, refs) : new Map<string, Detail[]>();
    // キーワード: 依頼の語が、やり取りの記録に文字どおり含まれるか(型番・ファイル名・エラーのような手がかり)
    const keyword = this.store && !ABLATE.has('keyword') ? this.store.keywordSearch(this.queryTerms(text), refs) : new Map<string, number>();
    const topK = Math.max(0, ...keyword.values()) || 1;
    const conceptScore = traces.map((tr) => tr.concepts.reduce((s, c) => s + (act.get(c) ?? 0), 0));
    const topC = Math.max(0, ...conceptScore) || 1;
    const on = (name: string) => (ABLATE.has(name) ? 0 : 1);
    const ranked: [number, Trace, Detail[]][] = [];
    const features = new Map<Trace, Features>();
    traces.forEach((tr, k) => {
      // 依頼との類似度そのもので判断する(無関係な依頼でも、どれかは「他より際立って」近くなるため)
      const gist = Math.max(0, ((sim.get(tr) ?? 0) - P.GIST_ABS_MIN) / (P.GIST_ABS_FULL - P.GIST_ABS_MIN));
      const base = Math.log([tr.t, ...(tr.uses ?? [])].reduce((s, u) => s + (this.now - u + 1) ** -P.EP_DECAY, 0));
      const parts = (details.get(tr.ref ?? '') ?? []).filter((d) => d[0] >= P.CHUNK_SIM_MIN);
      const detail = parts.length ? Math.max(0, (parts[0][0] - P.CHUNK_SIM_MIN) / (P.CHUNK_SIM_FULL - P.CHUNK_SIM_MIN)) : 0;
      const here = cwd && tr.concepts.some((c) => isPlace(c) && atPlace(cwd, c)) ? 1 : 0;
      const kw = (keyword.get(tr.ref ?? '') ?? 0) / topK;
      const a = on('gist') * P.EP_W_GIST * gist + on('concept') * P.EP_W_CONCEPT * conceptScore[k] / topC
        + on('recency') * P.EP_W_BASE * base + on('detail') * P.EP_W_DETAIL * detail
        + P.EP_W_KEYWORD * kw + P.EP_W_PLACE * here;
      ranked.push([a, tr, parts]);
      features.set(tr, { gist, concept: conceptScore[k] / topC, base, detail, keyword: kw, place: here });
    });
    ranked.sort((a, b) => b[0] - a[0]);
    this.lastFeatures = features;   // 評価用
    this.lastCandidates = ranked.slice(0, P.CANDIDATES_KEPT).map(([, tr]) => tr);   // 評価用: 候補の段階で何が拾えていたか
    return ranked.slice(0, P.EPISODES_SHOWN * 4).map(([a, tr, parts]) => {
      const view = this.episodeView(tr, retrievalProbability(a, P.EP_TAU, P.EP_S));
      view.activation = a;
      // 依頼と関係する部分(依頼文そのものは見出しに出すので除く)
      view.details = parts.filter(([, k]) => k !== '依頼').slice(0, P.DETAILS_SHOWN).map(([s, k, t]) => ({ kind: k, text: t, sim: s }));
      // 手がかり: その出来事の概念のうち、今思い浮かんでいるもの(場所は別に出すので除く)
      view.hints = tr.concepts.filter((c) => !isPlace(c) && (act.get(c) ?? 0) > 0)
        .sort((x, y) => act.get(y)! - act.get(x)!).slice(0, P.HINTS_SHOWN);
      return view;
    });
  }

  /** キーワード検索に使う語: 依頼の名詞、英数字の語(型番・ファイル名など)、名前が一致した概念 */
  queryTerms(text: string): string[] {
    const terms = [...contentWords(text), ...(text.match(/[A-Za-z0-9_.\-]{3,}/g) ?? []),
      ...Object.entries(this.lastCues).filter(([c, s]) => s >= P.STRONG_CUE && !isPlace(c)).map(([c]) => c)];
    const deictic = new RegExp(`^(?:${DEICTIC.source})$`);
    return uniq(terms).filter((t) => Array.from(t).length >= 2 && !deictic.test(t)).slice(0, P.KEYWORD_TERMS);
  }

  episodeView(tr: Trace, conf: number | null): EpisodeView {
    const ref = tr.ref ?? '';
    const places = tr.concepts.filter(isPlace).map((c) => c + (this.placeExists(c) ? '' : '(今はない)')).slice(0, 2);
    return { trace: tr, conf, when: (tr.time ?? '').slice(0, 16).replace('T', ' '), details: [], id: ref.slice(0, P.SHORT_ID), ref,
      hints: [], prompt: tr.prompt ?? tr.concepts.join(' / '), reply: tr.reply ?? '', places, concepts: tr.concepts };
  }

  /** その場所(フォルダ)が今もあるか。このPCのパスでなければ、あるとみなす。確かめた結果は1分間使い回す
   *
   * 別のPCで覚えた場所(このPCにないユーザーのフォルダの中)は、なくなったのではなく確かめられないだけなので、あるとみなす。 */
  placeExists(place: string): boolean {
    if (!this.checkPlaces || !/^[A-Za-z]:/.test(place) || process.platform !== 'win32') return true;
    const hit = this.placeCache.get(place);
    if (hit && Date.now() - hit[1] <= 60_000) return hit[0];
    const user = place.match(/^([a-z]:\\users\\[^\\]+)/i);
    const otherPc = !!user && !isDir(user[1]);
    const ok = otherPc || isDir(place);
    this.placeCache.set(place, [ok, Date.now()]);
    return ok;
  }

  /** 関連する場所の確かさ = 思い出せる確率 × ほかの場所との競合で選ばれる確率(ACT-R の選択規則)
   *  競合はプロジェクト単位で考える(あるフォルダと、その中のフォルダは競合しない)。 */
  rankPlaces(act: Map<string, number>): PlaceView[] {
    // 今はもうない場所(移動・削除された)は、思い出しても役に立たないので弱める(干渉による忘却)
    const places = [...act].filter(([n, a]) => isPlace(n) && a > 0)
      .map(([n, a]) => [n, this.placeExists(n) ? a : a * P.STALE_PLACE] as [string, number]).sort((x, y) => y[1] - x[1]);
    const top = new Map<string, number>();   // プロジェクト(他のどの場所の中でもない場所) -> 活性
    const group = new Map<string, string>();
    for (const [n, a] of places) {
      let root = [...top.keys()].find((r) => isInside(n, r));
      if (root === undefined) {
        top.set(n, a);
        root = n;
      }
      group.set(n, root);
    }
    const z = [...top.values()].reduce((s, a) => s + Math.exp(a / P.CHOICE_T), 0) || 1;
    const choice = new Map([...top].map(([r, a]) => [r, Math.exp(a / P.CHOICE_T) / z]));
    return places.map(([n, a]) => ({ name: n, conf: choice.get(group.get(n)!)! * retrievalProbability(a, P.RETRIEVAL_TAU, P.RETRIEVAL_S),
      stale: !this.placeExists(n) }));
  }

  /** エピソード記憶の痕跡を、指示との意味の近さで呼び起こす
   *
   * 要旨が指示と近い出来事ほど、また新しい出来事ほど強く呼び起こされ、その出来事に含まれていた
   * 概念(場所を含む)が思い浮かぶ。どの痕跡も指示と際立って近くなければ、何も呼び起こさない。
   * 戻り値: [概念 → 呼び起こされた強さ(合計1), 呼び起こしの確かさ 0〜1] */
  async recallTraces(text: string): Promise<[Map<string, number>, number]> {
    this.lastTraceSim = new Map();
    const emb = this.embedder;
    const traces = emb ? this.traces.filter((tr) => emb.index.has(tr.gist)) : [];
    if (!traces.length || !emb) return [new Map(), 0];
    const sims = await emb.similarities(text, traces.map((tr) => tr.gist));
    if (!sims) return [new Map(), 0];
    traces.forEach((tr, i) => this.lastTraceSim.set(tr, sims[i]));
    const [mean, std0] = meanStd(sims);
    const order = argsortDesc(sims);
    const best = sims[order[0]];
    const z = (best - mean) / (std0 || 1);
    const gate = Math.min(1, Math.max(0, (z - P.GIST_Z_MIN) / (P.GIST_Z_FULL - P.GIST_Z_MIN)))
      * Math.min(1, Math.max(0, (best - P.GIST_ABS_MIN) / (P.GIST_ABS_FULL - P.GIST_ABS_MIN)));
    if (gate <= 0) return [new Map(), 0];
    const weights = new Map<number, number>();
    for (const k of order.slice(0, P.GIST_TOPK)) {
      const recency = (this.now - traces[k].t + 1) ** -P.GIST_DECAY;
      weights.set(k, Math.exp((sims[k] - best) / P.GIST_TEMP) * recency);
    }
    const norm = [...weights.values()].reduce((s, w) => s + w, 0);
    const out = new Map<string, number>();
    for (const [k, w] of weights) for (const c of traces[k].concepts) out.set(c, (out.get(c) ?? 0) + w / norm);
    return [out, gate];
  }

  /** 活性の発生源から、海馬のパターン補完と皮質の拡散で、各概念の活性を求める
   *  reinstated: エピソード記憶の痕跡から、すでに呼び起こされている概念 */
  activate(senders: Map<string, number>, reinstatedIn?: Map<string, number>): [Map<string, number>, Episode[]] {
    // 海馬のパターン補完: 手がかりが一部でも合えば、記録全体が呼び起こされる
    // 多くの記録に共通する語(「アプリ」など)は、どれか1つを呼び起こす力が弱い
    const count = new Map<string, number>();
    for (const ep of this.hippocampus) for (const c of ep.concepts) count.set(c, (count.get(c) ?? 0) + 1);
    // 同じ概念が複数の記録で呼び起こされたら、その分だけ強く思い浮かぶ(足し合わせる)。
    // 手がかりの力は記録の数で割ってあるので、足しても手がかり1つ分を超えない
    const sources = new Map(senders);
    for (const [c, a] of reinstatedIn ?? []) sources.set(c, (sources.get(c) ?? 0) + a);
    const hits: Episode[] = [];
    const reinstated = new Map<string, number>();
    for (const ep of ABLATE.has('hippo') ? [] : this.hippocampus) {
      let match = 0;
      for (const c of ep.concepts) if (senders.has(c)) match += senders.get(c)! / count.get(c)!;
      match = Math.min(1, match);
      if (match <= 0) continue;
      const a = match * this.hippoStrength(ep);
      hits.push(ep);
      for (const c of ep.concepts) reinstated.set(c, (reinstated.get(c) ?? 0) + a * 0.9);
    }
    for (const [c, a] of reinstated) sources.set(c, Math.max(sources.get(c) ?? 0, Math.min(1, a)));

    // 皮質の活性化拡散(ACT-R): 手がかりから、つながった概念へ1段だけ活性が流れる。
    // 流れる量は連合強度 S_ji = ln(P(i|j) / P(i))(Anderson の合理的分析)で決まる:
    //   P(i|j) … 手がかり j があるときに概念 i が必要になる確率(j のつながりの中での i の割合)
    //   P(i)   … 普段から i が必要になる確率(i のつながりの多さ)
    // つながりの多い手がかりは P(i|j) が小さく(ファン効果)、何にでも出てくる概念は
    // P(i) が大きいので、どちらも強度が下がる。手がかりと特に結びついた概念にだけ活性が届く。
    // ただし比率だけだと、一度たまたま一緒に出てきた珍しい概念が過大評価されるので、
    // 実際によく一緒に出てくるか(P(i|j))でも重み付けする
    const { adj, degree, total } = this.associations();
    const scale = Math.log(Math.max(2, adj.size));   // ln(lift) の最大値の目安で 0〜1 にそろえる
    const act = new Map(sources);
    if (ABLATE.has('spread')) return [act, hits];
    const from = new Map(senders);
    for (const [c, a] of sources) from.set(c, a);
    for (const [j, a] of from) {
      const dj = degree.get(j) ?? 0;
      if (dj <= 0) continue;
      for (const [i, eff] of adj.get(j)!) {
        if (sources.has(i) || senders.has(i) || eff <= 0) continue;
        const pij = eff / dj;
        const s = (pij ** P.ASSOC_FREQ_WEIGHT * Math.log(pij / (degree.get(i)! / total))) / scale;
        if (s > 0) act.set(i, Math.min(1, (act.get(i) ?? 0) + a * s));
      }
    }
    return [act, hits];
  }

  /** 活性の高い断片から答え(場所)を組み立てる */
  reconstruct(act: Map<string, number>, cues: string[] = []): Answer {
    const active = new Map([...act].filter(([, a]) => a > 0));
    const places = [...active].filter(([n]) => isPlace(n)).sort((x, y) => y[1] - x[1]);
    if (!places.length || places[0][1] < P.RECALL_THRESHOLD) return { place: null, about: [], confidence: 0, ambiguous: [], uncertain: false };
    let [best, level] = places[0];
    // 迷うかどうかは、どのプロジェクト(場所)かの段階で決める
    const ambiguous = places.slice(1).filter(([n, a]) => !isInside(n, best) && a >= level * P.AMBIGUITY_RATIO).map(([n]) => n);
    // 中のフォルダまで具体的に思い出せるなら(検索閾値を超えていれば)、そちらを答えにする
    const inner = places.filter(([n, a]) => isInside(n, best) && a >= Math.max(P.RETRIEVAL_TAU, level * P.INNER_PLACE_RATIO));
    if (inner.length) [best, level] = inner[0];
    // 確信度 = 思い出せる確率 × ほかの場所との競合で選ばれる確率(ACT-R)。2位と僅差なら下がる
    const conf = this.rankPlaces(act).find((p) => p.name === best)!.conf;
    // 「何の場所か」: 指示に含まれていた固有に近い名前を優先し、なければ
    // その場所とつながる概念のうち、他の場所とはつながらない固有のもの
    const linked = new Set<string>();
    const neighborsOf = (n: string) => {
      const out = new Set<string>();
      for (const [m] of this.associations().adj.get(n) ?? []) out.add(m);
      for (const ep of this.hippocampus) if (ep.concepts.includes(n)) for (const m of ep.concepts) if (m !== n) out.add(m);
      return out;
    };
    for (const n of neighborsOf(best)) if (active.has(n) && !isPlace(n)) linked.add(n);
    const cueSet = new Set(cues);
    const spec = (n: string): [number, number, number, number] => {
      const placesLinked = [...neighborsOf(n)].filter(isPlace).length;
      return [placesLinked > 2 ? 1 : 0, cueSet.has(n) ? 0 : 1, placesLinked, -active.get(n)!];
    };
    const cmp = (a: number[], b: number[]) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] - b[i]; return 0; };
    const about = [...linked].map((n) => [n, spec(n)] as const).sort((x, y) => cmp(x[1], y[1])).slice(0, 1).map(([n]) => n);
    return { place: best, about, confidence: conf, activation: level, ambiguous, uncertain: conf < 0.5 };
  }

  /** 想起による強化: 実際に使ったつながりが強まる */
  reinforce(cues: string[], answer: Answer, hippoHits: Episode[]): void {
    const used = new Set([...cues, ...answer.about]);
    if (answer.place) used.add(answer.place);
    for (const [a, b] of pairs([...used])) {
      const link = this.cortex[linkKey(a, b)];
      if (link) {
        link.w += P.RECALL_LR * (1 - link.w);
        this.use(link);
      }
    }
    for (const ep of hippoHits) if (ep.concepts.some((c) => used.has(c))) ep.t = this.now;   // 思い出した海馬の記録は鮮明さを取り戻す
    this.attend([...used]);
  }

  /** ワーキングメモリの各概念の強さ(0〜1)
   *  いちばん新しい話題が注意の焦点。それより前の話題は、1つさかのぼるごとに弱まる。時間が経つと全体が薄れる。 */
  workingMemory(): Map<string, number> {
    const order = [...new Set(Object.values(this.context).map(([, s]) => s))].sort((a, b) => b - a);
    return new Map(Object.entries(this.context).map(([c, [t, s]]) =>
      [c, P.FOCUS_DECAY ** order.indexOf(s) * 0.5 ** ((this.now - t) / P.CONTEXT_HALFLIFE)]));
  }

  /** ワーキングメモリに入れる。容量を超えたら古いものから押し出される */
  attend(concepts: string[]): void {
    this.seq += 1;
    for (const c of concepts) this.context[c] = [this.now, this.seq];
    const recent = Object.entries(this.context).sort((a, b) => b[1][1] - a[1][1]).slice(0, P.WM_CAPACITY);
    this.context = Object.fromEntries(recent);
  }
}

function isDir(p: string): boolean {
  try {
    return fs.statSync(p).isDirectory();
  } catch {
    return false;
  }
}
