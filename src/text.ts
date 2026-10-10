// 文を語に分ける(日本語の形態素解析)
//
// kuromoji(IPADIC 辞書。Python 版で使っていた janome と同じ辞書)で品詞まで分ける。
// kuromoji はブラウザ用の1ファイル版と辞書を、最初の準備で取ってくる(版を固定し、SHA-256 で照合する)。
// まだ取ってきていないときは、Node.js に入っている Intl.Segmenter で代わりに区切る(品詞が分からないので、
// 漢字・カタカナ・英数字だけでできた語を名詞とみなす。精度は落ちる)。

import fs from 'node:fs';
import { createRequire } from 'node:module';
import path from 'node:path';
import { DATA } from './config.ts';
import type { Asset } from './llama.ts';

const KUROMOJI = 'https://cdn.jsdelivr.net/npm/kuromoji@0.1.2';
const KUROMOJI_DIR = path.join(DATA, 'runtime', 'kuromoji');

/** kuromoji と辞書 */
export const KUROMOJI_ASSETS: Asset[] = [
  ['kuromoji.js', 307967, 'ab0ac10f1c4a9b5e246e63e57b04ba14d3a6148cd7e8d62cb97a960d4f85a3e6'],
  ['dict/base.dat.gz', 3956825, '0803327762e1c93ca731e4319ab8343340f2806bb84941207782cde9d2d5a8eb'],
  ['dict/cc.dat.gz', 1692067, '02b7631be0d4de3a1a75cd9f9cc51536e4f94c9e6b389b813e06ba0f6e7de765'],
  ['dict/check.dat.gz', 3111633, '193ae0035fff6fe812b58d9ee730e7a7d7ee601d918481ce51075c58114f6cc9'],
  ['dict/tid.dat.gz', 1605820, 'd43d831cb6fb0f0a411739cd287a6d5e998e121a8daca614df14a81a0dcac586'],
  ['dict/tid_map.dat.gz', 1485576, '33efd5ffd87a70f669add093fa39dee44341d58f940844ef107c8fd98bb795b2'],
  ['dict/tid_pos.dat.gz', 5916009, '60dbfc99a6ab993f30c5dab648bec6ad7f9aaefa5c14e1843837d95e509f8895'],
  ['dict/unk.dat.gz', 10512, 'f7f991cdeb9bfd3e9c0e4577cc50ee0815a11c508cccd444a9d3ab3c81521100'],
  ['dict/unk_char.dat.gz', 306, '9a8e86fd9aff32d323fbb59f5a7006f05927a11f8173c90712cc56293aeb3225'],
  ['dict/unk_compat.dat.gz', 338, '50f60aa29bc2e86c2903ab8c825bb6fa604d2b294d96941c1d3924259791899d'],
  ['dict/unk_invoke.dat.gz', 1140, '6b210889548457c3006913afd12c8b525562255f2709e404604be9614a25e94c'],
  ['dict/unk_map.dat.gz', 1190, '6df12460e5477230bb6fd9641def918b699fc0a8868016b6c9f794488630509b'],
  ['dict/unk_pos.dat.gz', 10540, '5b183a29f281acc7e0542beca47b83f7985047c0a2d27e78a66f32276be5ad11'],
].map(([name, size, sha256]) => ({ name: name as string, size: size as number, sha256: sha256 as string, url: `${KUROMOJI}/${name}` }));

export const kuromojiPath = (a: Asset) => path.join(KUROMOJI_DIR, ...a.name.split('/'));

export function kuromojiReady(): boolean {
  return KUROMOJI_ASSETS.every((a) => {
    try {
      return fs.statSync(kuromojiPath(a)).size === a.size;
    } catch {
      return false;
    }
  });
}

interface Token { surface_form: string; pos: string; pos_detail_1: string; word_position: number }
let tokenizer: { tokenize(text: string): Token[] } | null = null;

/** kuromoji を読み込む(使う前に1回。取ってきていなければ何もしない) */
export async function initJapanese(): Promise<boolean> {
  if (tokenizer) return true;
  if (!kuromojiReady()) return false;
  // ブラウザ用の版は、辞書を XMLHttpRequest で読む。手元のファイルを読む形に置き換える
  const g = globalThis as any;
  g.XMLHttpRequest ??= class {
    url = '';
    status = 0;
    response: ArrayBuffer | null = null;
    responseType = '';
    onload: (() => void) | null = null;
    onerror: ((e: unknown) => void) | null = null;
    open(_method: string, url: string) { this.url = url; }
    send() {
      try {
        const buf = fs.readFileSync(this.url);
        this.response = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
        this.status = 200;
        this.onload?.();
      } catch (e) {
        this.status = 404;
        this.onerror?.(e);
      }
    }
  };
  const kuromoji = createRequire(import.meta.url)(path.join(KUROMOJI_DIR, 'kuromoji.js'));
  tokenizer = await new Promise((resolve, reject) =>
    kuromoji.builder({ dicPath: path.join(KUROMOJI_DIR, 'dict') + path.sep }).build((err: Error | null, t: any) => (err ? reject(err) : resolve(t))));
  return true;
}

// ---- kuromoji がないときの代わり(Intl.Segmenter) ----

const SEGMENTER = new Intl.Segmenter('ja', { granularity: 'word' });
const NOUNISH = /^[\p{Script=Han}\p{Script=Katakana}ー々〆ヶA-Za-z0-9０-９Ａ-Ｚａ-ｚ.\-]+$/u;

function segmentTokens(text: string): Token[] {
  return [...SEGMENTER.segment(text)].map((s) => ({ surface_form: s.segment, word_position: s.index + 1,
    pos: s.isWordLike && NOUNISH.test(s.segment) ? '名詞' : 'その他', pos_detail_1: /^[0-9０-９]+$/.test(s.segment) ? '数' : '一般' }));
}

function tokens(text: string): Token[] {
  return tokenizer ? tokenizer.tokenize(text) : segmentTokens(text);
}

/** 指示語(「この」「それ」など)。依頼が今の話題を指しているかの手がかり */
export const DEICTIC = /この|その|あの|これ|それ|あれ|ここ|そこ|あそこ|こっち|そっち|さっき|例の|今の/;
const DEICTIC_FULL = new RegExp(`^(?:${DEICTIC.source})$`);
const SKIP_HEAD = new Set(['非自立', '代名詞', '特殊', '接尾', '副詞可能', '数']);   // 複合名詞の先頭にしない名詞
const NOT_CONTENT = new Set(['代名詞', '非自立', '接尾', '数']);

/** 語の区切りの位置(文の先頭からの文字数)。概念の名前が語の途中に当たっていないかを確かめるのに使う */
export function wordBoundaries(text: string): Set<number> {
  const bounds = new Set<number>([0]);
  let pos = 0;
  for (const t of tokens(text)) {
    pos += t.surface_form.length;
    bounds.add(pos);
  }
  return bounds;
}

/** 複合名詞を取り出す(連続する名詞をつなげる) */
export function nounRuns(text: string): string[] {
  const out: string[] = [];
  let buf: string[] = [];
  for (const t of tokens(text)) {
    if (t.pos === '名詞' && (buf.length || !SKIP_HEAD.has(t.pos_detail_1))) {
      buf.push(t.surface_form);
      continue;
    }
    if (buf.length) out.push(buf.join(''));
    buf = [];
  }
  if (buf.length) out.push(buf.join(''));
  return out.map((w) => w.trim()).filter(Boolean);
}

/** 文の中の内容語(名詞。指示語などは除く) */
export function contentWords(text: string): string[] {
  return tokens(text).filter((t) => t.pos === '名詞' && !NOT_CONTENT.has(t.pos_detail_1) && !DEICTIC_FULL.test(t.surface_form))
    .map((t) => t.surface_form);
}

export function clip(text: string, limit: number): string {
  const t = String(text).split(/\s+/).filter(Boolean).join(' ');
  const cs = Array.from(t);
  return cs.length <= limit ? t : cs.slice(0, limit).join('') + '…';
}
