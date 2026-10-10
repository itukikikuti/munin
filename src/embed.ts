// 意味のベクトルを計算して覚えておく(作り直せるキャッシュ)
//
// 鍵ごとにベクトルを1つ持つ。鍵は、概念ならその名前、依頼の要旨なら "Q:" + 文のハッシュ、
// 記録の部分なら "D:" + 文のハッシュ。保存は vectors/<名前>.f32(ベクトルを並べたもの)と
// vectors/<名前>.json(モデル・次元・鍵の並び)。足した分だけを後ろに書き足す。

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { VECTORS } from './config.ts';
import { embed, MODELS } from './llama.ts';

export const MAX_TEXT = 2000;   // 貼り付けたログなどで長すぎる文は、先頭だけで意味をとる
const BATCH = 64;
const QUERY_BATCH = 16;

/** 先頭 n 文字(サロゲートペアを分けない。Python の文字列の切り方と同じ) */
export function head(text: string, n = MAX_TEXT): string {
  if (text.length <= n) return text;
  return Array.from(text).slice(0, n).join('');
}

const sha1 = (t: string) => crypto.createHash('sha1').update(head(t), 'utf8').digest('hex');
/** 依頼の要旨のベクトルの鍵 */
export const gistKey = (text: string) => 'Q:' + sha1(text);
/** 検索される側の文のベクトルの鍵 */
export const docKey = (text: string) => 'D:' + sha1(text);

export class Embedder {
  readonly name: string;
  readonly model: string;
  readonly dim: number;
  readonly instruct: string;
  index = new Map<string, number>();
  available = true;
  private data: Float32Array;
  private rows = 0;
  private saved = 0;
  private lastQuery: [string, Float32Array] | null = null;

  /** name: 保存するファイルの名前。instruct: 検索する側の文(依頼)に付ける指示文 */
  constructor(name: string, model: string, instruct: string) {
    this.name = name;
    this.model = model;
    this.dim = MODELS[model].dim;
    this.instruct = instruct;
    this.data = new Float32Array(0);
    try {
      const meta = JSON.parse(fs.readFileSync(this.file('.json'), 'utf8'));
      if (meta.model === model && meta.dim === this.dim) {
        const buf = fs.readFileSync(this.file('.f32'));
        const n = Math.min(meta.names.length, Math.floor(buf.length / 4 / this.dim));
        this.data = new Float32Array(buf.buffer, buf.byteOffset, n * this.dim).slice();
        meta.names.slice(0, n).forEach((k: string, i: number) => this.index.set(k, i));
        this.rows = this.saved = n;
      }
    } catch {
      // まだない
    }
  }

  private file(ext: string): string {
    return path.join(VECTORS, this.name + ext);
  }

  /** 行 i のベクトル(コピーしない) */
  row(i: number): Float32Array {
    return this.data.subarray(i * this.dim, (i + 1) * this.dim);
  }

  vector(key: string): Float32Array | null {
    const i = this.index.get(key);
    return i === undefined ? null : this.row(i);
  }

  private add(key: string, v: Float32Array): void {
    if (this.index.has(key)) return;
    if ((this.rows + 1) * this.dim > this.data.length) {
      const grown = new Float32Array(Math.max(1024, this.rows * 2) * this.dim);
      grown.set(this.data.subarray(0, this.rows * this.dim));
      this.data = grown;
    }
    this.data.set(v.subarray(0, this.dim), this.rows * this.dim);
    this.index.set(key, this.rows++);
  }

  save(): void {
    if (this.saved === this.rows) return;
    fs.mkdirSync(VECTORS, { recursive: true });
    const fresh = this.data.subarray(this.saved * this.dim, this.rows * this.dim);
    const bytes = Buffer.from(fresh.buffer, fresh.byteOffset, fresh.byteLength);
    if (this.saved === 0) fs.writeFileSync(this.file('.f32'), bytes);
    else fs.appendFileSync(this.file('.f32'), bytes);
    const names = [...this.index.keys()];   // Map は入れた順(= 行の順)
    fs.writeFileSync(this.file('.json'), JSON.stringify({ model: this.model, dim: this.dim, names }));
    this.saved = this.rows;
  }

  private async fill(keys: string[], inputs: string[], batch: number, label: string, progress: boolean): Promise<boolean> {
    try {
      for (let i = 0; i < keys.length; i += batch) {
        const vs = await embed(this.model, inputs.slice(i, i + batch));
        vs.forEach((v, k) => this.add(keys[i + k], v));
        if (progress) process.stderr.write(`\r  ${label} ${Math.min(i + batch, keys.length)}/${keys.length}`);
        if (i % (batch * 50) === 0) this.save();
      }
      if (progress && keys.length) process.stderr.write('\n');
      this.save();
      return true;
    } catch (e) {
      this.available = false;
      process.stderr.write(`\n  埋め込みを計算できない: ${(e as Error).message}\n`);
      return false;
    }
  }

  /** まだ計算していない概念のベクトルを計算する(概念は指示文を付けない) */
  ensure(names: Iterable<string>, progress = false): Promise<boolean> {
    const missing = [...new Set(names)].filter((n) => !this.index.has(n));
    return this.fill(missing, missing, BATCH, '概念の埋め込み', progress);
  }

  /** まだ計算していない要旨(依頼の文)のベクトルを計算する */
  ensureGists(texts: Iterable<string>, progress = false): Promise<boolean> {
    const todo = new Map<string, string>();
    for (const t of texts) todo.set(gistKey(t), this.instruct + head(t));
    const missing = [...todo.keys()].filter((k) => !this.index.has(k));
    return this.fill(missing, missing.map((k) => todo.get(k)!), QUERY_BATCH, '要旨の埋め込み', progress);
  }

  /** まだ計算していない、検索される側の文(指示文を付けない)のベクトルを計算する */
  ensureDocs(texts: Iterable<string>, progress = false): Promise<boolean> {
    const todo = new Map<string, string>();
    for (const t of texts) todo.set(docKey(t), head(t));
    const missing = [...todo.keys()].filter((k) => !this.index.has(k));
    return this.fill(missing, missing.map((k) => todo.get(k)!), BATCH, '文の埋め込み', progress);
  }

  /** 文の要旨のベクトル(計算済みなら使い回す)。使えなければ null */
  async gist(text: string): Promise<Float32Array | null> {
    const key = gistKey(text);
    const have = this.vector(key);
    if (have) return have;
    if (this.lastQuery && this.lastQuery[0] === key) return this.lastQuery[1];
    try {
      const [q] = await embed(this.model, [this.instruct + head(text)]);
      this.lastQuery = [key, q];
      this.available = true;
      return q;
    } catch {
      this.available = false;
      return null;
    }
  }

  /** 覚える出来事の要旨のベクトルを保存し、鍵を返す */
  async rememberGist(text: string): Promise<string | null> {
    const key = gistKey(text);
    if (!this.index.has(key)) {
      const v = await this.gist(text);
      if (!v) return null;
      this.add(key, v);
    }
    return key;
  }

  /** ベクトル q と、各行の近さ(コサイン類似度。どちらも長さ1) */
  dots(q: Float32Array, rows: ArrayLike<number>): Float32Array {
    const out = new Float32Array(rows.length);
    const d = this.dim;
    const data = this.data;
    for (let k = 0; k < rows.length; k++) {
      const o = rows[k] * d;
      let s = 0;
      for (let j = 0; j < d; j++) s += data[o + j] * q[j];
      out[k] = s;
    }
    return out;
  }

  /** 文と、各鍵(概念・要旨)の意味の近さ。使えなければ null */
  async similarities(text: string, keys: string[]): Promise<Float32Array | null> {
    const q = await this.gist(text);
    if (!q) return null;
    return this.dots(q, keys.map((k) => this.index.get(k)!));
  }
}
