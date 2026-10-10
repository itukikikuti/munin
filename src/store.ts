// やり取りの記録(全文)を保存し、細かい単位で意味やキーワードで探せるようにする
//
// 記憶の痕跡(エピソード記憶)は、この記録への索引(鍵)だけを持つ。
// 海馬は記憶の中身そのものではなく、中身のありかを指す索引を持つ、という説(海馬インデックス理論)と同じ形。
//
//   記録     : 依頼の全文、Claude の返答の全文、ツールの実行(コマンド・ファイル)とその結果
//              (結果は先頭 RESULT_CHARS 文字まで)
//   チャンク : 記録を、依頼・返答の段落・コマンド・エラーなどの小さな単位に分けたもの。
//              それぞれに意味のベクトルを持たせ、依頼と関係する部分を探せるようにする

import { DatabaseSync } from 'node:sqlite';
import { docKey, Embedder } from './embed.ts';

export const CHUNK_MODEL = 'qwen3-embedding:0.6b';   // チャンクは数が多いので、軽くて速いモデルを使う
const CHUNK_INSTRUCT = "Instruct: Given a user's request about their work, retrieve passages from past work sessions that are relevant to it\nQuery: ";
export const RESULT_CHARS = 2000;   // ツールの結果は先頭だけ残す
const PARA_MIN = 80;                // 段落が短すぎるときは、次の段落とつなげる
const PARA_MAX = 600;               // 段落が長すぎるときは分ける
const PROMPT_PARTS = 3;             // 依頼が長い(ログの貼り付けなど)ときに、チャンクにする先頭の部分の数
const CHUNK_CANDIDATES = 300;       // 依頼と近いチャンクを、上位いくつまで見るか
const KEYWORD_MAX_HITS = 2000;      // これ以上のチャンクに出てくる語は、ありふれていて手がかりにならない

/** 文字(コードポイント)単位で切る(Python の文字列の切り方と同じにして、保存済みのベクトルの鍵を合わせる) */
export function cut(text: string, from: number, to?: number): string {
  if (to === undefined && from === 0) return text;
  const cs = Array.from(text);
  return cs.slice(from, to).join('');
}
const cpLen = (t: string) => Array.from(t).length;

export interface Step {
  kind: 'text' | 'tool' | 'thinking' | 'interrupt';
  text?: string;
  name?: string;
  summary?: string;
  result?: string;
  error?: boolean;
  said?: string;
}

export interface Rec {
  prompt: string;
  steps: Step[];
  time: string;
  cwd: string | null;
  src: { file?: string; uuid?: string; app?: string };
}

/** 文章を段落ごとのかたまりに分ける */
export function splitText(text: string, limit = PARA_MAX, minLen = PARA_MIN): string[] {
  const paras = text.split(/\n\s*\n/).map((p) => p.trim()).filter(Boolean);
  const out: string[] = [];
  let buf = '';
  for (const p of paras) {
    buf = buf ? `${buf}\n${p}` : p;
    const cs = Array.from(buf);
    if (cs.length >= minLen) {
      for (let i = 0; i < cs.length; i += limit) out.push(cs.slice(i, i + limit).join(''));
      buf = '';
    }
  }
  if (buf) out.push(cut(buf, 0, limit));
  return out;
}

/** ツールの実行を一言で表す */
export function toolSummary(name: string, inp: any): string {
  inp = inp || {};
  if (name === 'Bash' || name === 'PowerShell') {
    const desc = inp.description || '';
    return `コマンド: ${cut(String(inp.command || ''), 0, 400)}` + (desc ? `(${desc})` : '');
  }
  if (name === 'Edit' || name === 'Write' || name === 'NotebookEdit')
    return `${name === 'Write' ? '作成' : '編集'}: ${inp.file_path || inp.notebook_path || ''}`;
  if (name === 'Read') return `読み込み: ${inp.file_path || ''}`;
  if (name === 'Grep' || name === 'Glob') return `検索(${name}): ${inp.pattern || ''} ${inp.path || ''}`;
  if (name === 'WebSearch' || name === 'WebFetch') return `Web(${name}): ${inp.query || inp.url || ''}`;
  return `${name}: ${cut(JSON.stringify(inp), 0, 300)}`;
}

/** 記録をチャンク(種類, 文)に分ける */
export function makeChunks(rec: Rec): [string, string][] {
  const chunks: [string, string][] = splitText(rec.prompt).slice(0, PROMPT_PARTS).map((p) => ['依頼', p]);
  for (const step of rec.steps) {
    if (step.kind === 'text') {
      for (const p of splitText(step.text || '')) chunks.push(['返答', p]);
    } else if (step.kind === 'tool') {
      if (step.said) for (const p of splitText(step.said)) chunks.push(['ユーザー', p]);   // 質問への答えなど(ユーザー自身の言葉)
      chunks.push(['操作', step.summary || '']);
      if (step.error && step.result) chunks.push(['エラー', `${step.summary}\n${cut(step.result, 0, PARA_MAX)}`]);
    }
  }
  return chunks.filter(([, t]) => cpLen(t.trim()) >= 4);
}

export type Detail = [number, string, string];   // [類似度, 種類, 文]

export class Store {
  readonly db: DatabaseSync;
  readonly embedder: Embedder;
  private index: { owner: Int32Array; rows: Int32Array; keys: string[]; chunks: { key: string; kind: string; text: string }[] } | null = null;
  private chunkCount = -1;

  constructor(path: string, embedder?: Embedder) {
    this.db = new DatabaseSync(path);
    this.db.exec('PRAGMA journal_mode=WAL');
    this.db.exec('CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, data TEXT)');
    this.db.exec('CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY, key TEXT, kind TEXT, text TEXT)');
    this.db.exec('CREATE INDEX IF NOT EXISTS chunks_key ON chunks(key)');
    this.embedder = embedder ?? new Embedder('chunks', CHUNK_MODEL, CHUNK_INSTRUCT);
  }

  // ---- 保存 ----

  /** 記録を保存し、チャンクに分ける(同じ鍵で中身も同じならそのまま。取り込み方を変えて中身が変わったら置き換える) */
  put(key: string, rec: Rec): void {
    const data = JSON.stringify(rec);
    const row = this.db.prepare('SELECT data FROM records WHERE key=?').get(key) as { data: string } | undefined;
    if (row && row.data === data) return;
    if (row) this.db.prepare('DELETE FROM chunks WHERE key=?').run(key);
    this.db.prepare('INSERT OR REPLACE INTO records VALUES (?, ?)').run(key, data);
    const ins = this.db.prepare('INSERT INTO chunks (key, kind, text) VALUES (?, ?, ?)');
    for (const [k, t] of makeChunks(rec)) ins.run(key, k, t);
    this.index = null;
  }

  /** まとめて書く(速くするため、1つの取引にする) */
  transaction(fn: () => void): void {
    this.db.exec('BEGIN');
    try {
      fn();
      this.db.exec('COMMIT');
    } catch (e) {
      this.db.exec('ROLLBACK');
      throw e;
    }
  }

  delete(key: string): void {
    this.db.prepare('DELETE FROM chunks WHERE key=?').run(key);
    this.db.prepare('DELETE FROM records WHERE key=?').run(key);
    this.index = null;
  }

  keys(): string[] {
    return (this.db.prepare('SELECT key FROM records').all() as { key: string }[]).map((r) => r.key);
  }

  record(key: string): Rec | null {
    const row = this.db.prepare('SELECT data FROM records WHERE key=?').get(key) as { data: string } | undefined;
    return row ? JSON.parse(row.data) : null;
  }

  /** 鍵の先頭部分から記録の鍵を探す(1件に決まらなければ null) */
  find(short: string): string | null {
    const rows = this.db.prepare('SELECT key FROM records WHERE key LIKE ? LIMIT 2').all(short + '%') as { key: string }[];
    return rows.length === 1 ? rows[0].key : null;
  }

  /** まだベクトルのないチャンクを計算する */
  async ensureVectors(progress = false): Promise<boolean> {
    const texts = (this.db.prepare('SELECT text FROM chunks').all() as { text: string }[]).map((r) => r.text);
    const ok = await this.embedder.ensureDocs(texts, progress);
    this.index = null;
    return ok;
  }

  // ---- 検索 ----

  private build(): void {
    const all = this.db.prepare('SELECT key, kind, text FROM chunks ORDER BY id').all() as { key: string; kind: string; text: string }[];
    const chunks: typeof all = [];
    const rows: number[] = [];
    for (const c of all) {
      const i = this.embedder.index.get(docKey(c.text));
      if (i === undefined) continue;
      chunks.push(c);
      rows.push(i);
    }
    const keys = [...new Set(chunks.map((c) => c.key))].sort();
    const keyNo = new Map(keys.map((k, i) => [k, i]));
    this.index = { owner: Int32Array.from(chunks.map((c) => keyNo.get(c.key)!)), rows: Int32Array.from(rows), keys, chunks };
  }

  /** キーワード検索の索引(SQLite の全文検索 FTS5、trigram 方式)を、チャンクに合わせて作る */
  ensureKeywordIndex(): void {
    this.db.exec("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, content='chunks', content_rowid='id', tokenize='trigram')");
    // 索引に入っている文の数と最後の id(外部の表を中身にしているので、索引そのものの表で数える)
    const f = this.db.prepare('SELECT count(*) n, max(id) last FROM chunks_fts_docsize').get() as { n: number; last: number | null };
    const c = this.db.prepare('SELECT count(*) n, max(id) last FROM chunks').get() as { n: number; last: number | null };
    if (f.n !== c.n || f.last !== c.last) this.db.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')");
    this.chunkCount = c.n;
  }

  /** 語が文字どおり含まれるチャンクを探す(型番・ファイル名・エラーのような、意味より字面が大事な手がかり用)
   *
   * 3文字以上の語は全文検索の索引で、2文字の語(「受注」「波形」など)は部分一致で探す。
   * 多くのチャンクに出てくる語ほど重みを下げる(IDF)。戻り値: 記録の鍵 → そのやり取りのチャンクの中で一番高い点 */
  keywordSearch(terms: string[], allowed: Set<string>): Map<string, number> {
    const best = new Map<string, number>();
    if (!terms.length || !allowed.size) return best;
    if (this.chunkCount < 0) this.ensureKeywordIndex();
    const n = Math.max(1, this.chunkCount);
    const hits = new Map<number, [string, number]>();
    const fts = this.db.prepare('SELECT c.id id, c.key key FROM chunks_fts f JOIN chunks c ON c.id = f.rowid WHERE chunks_fts MATCH ? LIMIT ?');
    const like = this.db.prepare('SELECT id, key FROM chunks WHERE text LIKE ? LIMIT ?');
    for (const term of new Set(terms)) {
      const rows = (cpLen(term) >= 3
        ? fts.all('"' + term.replace(/"/g, '""') + '"', KEYWORD_MAX_HITS)
        : like.all(`%${term}%`, KEYWORD_MAX_HITS)) as { id: number; key: string }[];
      if (!rows.length || rows.length >= KEYWORD_MAX_HITS) continue;   // どこにでも出てくる語は手がかりにならない
      const idf = Math.log(n / rows.length);
      for (const { id, key } of rows) {
        if (!allowed.has(key)) continue;
        const h = hits.get(id);
        hits.set(id, [key, (h ? h[1] : 0) + idf]);
      }
    }
    for (const [key, sc] of hits.values()) best.set(key, Math.max(best.get(key) ?? 0, sc));
    return best;
  }

  /** 依頼と各チャンクの意味の近さ。allowed(記録の鍵の集合)に含まれる記録のチャンクだけを対象にする
   *  戻り値: 記録の鍵 → [類似度, 種類, 文](近い順) */
  async search(text: string, allowed: Set<string>): Promise<Map<string, Detail[]>> {
    const out = new Map<string, Detail[]>();
    if (!this.index) this.build();
    const { owner, rows, keys, chunks } = this.index!;
    if (!rows.length || !allowed.size) return out;
    const q = await this.embedder.gist(text);
    if (!q) return out;
    const ok = keys.map((k) => allowed.has(k));
    const idx: number[] = [];
    for (let i = 0; i < owner.length; i++) if (ok[owner[i]]) idx.push(i);
    if (!idx.length) return out;
    const sims = this.embedder.dots(q, idx.map((i) => rows[i]));
    const order = idx.map((_, k) => k).sort((a, b) => sims[b] - sims[a]).slice(0, CHUNK_CANDIDATES);
    for (const k of order) {
      const c = chunks[idx[k]];
      if (!out.has(c.key)) out.set(c.key, []);
      out.get(c.key)!.push([sims[k], c.kind, c.text]);
    }
    return out;
  }
}
