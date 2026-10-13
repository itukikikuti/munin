// Claude Code と Codex のセッション記録から記憶を作る(LLM は使わない)
//
// 1回のユーザー指示と、それに続く Claude の作業を1つの出来事として海馬に入れる。
//   場所     : 作業フォルダ(cwd)と、触ったファイルのあるプロジェクトのルート
//   話題の語 : ユーザーの発言と Claude の返答(全文)から取り出した名詞
//   ファイル : 編集・作成・読み込みしたファイルの名前(編集したものを優先)
// 実際の日時どおりに時間を進め、日付が変わるたびに眠る。

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import crypto from 'node:crypto';
import { Brain, P } from './brain.ts';
import { LIFE, RECORDS, STATE, config } from './config.ts';
import { Embedder } from './embed.ts';
import { extract, mentionsDone } from './intentions.ts';
import { RESULT_CHARS, Store, toolSummary, type Rec, type Step } from './store.ts';
import { initJapanese, nounRuns } from './text.ts';

export const PROJECTS = path.join(os.homedir(), '.claude', 'projects');
export const CODEX_SESSIONS = path.join(os.homedir(), '.codex', 'sessions');
const HOME = os.homedir();
const JST_MS = 9 * 3600_000;

// 1つの出来事として海馬が束ねる量には限りがあるので、出来事ごとに要素の数を絞る
const USER_TERMS = 6;           // 1回の指示から取り出す語の数
const REPLY_TERMS = 6;          // Claude の返答から取り出す語の数
const MAX_FILES = 5;
const MAX_PLACES = 4;
const MIN_DF = 2;               // 2回以上の出来事に出てくる語だけ使う(一度きりの語は雑音が多い)
const GENERIC_SESSION_RATIO = 0.3;   // 3割を超えるセッションに出てくる語は、ありふれていて手がかりにならない
const GENERIC_MIN_SESSIONS = 3;
const DAY_STARTS_AT = 4;        // 1日の区切り(時)
const PROMPT_EXCERPT = 120;     // 思い出したときに伝える、依頼の抜粋の長さ
const REPLY_EXCERPT = 200;      // 思い出したときに伝える、回答の抜粋の長さ(最後の返答 = まとめ)
const THINKING_CHARS = 4000;
export const SETTLE_SECONDS = 30 * 60;   // これより最近に書き換わったセッション記録の最後のやり取りは、まだ途中とみなす

/** 概念と依頼の意味に使う指示文(qwen3-embedding は、検索する側の文に指示文を付けると精度が上がる) */
export const QUERY_INSTRUCT = "Instruct: Given a user's request about their work, retrieve the concept names it refers to\nQuery: ";

const STOPWORDS = new Set(`
感じ 場合 状態 方法 必要 確認 修正 対応 問題 部分 内容 今回 以下 以上 一つ 二つ 全部 全体 結果 理由
意味 説明 自分 説明 今日 明日 昨日 最初 最後 途中 前回 次回 他 別 後 前 中 上 下 時 時間 日 回 個 件
ファイル フォルダ コード エラー 実行 作成 変更 追加 削除 使用 利用 設定 表示 機能 処理 動作 値 名前
ユーザー 質問 回答 返答 指示 お願い 了解 承知 ありがとう 可能 不要 不明 OK ok
`.split(/\s+/).filter(Boolean));
// 英語の機能語(Claude の英語の独り言やコマンドの出力に出てくる。どの記憶を指しているかの手がかりにならない)
const EN_STOPWORDS = new Set(`
the and for are but not you all any can had her was one our out has have him his how its let may new now
own see use way who did get got just than that this then them they there these those what when where which
while with will would could should into onto from your yours about after again also because been before
being both each even here more most much only other over same some such very want were why yet done make
let's i'm it's don't i'll we'll i've ok okay yes good great sure next first last still need now
`.split(/\s+/).filter(Boolean));

export interface Turn {
  t: string;                    // 発言の日時(ISO、UTC)
  cwd: string | null;
  user: string;
  session: string;              // セッション記録のファイル
  uuid: string | null;
  source: 'claude' | 'codex';
  reply: string[];
  said: string[];               // ユーザー自身の言葉(質問への答え・実行を断ったときの指示)
  edited: string[];
  read: string[];
  steps: Step[];
  pending?: Map<string, Step>;
  places?: string[];
  files?: string[];
  user_terms?: string[];
  reply_terms?: string[];
}

// ---- 語 ----

export function usable(w: string): boolean {
  const n = Array.from(w).length;
  if (n < 2 || n > 20 || STOPWORDS.has(w) || EN_STOPWORDS.has(w.toLowerCase())) return false;
  if (/^[\u3040-\u309f]+$/.test(w)) return false;           // ひらがなだけ(「こと」「やつ」など)
  if (/^[^\p{L}\p{M}]+$/u.test(w)) return false;             // 数字・記号だけ(文字を1つも含まない)
  if (/[\\/<>{}()\[\]=;:"'`|]/.test(w)) return false;        // パスやコードの断片
  if (/^[A-Za-z]{1,2}$/.test(w)) return false;
  return true;
}

/** 複合名詞を取り出す */
export function nouns(text: string): string[] {
  return nounRuns(text.replace(/[*#|`>~_\[\]]+/g, ' ')).map((w) => w.trim()).filter(usable);
}

// ---- 発言の取り出し ----

function textOf(content: any): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) return content.filter((b) => b?.type === 'text').map((b) => b.text ?? '').join('\n');
  return '';
}

const SYSTEM_TAGS = ['<task-notification', '<local-command-stdout', '<local-command-stderr', '<local-command-caveat',
  '<system-reminder', '<bash-stdout', '<bash-stderr', '<command-message',
  '<agent-message', '<cross-session-message'];   // サブエージェントや別の会話からの連絡
const WRAPPER_TAGS = /<\/?(?:pasted_content|artifact-[\w-]+)(?:\s[^<>]*)?>/g;
const INTERRUPTED = '[Request interrupted';
const USER_SAID = /the user said:\s*([\s\S]+)/;              // ツールの実行を断ったときの、ユーザーの言葉
const ANSWERS = /"((?:[^"\\]|\\.)*)"="((?:[^"\\]|\\.)*)"/g;  // 質問への答え("質問"="答え")

/** ユーザーの発言から、システムが差し込んだ部分を除く(何も残らなければ空)
 *
 * 完了通知・コマンドの出力・システムからの注意書きは、ユーザーの言葉ではないので捨てる。
 * アプリが依頼の前に付ける注意書き(<system-reminder>)は外し、残りを依頼とする。
 * 貼り付けた文章(<pasted_content>)などのタグは外して中身を残す。
 * コマンド(/goal など)は、引数があるときだけ「/名前 引数」として残す。 */
export function cleanPrompt(text: string): string {
  const s = text.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim();
  if (SYSTEM_TAGS.some((t) => s.startsWith(t))) return '';
  const m = s.match(/<command-name>\s*([\s\S]*?)\s*<\/command-name>/);
  if (m) {
    const args = (s.match(/<command-args>([\s\S]*?)<\/command-args>/)?.[1] ?? '').trim();
    return args ? `${m[1]} ${args}` : '';
  }
  return s.replace(WRAPPER_TAGS, '').trim();
}

/** ツールの結果の中にある、ユーザー自身の言葉(質問への答え、実行を断ったときの指示) */
function userWordsInResult(result: string): string {
  if (result.slice(0, 200).includes('answered')) return [...result.matchAll(ANSWERS)].map((m) => `「${m[1]}」→ ${m[2]}`).join('\n');
  return result.match(USER_SAID)?.[1].trim() ?? '';
}

function newTurn(t: string, cwd: string | null, text: string, session: string, uid: string | null, source: Turn['source'] = 'claude'): Turn {
  return { t, cwd, user: text, session, uuid: uid, source, reply: [], said: [], edited: [], read: [], steps: [], pending: new Map() };
}

async function* lines(file: string): AsyncGenerator<any> {
  const rl = readline.createInterface({ input: fs.createReadStream(file, { encoding: 'utf8' }), crlfDelay: Infinity });
  for await (const line of rl) {
    if (!line) continue;
    try {
      yield JSON.parse(line);
    } catch {
      // 壊れた行は飛ばす
    }
  }
}

function listFiles(dir: string, depth: number): string[] {
  const out: string[] = [];
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isFile() && e.name.endsWith('.jsonl')) out.push(p);
    else if (e.isDirectory() && depth > 0) out.push(...listFiles(p, depth - 1));
  }
  return out;
}

/** Claude Code のセッション記録(サブエージェントの記録は除く) */
export const claudeFiles = () => listFiles(PROJECTS, 1);
export const codexFiles = () => listFiles(CODEX_SESSIONS, 5);

/** セッション記録を読み、ユーザー指示ごとの出来事にまとめる
 *
 * ユーザーの言葉は、普通の発言のほかに、次の形でも残っている。どれも本人の判断や好みなので拾う。
 *   作業中に送った発言     : 添付(queued_command)として記録される → 新しい指示として扱う
 *   質問への答え           : ツール(AskUserQuestion)の結果として記録される → その出来事の中のユーザーの言葉
 *   実行を断ったときの指示 : ツールの結果(「the user said: …」)
 *   途中で止めたこと       : 「[Request interrupted …」 → その方向が違ったという合図として残す */
export async function readTurns(files: string[] = claudeFiles(), codex: string[] = codexFiles(), progress = false): Promise<Turn[]> {
  const turns: Turn[] = [];
  const seen = new Set<string>();   // セッションを再開・分岐すると履歴がコピーされるので、同じ発言は1回だけ数える
  for (const [i, file] of files.entries()) {
    if (progress) process.stderr.write(`\r  読み込み ${i + 1}/${files.length}`);
    let cur: Turn | null = null;
    const queued = new Set<string>();   // この会話で、作業中に送られた発言(あとで普通の発言としても記録されることがある)
    for await (const d of lines(file)) {
      if (d.isSidechain || d.isMeta || d.isCompactSummary) continue;
      const msg = d.message ?? {};
      if (d.type === 'attachment') {
        const a = d.attachment ?? {};
        if (a.type !== 'queued_command' || a.commandMode !== 'prompt') continue;
        const text = cleanPrompt(textOf(a.prompt));
        const key = d.uuid ?? `${a.timestamp}|${text}`;
        if (!text || seen.has(key) || queued.has(text)) continue;
        seen.add(key);
        queued.add(text);
        cur = newTurn(a.timestamp ?? d.timestamp, d.cwd ?? null, text, file, d.uuid ?? null);
        turns.push(cur);
        continue;
      }
      if (d.type === 'user' && cur && Array.isArray(msg.content)) {
        for (const c of msg.content) {   // ツールの結果を、そのツールの実行に結びつける
          if (c?.type === 'tool_result' && cur.pending!.has(c.tool_use_id)) {
            const step = cur.pending!.get(c.tool_use_id)!;
            cur.pending!.delete(c.tool_use_id);
            const result = textOf(c.content);
            step.result = Array.from(result).slice(0, RESULT_CHARS).join('');
            step.error = !!c.is_error;
            const said = userWordsInResult(result);
            if (said) {
              step.said = said;
              cur.said.push(said);
            }
          }
        }
      }
      if (d.type === 'user') {
        const raw = textOf(msg.content);
        if (raw.trimStart().startsWith(INTERRUPTED)) {
          if (cur) cur.steps.push({ kind: 'interrupt' });   // ユーザーが途中で止めた
          continue;
        }
        const text = cleanPrompt(raw);
        if (!text) continue;   // ツールの結果・コマンド出力・システムの通知は指示ではない
        const key = d.uuid ?? `${d.timestamp}|${text}`;
        if (seen.has(key)) {
          cur = null;   // 続く Claude の作業も、すでに数えた分
          continue;
        }
        seen.add(key);
        if (queued.has(text)) continue;   // 作業中に送った発言が、あとで普通の発言としても記録されたもの
        cur = newTurn(d.timestamp, d.cwd ?? null, text, file, d.uuid ?? null);
        turns.push(cur);
      } else if (d.type === 'assistant' && cur) {
        for (const b of msg.content ?? []) {
          if (b?.type === 'text' && (b.text ?? '').trim()) {
            cur.reply.push(b.text);
            cur.steps.push({ kind: 'text', text: b.text });
          } else if (b?.type === 'thinking' && (b.thinking ?? '').trim()) {
            // 考えた過程(なぜそうしたか)。記録には残すが、量が多いので検索の対象にはしない
            cur.steps.push({ kind: 'thinking', text: b.thinking.slice(0, THINKING_CHARS) });
          } else if (b?.type === 'tool_use') {
            const inp = b.input ?? {};
            const step: Step = { kind: 'tool', name: b.name, summary: toolSummary(b.name, inp) };
            cur.steps.push(step);
            cur.pending!.set(b.id, step);
            const fp = inp.file_path || inp.notebook_path;
            if (!fp) continue;
            const kind = ['Edit', 'Write', 'NotebookEdit'].includes(b.name) ? 'edited' : b.name === 'Read' ? 'read' : null;
            if (kind && !cur[kind].includes(fp)) cur[kind].push(fp);
          }
        }
      }
    }
  }
  if (progress && files.length) process.stderr.write('\n');
  turns.push(...(await readCodexTurns(codex)));
  for (const t of turns) delete t.pending;
  return turns.filter((t) => t.t);
}

const CODEX_AGENT_HISTORY = 'The following is the Codex agent history';

/** uuid v5(Python の uuid.uuid5(NAMESPACE_URL, name) と同じ) */
function uuid5(name: string): string {
  const ns = Buffer.from('6ba7b8119dad11d180b400c04fd430c8', 'hex');
  const h = crypto.createHash('sha1').update(Buffer.concat([ns, Buffer.from(name, 'utf8')])).digest();
  h[6] = (h[6] & 0x0f) | 0x50;
  h[8] = (h[8] & 0x3f) | 0x80;
  const x = h.subarray(0, 16).toString('hex');
  return `${x.slice(0, 8)}-${x.slice(8, 12)}-${x.slice(12, 16)}-${x.slice(16, 20)}-${x.slice(20)}`;
}

/** Codex のセッション記録を、Claude Code と同じ形の出来事にする
 *  完了した項目(item_completed)を正にする。承認の審査役などのサブエージェントの会話は除く。 */
async function readCodexTurns(files: string[]): Promise<Turn[]> {
  const turns: Turn[] = [];
  for (const file of [...files].sort()) {
    let cur: Turn | null = null;
    let sid = path.basename(file);
    let cwd: string | null = null;
    for await (const d of lines(file)) {
      const p = d.payload ?? {};
      if (d.type === 'session_meta') {
        sid = p.id || sid;
        cwd = p.cwd ?? null;
        if ((p.source && typeof p.source === 'object') || (p.thread_source != null && p.thread_source !== 'user')) break;
        continue;
      }
      if (d.type === 'turn_context') {
        cwd = p.cwd || cwd;
        continue;
      }
      if (d.type !== 'event_msg' || p.type !== 'item_completed') continue;
      const it = p.item ?? {};
      const kind = it.type;
      if (kind === 'UserMessage') {
        const text = cleanPrompt((it.content ?? []).map((c: any) => c.text ?? '').join('\n'));
        if (text.startsWith(CODEX_AGENT_HISTORY)) break;   // 古い版の審査役の会話
        if (!text) continue;
        // 古い版は項目の id が会話の中でしか一意でないので、会話の id と合わせて鍵にする
        cur = newTurn(d.timestamp, cwd, text, file, uuid5(`codex:${sid}:${it.id}`), 'codex');
        turns.push(cur);
      } else if (!cur) {
        continue;
      } else if (kind === 'AgentMessage') {
        const text = (it.content ?? []).map((c: any) => c.text ?? '').join('\n').trim();
        if (text) {
          cur.reply.push(text);
          cur.steps.push({ kind: 'text', text });
        }
      } else if (kind === 'Plan' && it.text) {
        cur.steps.push({ kind: 'text', text: it.text });
      } else if (kind === 'Reasoning') {
        const text = (it.summary_text ?? []).join('\n');
        if (text.trim()) cur.steps.push({ kind: 'thinking', text: text.slice(0, THINKING_CHARS) });
      } else if (kind === 'CommandExecution') {
        const cmd = Array.isArray(it.command) ? it.command.join(' ') : String(it.command ?? '');
        cur.steps.push({ kind: 'tool', name: 'Bash', summary: toolSummary('Bash', { command: cmd }),
          result: Array.from(String(it.aggregated_output ?? '')).slice(0, RESULT_CHARS).join(''),
          error: it.exit_code !== 0 && it.exit_code != null });
        for (const pc of it.parsed_cmd ?? []) if (pc.type === 'read' && pc.path && !cur.read.includes(pc.path)) cur.read.push(pc.path);
      } else if (kind === 'FileChange') {
        for (const fp of Object.keys(it.changes ?? {})) {
          cur.steps.push({ kind: 'tool', name: 'Edit', summary: toolSummary('Edit', { file_path: fp }) });
          if (!cur.edited.includes(fp)) cur.edited.push(fp);
        }
      } else if (kind === 'WebSearch') {
        cur.steps.push({ kind: 'tool', name: 'WebSearch', summary: toolSummary('WebSearch', { query: it.query ?? '' }) });
      } else if (kind === 'McpToolCall') {
        const name = `${it.server}.${it.tool}`;
        cur.steps.push({ kind: 'tool', name, summary: toolSummary(name, it.arguments ?? {}),
          result: Array.from(JSON.stringify(it.result ?? null)).slice(0, RESULT_CHARS).join('') });
      }
    }
  }
  return turns;
}

// ---- 場所 ----

/** どのプロジェクトでもない場所(ホーム、一時フォルダ、設定フォルダ) */
function ignoredPlace(p: string): boolean {
  const low = p.toLowerCase().replace(/[\\/]+$/, '');
  return low === HOME.toLowerCase() || low.includes('\\temp') || low.includes('\\appdata\\') || low.includes('\\.claude') || low.startsWith('/tmp');
}

/** パス → プロジェクトのルート(C:\dev\〜 や Obsidian の保管庫) */
function projectRoot(p: string): string | null {
  const m = p.match(/^(c:\\dev\\[^\\]+)/i);
  if (m) return m[1];
  const o = p.match(/^(.*\\documents\\obsidian)(\\|$)/i);
  return o ? o[1] : null;
}

const dirname = (p: string) => p.replace(/[\\/][^\\/]*$/, '');
const basename = (p: string) => p.split(/[\\/]/).pop() ?? '';

/** 作業フォルダと触ったファイルから、記憶する場所を決める
 *  編集したファイルは、そのファイルのあるフォルダまで覚える。読んだだけのファイルは、プロジェクトのルートだけ覚える。 */
function placesOf(cwd: string | null, edited: string[], read: string[]): string[] {
  const places: string[] = [];
  if (cwd && !ignoredPlace(cwd)) {
    places.push(cwd);
    const root = projectRoot(cwd);
    if (root) places.push(root);
  }
  for (const fp of edited) {
    const root = projectRoot(fp);
    if (root && !ignoredPlace(fp)) {
      const folder = dirname(fp);
      places.push(...(folder.toLowerCase() !== root.toLowerCase() ? [folder, root] : [root]));
    }
  }
  for (const fp of read) {
    const root = projectRoot(fp);
    if (root && !ignoredPlace(fp)) places.push(root);
  }
  return places;
}

/** 大文字小文字だけが違うパス(C:\dev\ET と C:\dev\et)を、よく使われる表記にそろえ、場所とファイル名を決める */
export function unifyCase(turns: Turn[]): void {
  const best = (counts: Map<string, number>) => {
    const b = new Map<string, string>();
    for (const [p, c] of counts) {
      const k = p.toLowerCase();
      if (!b.has(k) || c > counts.get(b.get(k)!)!) b.set(k, p);
    }
    return b;
  };
  const seen = new Map<string, number>();
  for (const t of turns) for (const p of [t.cwd ?? '', ...t.edited, ...t.read]) seen.set(p, (seen.get(p) ?? 0) + 1);
  const b1 = best(seen);
  const fix = (p: string | null) => (p ? b1.get(p.toLowerCase()) ?? p : p);
  for (const t of turns) {
    t.cwd = fix(t.cwd);
    t.edited = t.edited.map((p) => fix(p)!);
    t.read = t.read.map((p) => fix(p)!);
  }
  const roots = new Map<string, number>();
  for (const t of turns) {
    t.places = placesOf(t.cwd, t.edited, t.read);
    for (const p of t.places) roots.set(p, (roots.get(p) ?? 0) + 1);
  }
  const b2 = best(roots);
  for (const t of turns) {
    t.places = [...new Set(t.places!.map((p) => b2.get(p.toLowerCase())!))].slice(0, MAX_PLACES);
    t.files = [...new Set([...t.edited, ...t.read].map(basename).filter(Boolean))];
  }
}

// ---- 語の選び方 ----

/** どの語を概念として覚えるか(その時点までに見たやり取りだけから判断する。未来の頻度は知らない) */
export class Vocabulary {
  turns = 0;
  df = new Map<string, number>();    // 語ごとに、いくつの出来事に出てきたか
  sdf = new Map<string, number>();   // 語ごとに、いくつのセッションに出てきたか
  sessionTerms = new Map<string, Set<string>>();

  observe(t: Turn): void {
    this.turns += 1;
    const terms = new Set([...t.user_terms!, ...t.reply_terms!, ...t.files!]);
    for (const w of terms) this.df.set(w, (this.df.get(w) ?? 0) + 1);
    if (!this.sessionTerms.has(t.session)) this.sessionTerms.set(t.session, new Set());
    const seen = this.sessionTerms.get(t.session)!;
    for (const w of terms) if (!seen.has(w)) {
      this.sdf.set(w, (this.sdf.get(w) ?? 0) + 1);
      seen.add(w);
    }
  }

  usable(w: string): boolean {
    // 一度きりの語は雑音が多い。多くのセッションにまたがる語は、話題を表さないありふれた語
    const maxSdf = Math.max(GENERIC_MIN_SESSIONS, this.sessionTerms.size * GENERIC_SESSION_RATIO);
    return (this.df.get(w) ?? 0) >= MIN_DF && (this.sdf.get(w) ?? 0) <= maxSdf;
  }

  /** TF-IDF の高い語から k 個 */
  pick(terms: string[], k: number): string[] {
    const tf = new Map<string, number>();
    for (const w of terms) tf.set(w, (tf.get(w) ?? 0) + 1);
    const scored: [number, string][] = [];
    for (const [w, c] of tf) if (this.usable(w)) scored.push([c * Math.log(this.turns / this.df.get(w)!), w]);
    scored.sort((a, b) => b[0] - a[0] || (a[1] < b[1] ? 1 : a[1] > b[1] ? -1 : 0));
    return scored.slice(0, k).map(([, w]) => w);
  }
}

/** やり取りから名詞を取り出しておく */
function analyze(t: Turn): void {
  if (!t.user_terms) {
    t.user_terms = nouns([t.user, ...t.said].join('\n'));   // 質問への答えなども本人の言葉
    t.reply_terms = nouns(t.reply.join('\n'));
  }
}

// ---- 時間 ----

/** 日時(UTC の ISO)→ 日本時間の暦の値(ミリ秒) */
const jstMs = (iso: string) => Date.parse(iso) + JST_MS;
/** 日本時間の ISO 表記(2026-10-08T09:05:18.883+09:00) */
export const jstIso = (iso: string) => new Date(jstMs(iso)).toISOString().replace('Z', '+09:00');
/** 1日の区切りを午前4時とした日付(深夜の作業は前の日の続き) */
const dayOf = (ms: number) => new Date(ms - DAY_STARTS_AT * 3600_000).toISOString().slice(0, 10);
const midnight = (ms: number) => Math.floor(ms / 86400_000) * 86400_000;

/** 過去のやり取りを、実際の日時どおりに体験させる(日付が変わるたびに眠る) */
export class Life {
  brain: Brain;
  vocab = new Vocabulary();
  origin = 0;          // 体験を始めた日の 0時(日本時間の暦の値、ミリ秒)
  day = '';
  nights = 0;
  learned = 0;
  session: string | null = null;
  scanned: Record<string, number> = {};   // セッション記録のファイル -> 最後に読み切ったときの更新時刻(ミリ秒)

  constructor(brain: Brain, first?: Turn) {
    this.brain = brain;
    if (first) {
      const start = jstMs(first.t);
      this.origin = midnight(start);
      this.day = dayOf(start);
    }
  }

  save(file: string): void {
    const v = this.vocab;
    fs.writeFileSync(file, JSON.stringify({ turns: v.turns, df: Object.fromEntries(v.df), sdf: Object.fromEntries(v.sdf),
      session_terms: Object.fromEntries([...v.sessionTerms].map(([s, ts]) => [s, [...ts]])),
      origin_ms: this.origin, day: this.day, nights: this.nights, learned: this.learned, session: this.session, scanned: this.scanned }));
  }

  /** 保存した状態から続きを体験する(なければ null) */
  static restore(brain: Brain, file: string): Life | null {
    if (!fs.existsSync(file)) return null;
    const d = JSON.parse(fs.readFileSync(file, 'utf8'));
    if (d.origin_ms === undefined) return null;   // Python 版の形式は使えない(取り込み直す)
    const life = new Life(brain);
    life.vocab.turns = d.turns;
    life.vocab.df = new Map(Object.entries(d.df));
    life.vocab.sdf = new Map(Object.entries(d.sdf));
    life.vocab.sessionTerms = new Map(Object.entries(d.session_terms).map(([s, ts]) => [s, new Set(ts as string[])]));
    life.origin = d.origin_ms;
    life.day = d.day;
    life.nights = d.nights;
    life.learned = d.learned;
    life.session = d.session;
    life.scanned = d.scanned ?? {};
    return life;
  }

  hours(ms: number): number {
    return (ms - this.origin) / 3600_000;
  }

  /** そのやり取りの時刻まで時間を進める */
  advance(t: Turn): void {
    const at = jstMs(t.t);
    const b = this.brain;
    while (dayOf(at) > this.day) {
      // 日をまたいだ: その日の最後の作業のあとに眠る。
      // 次の作業まで十分空いていれば、起きた時刻が次の作業の直前になるように眠り始める
      const next = new Date(Date.parse(this.day + 'T00:00:00Z') + 86400_000);
      this.day = next.toISOString().slice(0, 10);
      b.now = Math.max(b.now, Math.min(this.hours(at) - P.SLEEP_HOURS, this.hours(next.getTime()) + DAY_STARTS_AT));
      // 次の作業までの間しか眠れない(徹夜に近い夜は短い睡眠になる)
      b.sleep(Math.max(0.5, Math.min(P.SLEEP_HOURS, this.hours(at) - b.now)));
      this.nights += 1;
    }
    b.now = Math.max(b.now, this.hours(at));
    // 新しい会話(セッション)が始まったら、前の会話で考えていたことは持ち越さない
    if (t.session !== this.session) {
      b.context = {};
      this.session = t.session;
    }
  }

  /** やり取りを1つの出来事(一緒に覚える概念の組)にする */
  episode(t: Turn): string[] {
    analyze(t);
    this.vocab.observe(t);
    const v = this.vocab;
    const concepts = [...t.places!, ...v.pick(t.user_terms!, USER_TERMS), ...v.pick(t.reply_terms!, REPLY_TERMS),
      ...t.files!.filter((f) => v.usable(f)).slice(0, MAX_FILES)];
    return [...new Set(concepts)];
  }

  /** やり取りを体験したあと、海馬に記録する(concepts: 先に作ってあれば、その出来事の概念) */
  async remember(t: Turn, concepts?: string[]): Promise<void> {
    concepts ??= this.episode(t);
    if (concepts.length < 2) return;
    const time = jstIso(t.t);
    const note = { prompt: excerpt(t.user, PROMPT_EXCERPT), reply: excerpt(t.reply.at(-1) ?? '', REPLY_EXCERPT),
      time, session: path.basename(t.session), ref: recordKey(t) };
    // このやり取りで終わったことがあれば、前に覚えた「やること」を完了にしてから、新しい「やること」を覚える
    this.brain.settle(concepts, mentionsDone(t.user + '\n' + t.reply.slice(-2).join('\n')));
    await this.brain.learn(concepts, t.user, note);   // 要旨は依頼文(その出来事が何についてだったか)
    this.brain.intend(extract(t.user, t.reply, time.slice(0, 10)), concepts, recordKey(t), time);
    this.learned += 1;
  }

  async experience(t: Turn): Promise<void> {
    this.advance(t);
    await this.remember(t);
  }
}

/** 思い出したときに伝える抜粋(Markdown の記号や改行を除いて、先頭から limit 文字) */
function excerpt(text: string, limit: number): string {
  const t = text.replace(/<[^>]+>/g, ' ').replace(/[*#|`>~_]+/g, '').replace(/\s+/g, ' ').trim();
  const cs = Array.from(t);
  return cs.length <= limit ? t : cs.slice(0, limit).join('') + '…';
}

/** やり取りの記録の鍵(発言のID。なければ会話と時刻) */
export function recordKey(t: Turn): string {
  return t.uuid || `${path.basename(t.session)}:${t.t}`;
}

export function buildRecord(t: Turn): Rec {
  return { prompt: t.user, steps: t.steps, time: t.t, cwd: t.cwd, src: { file: t.session, uuid: t.uuid ?? undefined, app: t.source } };
}

/** やり取りの記録(全文)を保存し、細かい単位の検索に使うベクトルを計算しておく */
export async function storeRecords(store: Store, turns: Turn[], progress = true): Promise<boolean> {
  store.transaction(() => {
    for (const t of turns) store.put(recordKey(t), buildRecord(t));
  });
  return store.ensureVectors(progress);
}

/** 保存してある記録から、出来事を作り直す(元のセッション記録が消えたとき用) */
function turnFromRecord(key: string, rec: Rec): Turn {
  const steps = rec.steps;
  const files = (prefix: string) => [...new Set(steps.filter((s) => s.kind === 'tool' && s.summary?.startsWith(prefix)).map((s) => s.summary!.slice(prefix.length)))];
  const src = rec.src ?? {};
  return { t: rec.time, cwd: rec.cwd, user: rec.prompt, session: src.file || key, uuid: key, source: (src.app as Turn['source']) ?? 'claude',
    reply: steps.filter((s) => s.kind === 'text').map((s) => s.text!), said: steps.filter((s) => s.said).map((s) => s.said!),
    edited: [...files('編集: '), ...files('作成: ')], read: files('読み込み: '), steps };
}

/** セッションを読み、時刻順に並べる
 *  元のセッション記録がもう残っていないやり取りは、保存してある記録から補う(Claude Code は古いセッション記録を自動で消すので、
 *  記憶の原本は store に持つ)。元の記録は残っているのに出来事にならなかった記録は、取り込み方を変えて除いたものなので消す。 */
export async function loadTurns(store: Store | null, progress = true): Promise<Turn[]> {
  const turns = await readTurns(claudeFiles(), codexFiles(), progress);
  if (store) {
    const have = new Set(turns.map(recordKey));
    const lost: Turn[] = [];
    store.transaction(() => {
      for (const k of store.keys()) {
        if (have.has(k)) continue;
        const rec = store.record(k)!;
        if (rec.src?.file && fs.existsSync(rec.src.file)) store.delete(k);
        else lost.push(turnFromRecord(k, rec));
      }
    });
    if (lost.length) process.stderr.write(`  元の記録が消えていたやり取り ${lost.length}件を、保存してある記録から補いました\n`);
    turns.push(...lost);
  }
  turns.sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
  unifyCase(turns);
  return turns;
}

/** 記憶を開く(概念と依頼の意味は 8b、記録の部分は 0.6b。GPU がなければ設定で両方 0.6b にできる) */
export function openBrain(state = STATE): Brain {
  const model = config().embed_model || 'qwen3-embedding:8b';
  return new Brain(state, new Embedder('meaning', model, QUERY_INSTRUCT), new Store(RECORDS));
}

/** セッション記録の新しいやり取りだけを、続きとして体験させる(使いながら覚える)
 *
 * すでに記録してあるやり取りは覚え直さない(記録の中身は、続きが書き足されていれば新しくする)。
 * settle: 最近書き換わったファイルの最後のやり取りは、まだ途中かもしれないので覚えない
 *         (返答が終わったときのフックから呼ぶときは、終わっていると分かっているので false)。 */
export async function learnNew(life: Life, files: string[] = [], codex: string[] = [], settle = false): Promise<number> {
  const brain = life.brain;
  const store = brain.store!;
  let turns = await readTurns(files, codex);
  if (!turns.length) return 0;
  turns.sort((a, b) => (a.t < b.t ? -1 : a.t > b.t ? 1 : 0));
  if (settle) {
    const recent = new Set([...files, ...codex].filter((p) => Date.now() - fs.statSync(p).mtimeMs < SETTLE_SECONDS * 1000));
    const last = new Map<string, Turn>();
    for (const t of turns) if (recent.has(t.session)) last.set(t.session, t);
    turns = turns.filter((t) => last.get(t.session) !== t);
  }
  const known = new Set(store.keys());
  const fresh = turns.filter((t) => !known.has(recordKey(t)));
  unifyCase(turns);
  await brain.embedder?.ensureGists(fresh.map((t) => t.user));
  await storeRecords(store, turns, false);   // 記録は、続きが書き足されていれば新しくする
  for (const t of fresh) await life.experience(t);
  if (brain.embedder && fresh.length) await brain.embedder.ensure(brain.knownConcepts());   // 新しく覚えた概念の埋め込み
  return fresh.length;
}

/** すべてのセッション記録から記憶を作り直す */
export async function importAll(): Promise<void> {
  const t0 = Date.now();
  await initJapanese();
  for (const f of [STATE, LIFE]) fs.rmSync(f, { force: true });
  const brain = openBrain();
  // 読む前の更新時刻を覚えておく(読んでいる間に書き足された分は、あとで続きとして覚える)
  const scanned = Object.fromEntries([...claudeFiles(), ...codexFiles()].map((p) => [p, fs.statSync(p).mtimeMs]));
  const turns = await loadTurns(brain.store);
  if (!turns.length) {
    console.log('セッション記録が見つからない');
    return;
  }
  console.log(`出来事 ${turns.length}件 (${turns[0].t.slice(0, 10)} 〜 ${turns.at(-1)!.t.slice(0, 10)})`);
  await brain.embedder!.ensureGists(turns.map((t) => t.user), true);   // 出来事の要旨のベクトルを、先にまとめて計算しておく
  await storeRecords(brain.store!, turns);                                 // やり取りの記録(全文)と、細かい単位の検索用のベクトル
  const life = new Life(brain, turns[0]);
  for (const [i, t] of turns.entries()) {
    if (i % 50 === 0) process.stderr.write(`\r  体験 ${i}/${turns.length}`);
    await life.experience(t);
  }
  process.stderr.write('\n');
  brain.context = {};
  brain.save();
  life.scanned = scanned;
  life.save(LIFE);
  const known = brain.knownConcepts();
  const places = [...known].filter((c) => /^[a-z]:|^\//i.test(c)).length;
  console.log(`海馬に入れた出来事 ${life.learned}件 / 眠った回数 ${life.nights}晩`);
  console.log(`皮質のリンク ${Object.keys(brain.cortex).length}本 / 概念 ${known.size}個 / 場所 ${places}か所 / 海馬に残っている記録 ${brain.hippocampus.length}件`);
  // 思い出すときに使う概念の埋め込みを、先にまとめて計算しておく
  console.log((await brain.embedder!.ensure(known, true)) ? '埋め込みを計算しました' : '埋め込みを計算できませんでした(思い出すときに計算します)');
  brain.save();
  console.log(`所要時間 ${Math.round((Date.now() - t0) / 1000)}秒`);
}
