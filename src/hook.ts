// Claude Code のフック
//
//   SessionStart(会話の始まり)        : node src/hook.ts --session-start   準備ができているか確かめ、常駐サーバーを起こしておく
//   UserPromptSubmit(メッセージを送った): node src/hook.ts                  思い出したことを差し込む
//   Stop(返答が終わった)              : node src/hook.ts --learn           そのやり取りを覚える
//
// 記憶は常駐サーバー(server.ts)が持っていて、フックはそこへ問い合わせるだけ(毎回記憶を読み込むと時間がかかるため)。
// サーバーが動いていなければ起動し、その回は何も差し込まない。
// フックが失敗しても会話は止めない(何も差し込まずに終わる)。失敗に気づけるように logs/hook_errors.log に残す。

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { CODE, LOG_DIR, logLine, PORT, STATE } from './config.ts';
import { MODELS, modelReady, serverExe } from './llama.ts';
import { kuromojiReady } from './text.ts';
import { config } from './config.ts';

const BASE = `http://127.0.0.1:${PORT}`;
const TIMEOUT = 12_000;   // フック全体の制限時間(hooks.json の timeout)より短くする
const SYSTEM_MESSAGES = ['<task-notification', '<local-command', '<bash-', '<agent-message', '<cross-session-message', '<command-'];

/** 常駐サーバーを、画面を出さずに起動する(このフックが終わっても動き続けるように切り離す) */
export function startServer(): void {
  fs.mkdirSync(LOG_DIR, { recursive: true });
  const log = fs.openSync(path.join(LOG_DIR, 'server.log'), 'a');
  const child = spawn(process.execPath, ['--no-warnings', path.join(CODE, 'src', 'server.ts')],
    { cwd: CODE, detached: true, stdio: ['ignore', log, log], windowsHide: true });
  child.unref();
}

export async function serverUp(): Promise<boolean> {
  try {
    return (await fetch(`${BASE}/api/status`, { signal: AbortSignal.timeout(2000) })).ok;
  } catch {
    return false;
  }
}

/** 動かすのに足りないもの(なければ空) */
export function problems(): string[] {
  const out: string[] = [];
  if (!serverExe()) out.push('埋め込みを計算する llama.cpp がまだない');
  if (!kuromojiReady()) out.push('日本語を解析する kuromoji の辞書がまだない');
  const model = config().embed_model || 'qwen3-embedding:8b';
  for (const id of new Set([model, 'qwen3-embedding:0.6b'])) if (!modelReady(id)) out.push(`埋め込みモデル ${id}(${(MODELS[id].size / 1e9).toFixed(1)}GB)がまだない`);
  return out;
}

function emit(event: string, context: string): void {
  process.stdout.write(JSON.stringify({ hookSpecificOutput: { hookEventName: event, additionalContext: context } }));
}

async function readStdin(): Promise<any> {
  // Windows では標準入力を一度に読むと空になることがあるので、最後まで流れとして読み切る
  const chunks: Buffer[] = [];
  for await (const c of process.stdin) chunks.push(c as Buffer);
  const raw = Buffer.concat(chunks).toString('utf8');
  return raw.trim() ? JSON.parse(raw) : {};
}

async function sessionStart(): Promise<void> {
  const lacking = problems();
  if (lacking.length) {
    emit('SessionStart', 'munin(記憶のプラグイン)は、次の理由でまだ使えません。ユーザーに伝え、/munin:setup の実行をすすめてください。\n- ' + lacking.join('\n- '));
    return;
  }
  if (!(await serverUp())) startServer();   // 最初のメッセージで思い出せるように、先に起こしておく
  if (!fs.existsSync(STATE)) {
    emit('SessionStart', 'munin(記憶のプラグイン)は、これまでの会話から最初の記憶を作っています(会話の量によって数分〜数十分)。できるまでは思い出したことは差し込まれません。');
  }
}

async function post(api: string, body: unknown): Promise<any> {
  const r = await fetch(`${BASE}${api}`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body), signal: AbortSignal.timeout(TIMEOUT) });
  if (!r.ok) throw new Error(`HTTP ${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.json();
}

async function main(): Promise<void> {
  let event: any;
  try {
    event = await readStdin();
  } catch (e) {
    logLine('hook_errors.log', `入力を読めない: ${(e as Error).message}`);
    return;
  }
  const args = process.argv.slice(2);
  if (args.includes('--session-start')) return sessionStart();
  try {
    if (args.includes('--learn')) {
      // 返答が終わった: その会話の新しいやり取りを覚えてもらう(サーバーが裏で覚えるので、すぐ戻る)
      if (event.transcript_path) await post('/api/learn_session', { transcript_path: event.transcript_path });
      return;
    }
    // アプリが依頼の前に付けるシステムの注意書きは、ユーザーの言葉ではないので除く
    const prompt = String(event.prompt ?? '').replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim();
    if (!prompt || prompt.startsWith('/')) return;   // コマンドには記憶は要らない
    if (SYSTEM_MESSAGES.some((t) => prompt.startsWith(t))) return;   // 裏の作業の完了通知などは、ユーザーの依頼ではない
    const r = await post('/api/hook', { prompt, session_id: event.session_id, cwd: event.cwd, transcript_path: event.transcript_path });
    if (r.text) emit('UserPromptSubmit', r.text);
  } catch (e) {
    const msg = (e as Error).message;
    if (/ECONNREFUSED|fetch failed/i.test(msg + String((e as any).cause))) {
      if (!(await serverUp())) {
        startServer();
        logLine('hook_errors.log', 'サーバーが動いていなかったので起動した(今回は差し込みなし)');
        return;
      }
    }
    logLine('hook_errors.log', `問い合わせに失敗: ${msg}`);
  }
}

// 直接実行されたときだけ動く(ほかのファイルから読み込まれたときは、関数を貸すだけ)
if ((import.meta as any).main) await main();
