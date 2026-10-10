// Claude が自分で使える記憶の道具(MCP サーバー、標準入出力)
//
//   recall        : 言葉を手がかりに、関係する過去を思い出す(差し込まれた分では足りないときに、自分から探す)
//   recall_detail : 思い出したことに付いている鍵 [xxxxxxxx] から、そのやり取りの記録を開く
//
// 記憶は常駐サーバー(server.ts)が持っているので、ここはそこへ取り次ぐだけ。サーバーが動いていなければ起動して待つ。

import readline from 'node:readline';
import { PORT } from './config.ts';
import { serverUp, startServer } from './hook.ts';

const BASE = `http://127.0.0.1:${PORT}`;
const VERSION = '0.2.0';
const TOOLS = [
  { name: 'recall',
    description: 'これまでの会話(Claude Code と Codex のすべてのセッション)の記憶から、言葉を手がかりに関係する過去を思い出す。'
      + 'メッセージごとに自動で差し込まれる記憶では足りないとき、別の言い方や関係しそうな語で探し直すのに使う。'
      + '結果は手がかりと鍵の形で返るので、詳しく見たいものは recall_detail で開く。',
    inputSchema: { type: 'object', properties: {
      query: { type: 'string', description: '思い出したいことの手がかり(依頼文のような文でも、語の並びでもよい)' },
      cwd: { type: 'string', description: '今の作業フォルダ(あれば、その場所での出来事を思い出しやすくなる)' } }, required: ['query'] } },
  { name: 'recall_detail',
    description: '思い出したことに付いている鍵(例: [8351eb2f] の 8351eb2f)から、そのやり取りの記録を開く。依頼の全文、返答、使ったツールとその結果が見られる。',
    inputSchema: { type: 'object', properties: {
      key: { type: 'string', description: '記録の鍵(先頭8文字でよい)' },
      full: { type: 'boolean', description: '長い返答やツールの結果も省略せずに見る', default: false } }, required: ['key'] } },
];

/** 常駐サーバーに問い合わせる(動いていなければ起動して、準備ができるまで待つ) */
async function call(api: string, body?: unknown): Promise<any> {
  if (!(await serverUp())) {
    startServer();
    for (let i = 0; i < 120 && !(await serverUp()); i++) await new Promise((r) => setTimeout(r, 1000));
  }
  const r = await fetch(`${BASE}${api}`, body === undefined ? { signal: AbortSignal.timeout(60_000) }
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal: AbortSignal.timeout(60_000) });
  return r.json();
}

async function runTool(name: string, args: any): Promise<string> {
  if (name === 'recall') {
    const r = await call('/api/search', { query: args.query ?? '', cwd: args.cwd });
    if (r.importing) return 'まだ最初の記憶を作っている途中です。しばらくしてから使ってください。';
    return r.text || '関係しそうな記憶は思い出せませんでした。';
  }
  if (name === 'recall_detail') {
    const q = new URLSearchParams({ id: String(args.key ?? '').replace(/[\[\]\s]/g, '') });
    if (args.full) q.set('full', '1');
    const r = await call(`/api/record?${q}`);
    return r.text || r.error || '記録が見つかりませんでした。';
  }
  throw new Error(`知らない道具: ${name}`);
}

async function handle(msg: any): Promise<any> {
  const { method, id } = msg;
  if (id === undefined || id === null) return null;   // 通知(initialized など)には返事をしない
  let result: any;
  if (method === 'initialize') {
    result = { protocolVersion: msg.params?.protocolVersion ?? '2025-06-18', capabilities: { tools: {} }, serverInfo: { name: 'munin', version: VERSION } };
  } else if (method === 'tools/list') {
    result = { tools: TOOLS };
  } else if (method === 'tools/call') {
    try {
      result = { content: [{ type: 'text', text: await runTool(msg.params?.name, msg.params?.arguments ?? {}) }], isError: false };
    } catch (e) {   // 道具の失敗は、Claude に伝えて続ける
      result = { content: [{ type: 'text', text: `失敗しました: ${(e as Error).message}` }], isError: true };
    }
  } else if (method === 'ping') {
    result = {};
  } else {
    return { jsonrpc: '2.0', id, error: { code: -32601, message: `method not found: ${method}` } };
  }
  return { jsonrpc: '2.0', id, result };
}

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
  if (!line.trim()) continue;
  let reply: any;
  try {
    reply = await handle(JSON.parse(line));
  } catch {
    reply = { jsonrpc: '2.0', id: null, error: { code: -32700, message: 'parse error' } };
  }
  if (reply) process.stdout.write(JSON.stringify(reply) + '\n');
}
