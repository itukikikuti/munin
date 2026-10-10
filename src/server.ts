// munin の常駐サーバー(http://127.0.0.1:8770)
//
// 記憶を読み込んだまま待ち、Claude Code のフック・道具(MCP)・画面からの問い合わせに答える。
//   /api/hook           メッセージを送ったとき: 思い出したことを返し、何を出したかを logs/recall_log.jsonl に残す
//   /api/learn_session  返答が終わったとき: その会話の新しいやり取りを、裏で順番に覚える
//   /api/search         Claude が自分で探すとき(道具の recall)
//   /api/record         鍵から記録を開く(道具の recall_detail と画面)
//   /                   記憶を見る画面(ネットワーク図つき)
// 記憶がまだなければ、起動したときに、これまでのすべての会話から最初の記憶を作る。
// 1時間ごとに、書き足されたセッション記録(Codex の会話や、フックが動かなかった会話)を拾い直す。

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { Brain, isPlace, recollectionText } from './brain.ts';
import { appendJsonl, CODE, config, DATA, LIFE, LOG_DIR, logLine, PORT, RECORDS, stamp, STATE } from './config.ts';
import { render } from './detail.ts';
import { Embedder } from './embed.ts';
import { GraphMap } from './graph.ts';
import { asksTodo } from './intentions.ts';
import { claudeFiles, codexFiles, learnNew, Life, QUERY_INSTRUCT, SETTLE_SECONDS } from './sessions.ts';
import { Store } from './store.ts';
import { initJapanese } from './text.ts';

const AWAKE_HOURS = 16;
const HIPPO_SHOWN = 60;          // 画面の一覧に出す海馬の記録(新しい順)
const LINKS_SHOWN = 200;         // 画面の一覧に出す皮質のリンク(強い順)
const AUTOSAVE_MS = 10 * 60_000; // フックでの想起による強化を、まとめて保存する間隔
const CATCH_UP_MS = 3600_000;    // 書き足されたセッション記録を拾い直す間隔

let brain: Brain | null = null;
let life: Life | null = null;
let embedder: Embedder | null = null;
let store: Store | null = null;
let mtime: number | null = null;
// 図の作り直しが必要な変更があるたびに増える。起動時刻から始めて、再起動前のタイルをブラウザが使い回さないようにする
let version = Math.floor(Date.now() / 1000);
let liveSession: string | null = null;   // フックで最後に思い出した会話
let importing = !fs.existsSync(STATE);   // 記憶がまだない(初めて使う)なら、最初の記憶を作る
let dirty = false;                       // フックでの想起による強化が、まだ保存されていない
const graph = new GraphMap(path.join(DATA, 'layout.json'));
const learnQueue: string[] = [];

/** 記憶の読み書きを1つずつにする */
let tail: Promise<unknown> = Promise.resolve();
function locked<T>(fn: () => Promise<T> | T): Promise<T> {
  const p = tail.then(fn, fn);
  tail = p.catch(() => undefined);
  return p;
}

const statMs = (f: string) => { try { return fs.statSync(f).mtimeMs; } catch { return null; } };

/** 記憶を読み込む(ファイルが外から書き換えられていたら読み直す。そのときは意味のベクトルも読み直す) */
function load(): Brain {
  const m = statMs(STATE);
  if (!brain || m !== mtime) {
    const model = config().embed_model || 'qwen3-embedding:8b';
    if (!embedder || brain) embedder = new Embedder('meaning', model, QUERY_INSTRUCT);
    if (!store) store = new Store(RECORDS);
    else if (brain) store = new Store(RECORDS);   // 取り込み直しで記録のベクトルも増えている
    brain = new Brain(STATE, embedder, store);
    life = Life.restore(brain, LIFE);   // 全件の取り込み直しのあとは、その続きから覚える
    mtime = m;
    version += 1;
  }
  return brain;
}

function save(): void {
  brain!.save();
  mtime = statMs(STATE);
}

function snapshot(b: Brain) {
  const concepts = b.knownConcepts();
  const links = Object.entries(b.cortex).sort((x, y) => y[1].w - x[1].w).slice(0, LINKS_SHOWN);
  return {
    now: b.now, version,
    hippocampus: [...b.hippocampus].sort((x, y) => y.t - x.t).slice(0, HIPPO_SHOWN)
      .map((ep) => ({ concepts: ep.concepts, strength: b.hippoStrength(ep), nights: ep.nights })),
    hippocampus_total: b.hippocampus.length,
    cortex: links.map(([k, l]) => ({ a: k.split('\t')[0], b: k.split('\t')[1], w: l.w, access: b.access(l), eff: b.effective(l) })),
    cortex_total: Object.keys(b.cortex).length,
    concepts_total: concepts.size,
    places_total: [...concepts].filter(isPlace).length,
  };
}

/** Claude Code のフック(メッセージ送信時)から呼ばれる: 思い出したことを返し、何を出したかを記録する
 *  記録は、しばらく使ったあとで「差し込んだ記憶を Claude が実際に使ったか」を分析するためのもの。 */
async function hookRecall(b: Brain, body: any) {
  if (importing) return { text: '', ms: 0 };   // 最初の記憶を作っている途中
  const t0 = Date.now();
  const prompt = String(body.prompt ?? '').trim();
  const sid = body.session_id ?? '';
  const session = sid ? `${sid}.jsonl` : null;   // 記憶の中では、会話をセッション記録のファイル名で区別している
  if (session !== liveSession) {
    b.context = {};   // 新しい会話(または別の会話に切り替わった): 前の会話で考えていたことは持ち越さない
    liveSession = session;
  }
  const newConversation = !Object.keys(b.context).length;
  const r = await b.recollect(prompt, { session, cwd: body.cwd || null });
  dirty = true;
  const ms = Date.now() - t0;
  appendJsonl('recall_log.jsonl', {
    time: stamp(), session_id: sid, cwd: body.cwd, transcript_path: body.transcript_path, prompt, new_conversation: newConversation,
    episodes: r.episodes.map((e) => ({ ref: e.ref, conf: round(e.conf), prompt: e.prompt, when: e.when })),
    ranked: r.ranked.map((e) => ({ ref: e.ref, conf: round(e.conf) })),
    todos: r.todos.map((it) => ({ id: it.id, text: it.text, due: it.due })),
    recent: r.recent.map((e) => e.ref), places: r.places.map((p) => p.name), concepts: r.concepts.map((c) => c.name),
    cues: r.cues.slice(0, 20), text: r.text, ms,
  });
  return { text: r.text, ms };
}

const round = (x: number | null) => (x === null ? null : Math.round(x * 1000) / 1000);

/** セッション記録の新しいやり取りを覚え、すぐ保存する(呼ぶ側で locked の中から呼ぶ) */
async function learn(files: string[] = [], codex: string[] = [], settle = false): Promise<number> {
  const b = load();
  if (!life) return 0;
  const t0 = Date.now();
  const keep = [b.context, b.seq] as const;   // 覚えるときの体験で、今の会話のワーキングメモリを崩さない
  const n = await learnNew(life, files, codex, settle);
  [b.context, b.seq] = [keep[0], keep[1]];
  if (n) {
    save();
    life.save(LIFE);
    dirty = false;
  }
  appendJsonl('learn_log.jsonl', { time: stamp(), files: files.length + codex.length, learned: n, ms: Date.now() - t0 });
  return n;
}

/** 返答が終わった会話を、順番に覚える */
async function learner(): Promise<void> {
  for (;;) {
    const file = learnQueue.shift();
    if (!file) {
      await new Promise((r) => setTimeout(r, 500));
      continue;
    }
    if (!fs.existsSync(file) || importing) continue;
    try {
      await locked(() => learn([file]));
    } catch (e) {
      logLine('server_errors.log', `覚えられない: ${(e as Error).stack}`);   // 覚えるのに失敗しても、思い出す方は止めない
    }
  }
}

/** 前に読んだあとで書き足されたセッション記録を拾い直す(Codex の会話や、フックが動かなかった会話) */
async function catchUp(): Promise<void> {
  for (;;) {
    while (importing) await new Promise((r) => setTimeout(r, 10_000));
    try {
      await locked(async () => {
        load();
        if (!life) return;
        const changed = (ps: string[]) => ps.filter((p) => (statMs(p) ?? 0) > (life!.scanned[p] ?? 0));
        const files = changed(claudeFiles());
        const codex = changed(codexFiles());
        if (!files.length && !codex.length) return;
        await learn(files, codex, true);
        for (const p of [...files, ...codex]) {   // 途中かもしれないファイルは、次の回にもう一度読む
          const m = statMs(p) ?? 0;
          if (Date.now() - m >= SETTLE_SECONDS * 1000) life!.scanned[p] = m;
        }
        life!.save(LIFE);
      });
    } catch (e) {
      logLine('server_errors.log', `拾い直せない: ${(e as Error).stack}`);
    }
    await new Promise((r) => setTimeout(r, CATCH_UP_MS));
  }
}

async function autosave(): Promise<void> {
  for (;;) {
    await new Promise((r) => setTimeout(r, AUTOSAVE_MS));
    await locked(() => {
      // 取り込み直しなどで記憶のファイルが外から書き換えられていたら、保存せずに読み直しに任せる
      if (dirty && brain && statMs(STATE) === mtime) save();
      dirty = false;
    });
  }
}

/** 最初のメッセージで待たせないよう、記憶と検索の準備を起動時に済ませておく。記憶がまだなければ最初の記憶を作る */
async function warmUp(): Promise<void> {
  if (importing) {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    const log = fs.openSync(path.join(LOG_DIR, 'import.log'), 'a');
    await new Promise<void>((resolve) => {
      const child = spawn(process.execPath, ['--no-warnings', path.join(CODE, 'src', 'cli.ts'), 'import'],
        { cwd: CODE, stdio: ['ignore', log, log], windowsHide: true });
      child.on('exit', () => resolve());
      child.on('error', () => resolve());
    });
    importing = false;
  }
  await locked(async () => {
    const b = load();
    await b.recollect('起動', { reinforce: false });
    b.context = {};
  });
}

// ---- HTTP ----

async function handle(p: string, body: any): Promise<[number, any]> {
  return locked(async () => {
    const b = load();
    let result: any = {};
    if (p === '/api/state') {
      // 中身だけ返す
    } else if (p === '/api/learn') {   // 画面から手で覚えさせる
      const concepts = (body.concepts ?? []).map((c: string) => c.trim()).filter(Boolean);
      if (concepts.length < 2) return [400, { error: '関連する概念を2つ以上入力してください' }];
      await b.learn(concepts);
      result = { log: [`海馬に記録: ${concepts.join(' / ')}`] };
    } else if (p === '/api/recall') {   // 画面から試しに思い出す
      const text = String(body.text ?? '').trim();
      if (!text) return [400, { error: '指示を入力してください' }];
      const r: any = await b.recollect(text, { cwd: body.cwd || null });
      delete r.ranked;
      result = { recall: { ...r, query: text, cue_strength: b.lastCues, embedding: !!b.embedder?.available } };
      if (graph.version !== null) result.recall.overlay = graph.overlay(b.lastActivation, r.cues);
    } else if (p === '/api/hook') {
      return [200, await hookRecall(b, body)];
    } else if (p === '/api/learn_session') {   // 返答が終わった: その会話の新しいやり取りを、あとで(別に)覚える
      if (body.transcript_path) learnQueue.push(body.transcript_path);
      return [200, { queued: learnQueue.length }];
    } else if (p === '/api/search') {   // Claude が自分で探す(道具の recall)。今の会話のワーキングメモリは変えない
      if (importing) return [200, { importing: true, text: '' }];
      const query = String(body.query ?? '').trim();
      const r = await b.recollect(query, { useContext: false, reinforce: false, cwd: body.cwd || null });
      // 探すときは、会話の始まりに出す「直前の作業」は要らない。やることは、尋ねられたときだけ出す
      r.recent = [];
      if (!asksTodo(query)) r.todos = [];
      return [200, { text: recollectionText(r) }];
    } else if (p === '/api/sleep') {
      const log: string[] = [];
      for (let i = 0; i < Math.max(1, Math.min(60, Number(body.nights ?? 1))); i++) {
        const report = b.sleep();
        b.wait(AWAKE_HOURS);
        const count = (prefix: string[]) => report.filter((r) => prefix.some((x) => r.startsWith(x))).length;
        log.push(`${i + 1}晩目: 再生 ${count(['再生'])}件 / 皮質へ移行 ${count(['皮質へ'])}件 / 忘却 ${count(['消失', '海馬から忘却'])}件`);
      }
      result = { log };
    } else if (p === '/api/wait') {
      const hours = Math.max(0, Math.min(24 * 365, Number(body.hours ?? 1)));
      b.wait(hours);
      result = { log: [`${hours}時間経過`] };
    } else if (p === '/api/reset') {
      fs.rmSync(STATE, { force: true });
      brain = null;
      load();
      result = { log: ['記憶をすべて消去しました'] };
    } else {
      return [404, { error: 'not found' }];
    }
    if (p !== '/api/state') {
      save();
      if (p !== '/api/recall') version += 1;
    }
    result.state = snapshot(load());
    return [200, result];
  });
}

function send(res: http.ServerResponse, code: number, data: any, type = 'application/json; charset=utf-8', cache = false): void {
  const payload = Buffer.isBuffer(data) ? data : Buffer.from(typeof data === 'string' ? data : JSON.stringify(data));
  res.writeHead(code, { 'Content-Type': type, 'Content-Length': payload.length, 'Cache-Control': cache ? 'max-age=86400' : 'no-store' });
  res.end(payload);
}

async function currentGraph(): Promise<GraphMap> {
  const [b, v] = await locked(() => [load(), version] as const);
  await graph.update(b, v);
  return graph;
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url ?? '/', 'http://127.0.0.1');
  const q = Object.fromEntries(url.searchParams);
  try {
    if (req.method === 'GET') {
      if (url.pathname === '/' || url.pathname === '/index.html') return send(res, 200, fs.readFileSync(path.join(CODE, 'index.html')), 'text/html; charset=utf-8');
      if (url.pathname === '/api/status') return send(res, 200, { importing, loaded: brain !== null });   // 記憶を読み込まずにすぐ答える
      const tile = url.pathname.match(/^\/tiles\/(\d+)\/(-?\d+)\/(-?\d+)\.json$/);
      if (tile) return send(res, 200, (await currentGraph()).tile(Number(tile[1]), Number(tile[2]), Number(tile[3])), 'application/json; charset=utf-8', true);
      if (url.pathname === '/api/graph') return send(res, 200, (await currentGraph()).meta());
      if (url.pathname === '/api/node_at') return send(res, 200, { node: (await currentGraph()).nodeAt(Number(q.x), Number(q.y), Number(q.z)) });
      if (url.pathname === '/api/record') {
        const rec = await locked(() => {
          const b = load();
          const key = b.store?.find(String(q.id ?? '').replace(/[\[\]\s]/g, '')) ?? null;
          return key ? { key, rec: b.store!.record(key)! } : null;
        });
        return rec ? send(res, 200, { key: rec.key, text: render(rec.rec, 'full' in q) }) : send(res, 404, { error: '記録が見つからない' });
      }
      const [code, data] = await handle(url.pathname, {});
      return send(res, code, data);
    }
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString('utf8')) : {};
    const [code, data] = await handle(url.pathname, body);
    send(res, code, data);
  } catch (e) {
    logLine('server_errors.log', `${url.pathname}: ${(e as Error).stack}`);   // フックからの問い合わせの失敗に気づけるように
    send(res, 500, { error: String((e as Error).message) });
  }
});

server.on('error', (e: any) => {
  // すでに別の munin が待ち受けているなら、そちらに任せて終わる
  logLine('server.log', `待ち受けられない: ${e.code ?? e.message}`);
  process.exit(0);
});
server.listen(PORT, '127.0.0.1', () => {
  logLine('server.log', `起動した http://127.0.0.1:${PORT} pid=${process.pid}`);
  void initJapanese().then(warmUp).catch((e) => logLine('server_errors.log', `準備に失敗: ${e.stack}`));
  void autosave();
  void learner();
  void catchUp();
});
