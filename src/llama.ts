// 文の意味を数値(埋め込み)にする: llama.cpp の llama-server で qwen3-embedding を動かす
//
// Ollama と同じ実行エンジン(llama.cpp)と同じモデルファイル(Qwen 公式の GGUF)を使うので、Ollama と同じベクトルになる。
// 実行ファイルとモデルは最初の準備で取ってくる(版を固定し、SHA-256 で照合する)。
// Ollama にすでに同じモデルが入っていれば、そのファイルをそのまま使う(取ってこない)。
//
// GPU が複数あるとき(NVIDIA と CPU 内蔵の Intel など)、llama.cpp は既定でモデルを分けて載せ、遅い方に引きずられる。
// 内蔵 GPU を避けて、いちばん大きい GPU だけを使う。

import { spawn, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { DATA, logLine } from './config.ts';

export interface Asset {
  name: string;
  url: string;
  sha256: string;
  size: number;
}

const LLAMA = 'b11267';
const RELEASE = `https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA}`;

/** llama.cpp の実行ファイル(OS と CPU ごと)。Vulkan 版は NVIDIA・AMD・Intel のどの GPU でも動き、GPU がなければ CPU で動く */
const RUNTIMES: Record<string, Asset> = {
  'win32-x64': { name: `llama-${LLAMA}-bin-win-vulkan-x64.zip`, url: '', size: 33093219,
    sha256: 'b24fd5c53fa0c10b759604b95903f76877457d72c9999372f5ea6c440d7c6008' },
  'win32-arm64': { name: `llama-${LLAMA}-bin-win-cpu-arm64.zip`, url: '', size: 12053161,
    sha256: '5cde4a7e8a1fddf48d75d349b3b751c73524c827707aa9875c9485a566cbd449' },
  'linux-x64': { name: `llama-${LLAMA}-bin-ubuntu-vulkan-x64.tar.gz`, url: '', size: 31355150,
    sha256: '8b73cdad74e724aa6a59af0a6dfa0eaa0e9189d2424a4e4802e50f39f86a14ea' },
  'linux-arm64': { name: `llama-${LLAMA}-bin-ubuntu-vulkan-arm64.tar.gz`, url: '', size: 24664694,
    sha256: 'b80009c93f1636b260735411fbda933061ab93b3385a915a8f2c644c2329e007' },
  'darwin-arm64': { name: `llama-${LLAMA}-bin-macos-arm64.tar.gz`, url: '', size: 11767607,
    sha256: '4bfcdbf7b059f7de0062dcb74a5fd050661cc17319e917bb3058e90a8772870c' },
  'darwin-x64': { name: `llama-${LLAMA}-bin-macos-x64.tar.gz`, url: '', size: 11320213,
    sha256: 'd07394c4a3e7baa43216915528f9277bda79a126aaaca91af1a697ca5e8439f6' },
};
for (const a of Object.values(RUNTIMES)) a.url = `${RELEASE}/${a.name}`;

export interface Model extends Asset {
  /** 埋め込みの次元 */
  dim: number;
  /** 1つの文で読める長さ(トークン)。文は MAX_TEXT 文字までに切るので、ほとんどこれに収まる */
  ctx: number;
}

/** qwen3-embedding(Qwen 公式の GGUF。Ollama の qwen3-embedding:8b / :0.6b と同じファイル) */
export const MODELS: Record<string, Model> = {
  'qwen3-embedding:8b': {
    name: 'Qwen3-Embedding-8B-Q4_K_M.gguf', size: 4676804928, dim: 4096, ctx: 2048,
    url: 'https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF/resolve/69d0e58a13e463cd99a9b83e3f5fee7c10265fab/Qwen3-Embedding-8B-Q4_K_M.gguf',
    sha256: '3fcd3febec8b3fd64435204db75bf0dd73b91e8d0661e0331acfe7e7c3120b85' },
  'qwen3-embedding:0.6b': {
    name: 'Qwen3-Embedding-0.6B-Q8_0.gguf', size: 639150592, dim: 1024, ctx: 2048,
    url: 'https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/370f27d7550e0def9b39c1f16d3fbaa13aa67728/Qwen3-Embedding-0.6B-Q8_0.gguf',
    sha256: '06507c7b42688469c4e7298b0a1e16deff06caf291cf0a5b278c308249c3e439' },
};

/** モデルごとの llama-server の待ち受け番号 */
const PORTS: Record<string, number> = { 'qwen3-embedding:8b': 8771, 'qwen3-embedding:0.6b': 8772 };

const RUNTIME_DIR = path.join(DATA, 'runtime', `llama-${LLAMA}`);
const MODEL_DIR = path.join(DATA, 'models');

export function platformRuntime(): Asset | null {
  return RUNTIMES[`${process.platform}-${process.arch}`] ?? null;
}

/** llama-server の実行ファイル(まだ取ってきていなければ null) */
export function serverExe(): string | null {
  const name = process.platform === 'win32' ? 'llama-server.exe' : 'llama-server';
  const find = (dir: string, depth: number): string | null => {
    let entries: fs.Dirent[];
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return null;
    }
    for (const e of entries) if (e.isFile() && e.name === name) return path.join(dir, e.name);
    if (depth > 0) for (const e of entries) if (e.isDirectory()) {
      const hit = find(path.join(dir, e.name), depth - 1);
      if (hit) return hit;
    }
    return null;
  };
  return find(RUNTIME_DIR, 3);
}

/** モデルファイルの場所。Ollama に同じファイル(同じ SHA-256)があればそれを使う。なければ munin の置き場所 */
export function modelPath(id: string): string {
  const m = MODELS[id];
  const ollama = process.env.OLLAMA_MODELS || path.join(os.homedir(), '.ollama', 'models');
  const blob = path.join(ollama, 'blobs', `sha256-${m.sha256}`);
  if (sizeIs(blob, m.size)) return blob;
  return path.join(MODEL_DIR, m.name);
}

export function modelReady(id: string): boolean {
  return sizeIs(modelPath(id), MODELS[id].size);
}

function sizeIs(file: string, size: number): boolean {
  try {
    return fs.statSync(file).size === size;
  } catch {
    return false;
  }
}

/** 取ってきて SHA-256 で照合する(途中で失敗しても、次回やり直せるように .part に書いてから置き換える) */
export async function download(a: Asset, dest: string, onProgress?: (msg: string) => void): Promise<void> {
  if (sizeIs(dest, a.size)) return;
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  const res = await fetch(a.url, { signal: AbortSignal.timeout(6 * 3600_000) });
  if (!res.ok || !res.body) throw new Error(`${a.url} を取得できない(HTTP ${res.status})`);
  const tmp = `${dest}.part`;
  const out = fs.createWriteStream(tmp);
  const hash = crypto.createHash('sha256');
  let got = 0;
  let shown = 0;
  for await (const chunk of Readable.fromWeb(res.body as any)) {
    hash.update(chunk);
    got += chunk.length;
    if (!out.write(chunk)) await new Promise((r) => out.once('drain', r));
    if (onProgress && got - shown > a.size / 20) {
      shown = got;
      onProgress(`${a.name} ${Math.round((100 * got) / a.size)}%`);
    }
  }
  await new Promise<void>((resolve, reject) => out.end((e?: Error | null) => (e ? reject(e) : resolve())));
  const sha = hash.digest('hex');
  if (sha !== a.sha256 || got !== a.size) {
    fs.rmSync(tmp, { force: true });
    throw new Error(`${a.name} の中身が想定と違う(${got}バイト、${sha.slice(0, 12)}…)`);
  }
  fs.renameSync(tmp, dest);
  logLine('setup.log', `${a.name} を取得した`);
}

/** llama.cpp を取ってきて展開する */
export async function ensureRuntime(onProgress?: (msg: string) => void): Promise<string> {
  const have = serverExe();
  if (have) return have;
  const a = platformRuntime();
  if (!a) throw new Error(`この OS・CPU(${process.platform}-${process.arch})向けの llama.cpp がない`);
  const archive = path.join(DATA, 'runtime', a.name);
  await download(a, archive, onProgress);
  fs.mkdirSync(RUNTIME_DIR, { recursive: true });
  // Windows に入っている tar(bsdtar)は zip も展開できる。macOS・Linux の tar は tar.gz を展開できる
  const tar = process.platform === 'win32' ? path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe') : 'tar';
  const r = spawnSync(tar, ['-xf', archive, '-C', RUNTIME_DIR], { encoding: 'utf8' });
  if (r.status !== 0) throw new Error(`llama.cpp を展開できない: ${r.stderr || r.error}`);
  fs.rmSync(archive, { force: true });
  const exe = serverExe();
  if (!exe) throw new Error('展開した llama.cpp に llama-server がない');
  if (process.platform !== 'win32') fs.chmodSync(exe, 0o755);
  return exe;
}

export async function ensureModel(id: string, onProgress?: (msg: string) => void): Promise<string> {
  const p = modelPath(id);
  if (!sizeIs(p, MODELS[id].size)) await download(MODELS[id], p, onProgress);
  return p;
}

/** 使う GPU(内蔵 GPU を避けて、メモリのいちばん大きいもの)。GPU がなければ null(CPU で動く) */
export function chooseDevice(exe: string): string | null {
  const r = spawnSync(exe, ['--list-devices'], { encoding: 'utf8', timeout: 30_000 });
  const devices = [...`${r.stdout}\n${r.stderr}`.matchAll(/^\s*(\w+\d+):\s*(.+?)\s*\((\d+) MiB/gm)]
    .map((m) => ({ id: m[1], name: m[2], mib: Number(m[3]) }));
  const integrated = /Intel\(R\) (UHD|HD|Iris)|Microsoft Basic|llvmpipe|SwiftShader/i;
  const real = devices.filter((d) => !integrated.test(d.name));
  const pool = real.length ? real : devices;
  if (!pool.length) return null;
  return pool.sort((a, b) => b.mib - a.mib)[0].id;
}

const servers = new Map<string, Promise<string>>();

/** モデルの llama-server を使える状態にする(動いていなければ起動して待つ)。戻り値は呼び先の URL */
export function serve(id: string): Promise<string> {
  let p = servers.get(id);
  if (!p) {
    p = start(id).catch((e) => {
      servers.delete(id);
      throw e;
    });
    servers.set(id, p);
  }
  return p;
}

async function healthy(url: string, id: string): Promise<boolean> {
  try {
    const r = await fetch(`${url}/v1/models`, { signal: AbortSignal.timeout(2000) });
    if (!r.ok) return false;
    const j: any = await r.json();
    return (j.data ?? []).some((m: any) => m.id === id);
  } catch {
    return false;
  }
}

async function start(id: string): Promise<string> {
  const url = `http://127.0.0.1:${PORTS[id]}`;
  if (await healthy(url, id)) return url;   // 前に起動したものが動いていれば使い回す
  const exe = serverExe();
  if (!exe || !modelReady(id)) throw new Error('埋め込みの準備ができていない(/munin:setup を実行する)');
  const m = MODELS[id];
  // 4つの文を並べて計算する(1つずつより数倍速い)。1つの文は m.ctx トークンまで
  const args = ['-m', modelPath(id), '--alias', id, '--embedding', '--pooling', 'last',
    '-c', String(m.ctx * PARALLEL), '-b', String(m.ctx), '-ub', String(m.ctx), '--parallel', String(PARALLEL),
    '--host', '127.0.0.1', '--port', String(PORTS[id])];
  const device = chooseDevice(exe);
  if (device) {
    args.push('--device', device);
    // GPU に全部載せると速い。載りきらなければ、llama.cpp に GPU と CPU の配分を任せてやり直す
    if (await launch(id, url, exe, [...args, '-ngl', '999'], device)) return url;
  }
  if (await launch(id, url, exe, args, device)) return url;
  throw new Error(`llama-server(${id})が起動しない。logs を確かめる`);
}

const PARALLEL = 4;

async function launch(id: string, url: string, exe: string, args: string[], device: string | null): Promise<boolean> {
  fs.mkdirSync(path.join(DATA, 'logs'), { recursive: true });
  const log = fs.openSync(path.join(DATA, 'logs', `llama-${id.replace(/[^\w.-]/g, '_')}.log`), 'a');
  // munin の常駐サーバーが止まっても、次に起動したときに使い回せるよう切り離して動かす
  const child = spawn(exe, args, { detached: true, stdio: ['ignore', log, log], windowsHide: true });
  child.unref();
  logLine('server.log', `llama-server(${id})を起動した pid=${child.pid} device=${device ?? 'CPU'} ${args.includes('-ngl') ? '全部GPU' : '自動配分'}`);
  for (let i = 0; i < 600; i++) {
    if (await healthy(url, id)) return true;
    if (child.exitCode !== null) return false;
    await new Promise((r) => setTimeout(r, 500));
  }
  child.kill();
  return false;
}

/** 文を埋め込みにする(長さ1にそろえる) */
export async function embed(id: string, texts: string[]): Promise<Float32Array[]> {
  if (!texts.length) return [];
  const url = await serve(id);
  try {
    return await request(url, id, texts);
  } catch (e) {
    if (!(e instanceof TooLong)) throw e;
  }
  // どれかの文が長すぎた(記号や珍しい漢字が多いと、文字数のわりにトークンが多い)。1つずつ、通るまで短くしてやり直す
  const out: Float32Array[] = [];
  for (const t of texts) {
    let text = t;
    for (;;) {
      try {
        out.push((await request(url, id, [text]))[0]);
        break;
      } catch (e) {
        if (!(e instanceof TooLong) || text.length < 50) throw e;
        text = text.slice(0, Math.floor(text.length * 0.7));
      }
    }
  }
  return out;
}

class TooLong extends Error {}

async function request(url: string, id: string, texts: string[]): Promise<Float32Array[]> {
  const r = await fetch(`${url}/v1/embeddings`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ input: texts, model: id }), signal: AbortSignal.timeout(600_000),
  });
  if (!r.ok) {
    const body = await r.text();
    if (r.status === 400 || r.status === 500) if (/too large|exceed|context/i.test(body)) throw new TooLong(body);
    throw new Error(`埋め込みに失敗(HTTP ${r.status}): ${body.slice(0, 200)}`);
  }
  const j: any = await r.json();
  return j.data.sort((a: any, b: any) => a.index - b.index).map((d: any) => {
    const v = Float32Array.from(d.embedding as number[]);
    let n = 0;
    for (const x of v) n += x * x;
    n = Math.sqrt(n) || 1;
    for (let i = 0; i < v.length; i++) v[i] /= n;
    return v;
  });
}
