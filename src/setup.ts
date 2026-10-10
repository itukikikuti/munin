// munin を使えるようにする(最初の1回。/munin:setup から呼ばれる)
//
//   node src/cli.ts setup           GPU の大きさを見て、使う埋め込みモデルを決める
//   node src/cli.ts setup --full    GPU が小さくても大きいモデル(qwen3-embedding:8b)を使う
//   node src/cli.ts setup --light   軽いモデル(qwen3-embedding:0.6b)だけを使う
//
//   1. 埋め込みを計算する llama.cpp を取ってくる(約30MB)
//   2. 日本語を語に分ける kuromoji の辞書を取ってくる(約17MB)
//   3. 埋め込みモデルを取ってくる(8b は約4.7GB、0.6b は約0.6GB。Ollama に同じものがあればそれを使う)
//   4. 常駐サーバーを起動する。記憶がまだなければ、これまでのすべての会話から最初の記憶を作り始める

import fs from 'node:fs';
import path from 'node:path';
import { config, DATA, LOG_DIR, saveConfig, STATE } from './config.ts';
import { serverUp, startServer } from './hook.ts';
import { chooseDevice, download, ensureModel, ensureRuntime, MODELS, modelPath } from './llama.ts';
import { KUROMOJI_ASSETS, kuromojiPath, kuromojiReady } from './text.ts';
import { spawnSync } from 'node:child_process';

const FULL = 'qwen3-embedding:8b';
const LIGHT = 'qwen3-embedding:0.6b';
const GPU_MIB_FOR_FULL = 6000;   // 8b を GPU に載せるには 6GB 程度は要る

/** 使う GPU のメモリ(MiB)。GPU がなければ 0 */
function gpuMiB(exe: string): number {
  const device = chooseDevice(exe);
  if (!device) return 0;
  const r = spawnSync(exe, ['--list-devices'], { encoding: 'utf8', timeout: 30_000 });
  const m = `${r.stdout}\n${r.stderr}`.match(new RegExp(`^\\s*${device}:.*?\\((\\d+) MiB`, 'm'));
  return m ? Number(m[1]) : 0;
}

export async function setup(argv: string[]): Promise<number> {
  const [major, minor] = process.versions.node.split('.').map(Number);
  if (major < 22 || (major === 22 && minor < 18)) {
    console.log(`Node.js ${process.versions.node} では動きません。22.18 以上を入れてください(https://nodejs.org)。`);
    return 1;
  }
  fs.mkdirSync(DATA, { recursive: true });
  const progress = (msg: string) => console.log(`  ${msg}`);

  console.log('\n== 1. llama.cpp(埋め込みを計算する)');
  let exe: string;
  try {
    exe = await ensureRuntime(progress);
    console.log(`  あります: ${exe}`);
  } catch (e) {
    console.log(`  用意できませんでした: ${(e as Error).message}`);
    return 1;
  }

  console.log('\n== 2. kuromoji(日本語を語に分ける辞書、約17MB)');
  try {
    if (!kuromojiReady()) for (const a of KUROMOJI_ASSETS) await download(a, kuromojiPath(a), progress);
    console.log('  あります');
  } catch (e) {
    console.log(`  用意できませんでした: ${(e as Error).message}`);
    return 1;
  }

  console.log('\n== 3. 埋め込みモデル');
  const c = config();
  if (argv.includes('--full') || argv.includes('--light') || !c.embed_model) {
    const mib = gpuMiB(exe);
    const light = argv.includes('--light') || (!argv.includes('--full') && mib < GPU_MIB_FOR_FULL);
    c.embed_model = light ? LIGHT : FULL;
    saveConfig(c);
    console.log(light ? `  GPU が小さいかないので(${mib}MiB)、軽いモデルだけを使います(--full で大きいモデルにできます)`
      : `  GPU(${mib}MiB)があるので、精度の高い大きいモデルを使います`);
  }
  console.log(`  概念と依頼の意味: ${c.embed_model} / 記録の部分: ${LIGHT}`);
  for (const id of new Set([c.embed_model!, LIGHT])) {
    try {
      const had = fs.existsSync(modelPath(id));
      const p = await ensureModel(id, progress);
      console.log(`  ${id}: ${had ? 'あります' : '取ってきました'}(${p.includes('.ollama') ? 'Ollama のものを使います' : p})`);
    } catch (e) {
      console.log(`  ${id}(${(MODELS[id].size / 1e9).toFixed(1)}GB)を用意できませんでした: ${(e as Error).message}`);
      return 1;
    }
  }

  console.log('\n== 4. 記憶');
  if (await serverUp()) console.log('  常駐サーバーはもう動いています');
  else {
    startServer();
    console.log('  常駐サーバーを起動しました');
  }
  if (fs.existsSync(STATE)) console.log(`  記憶があります(${DATA})。準備ができました。`);
  else console.log('  これまでのすべての会話から、最初の記憶を作り始めました(会話の量によって数分〜数十分)。\n'
    + `  進み具合は ${path.join(LOG_DIR, 'import.log')} で見られます。できるまでは、思い出したことは差し込まれません。`);
  return 0;
}
