// Claude Code で使ったときの記録(logs/recall_log.jsonl)から、差し込んだ記憶が実際に使われたかを調べる
//
//   node src/cli.ts usage                     使い始めた日時(config.json の trial_start)以降
//   node src/cli.ts usage --since 2026-10-15
//
// フックは、メッセージごとに「何を思い出して差し込んだか」を記録している。
// それを、そのあと Claude が実際にしたこと(セッション記録の返答とツールの実行)と突き合わせる。
//
// 使われたとみなす証拠:
//   開いた   : Claude が recall_detail でその記憶の鍵を開いた(はっきり使った)
//   語が出た : その記憶に含まれる語・ファイル・場所(依頼文にない、2文字以上のもの)が、
//              そのあとの Claude の返答や操作に出てきた(使ったかもしれない。偶然の一致も含む)
// 確信度ごとに使われた割合を出し、確信度が当てになっているか(高いものほど使われているか)を見る。

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { Brain, isPlace } from './brain.ts';
import { config, LOG_DIR, STATE } from './config.ts';

const BINS: [number, number][] = [[0, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1.01]];

/** セッション記録の中で、その依頼のあとに Claude がしたこと(返答と操作の文、次の依頼まで) */
async function afterPrompt(transcript: string | undefined, prompt: string): Promise<string | null> {
  if (!transcript || !fs.existsSync(transcript)) return null;
  const out: string[] = [];
  let found = false;
  const rl = readline.createInterface({ input: fs.createReadStream(transcript, 'utf8'), crlfDelay: Infinity });
  for await (const line of rl) {
    let d: any;
    try { d = JSON.parse(line); } catch { continue; }
    if (d.isSidechain) continue;
    const msg = d.message ?? {};
    if (d.type === 'user' && typeof msg.content === 'string') {
      if (found) break;
      found = msg.content.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim() === prompt;
    } else if (found && d.type === 'assistant') {
      for (const b of msg.content ?? []) {
        if (b.type === 'text') out.push(b.text);
        else if (b.type === 'tool_use') out.push(JSON.stringify(b.input ?? {}));
      }
    }
  }
  rl.close();
  return found ? out.join('\n') : null;
}

export async function usage(argv: string[]): Promise<void> {
  const si = argv.indexOf('--since');
  const since = si >= 0 ? argv[si + 1] : config().trial_start ?? '';
  const file = path.join(LOG_DIR, 'recall_log.jsonl');
  if (!fs.existsSync(file)) {
    console.log('まだ記録がありません');
    return;
  }
  const entries = fs.readFileSync(file, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l)).filter((e) => e.time >= since);
  const brain = new Brain(STATE, null, null);
  const conceptsOf = new Map(brain.traces.filter((tr) => tr.ref).map((tr) => [tr.ref!, tr.concepts]));
  const byBin = BINS.map(() => [0, 0, 0]);   // [差し込んだ数, 開いた数, 語が出た数]
  let checked = 0, openedAny = 0, hintedAny = 0;
  for (const e of entries) {
    const after = await afterPrompt(e.transcript_path, e.prompt);
    if (after === null) continue;
    checked++;
    const prompt = e.prompt.toLowerCase();
    const low = after.toLowerCase();
    let opened = false, hinted = false;
    for (const ep of e.episodes) {
      const b = BINS.findIndex(([lo, hi]) => lo <= ep.conf && ep.conf < hi);
      byBin[b][0]++;
      if (after.includes(ep.ref.slice(0, 8))) {
        byBin[b][1]++;
        opened = true;
      }
      const words = (conceptsOf.get(ep.ref) ?? []).filter((c) => c.length >= 2 && !prompt.includes(c.toLowerCase()))
        .map((c) => (isPlace(c) ? c.replace(/[\\/]+$/, '').split(/[\\/]/).pop()! : c));
      if (words.some((w) => w && low.includes(w.toLowerCase()))) {
        byBin[b][2]++;
        hinted = true;
      }
    }
    if (opened) openedAny++;
    if (hinted) hintedAny++;
  }
  const n = Math.max(entries.length, 1);
  const ms = entries.map((e) => e.ms);
  if (since) console.log(`${since} 以降の記録`);
  console.log(`差し込んだ回数 ${entries.length}回 / セッション記録と突き合わせられたもの ${checked}回`);
  console.log(`差し込みの長さ 平均${(entries.reduce((s, e) => s + e.text.length, 0) / n).toFixed(0)}文字 / 何も差し込まなかった ${entries.filter((e) => !e.text).length}回`);
  console.log(`所要時間(サーバー側) 平均${(ms.reduce((s, x) => s + x, 0) / n).toFixed(0)}ms / 最大${Math.max(0, ...ms)}ms`);
  console.log(`記憶の鍵を開いた ${openedAny}回 / 差し込んだ記憶の語がそのあと出てきた ${hintedAny}回(突き合わせた${checked}回中)\n`);
  console.log('| 確信度 | 差し込んだ記憶 | 鍵を開いた | 語が出てきた |\n|---|---|---|---|');
  BINS.forEach(([lo, hi], i) => {
    const [shown, opened, hinted] = byBin[i];
    const rate = (k: number) => (shown ? `${Math.round((100 * k) / shown)}%` : '-');
    console.log(`| ${lo.toFixed(1)}〜${Math.min(hi, 1).toFixed(1)} | ${shown} | ${opened}(${rate(opened)}) | ${hinted}(${rate(hinted)}) |`);
  });
}
