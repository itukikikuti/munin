// 実際のセッション記録で、関連することを思い出せるかを測る(LLM は使わない)
//
//   node src/cli.ts eval                      調整用の期間で評価する(普段はこれ)
//   node src/cli.ts eval --mode test          最終確認用の期間で評価する(調整が終わったときだけ)
//   node src/cli.ts eval --ablate sleep       仕組みを1つ外して評価する
//   node src/cli.ts eval --set EP_W_PLACE=1   パラメータを変えて評価する
//
// 期間の分け方: TEST_FROM 以降は最終確認用に取り置き、調整のあいだは使わない。その手前の VALID_N 件を調整用にする。
// 過去のやり取り(学習分)で記憶を作ったあと、評価分のやり取りを時刻順に1件ずつ、
// 「依頼から関連することを思い出させる → 実際の出来事を覚えさせる」の順で体験させる。
//
// 正解: そのやり取りで実際に使われた語・ファイル・場所(依頼と Claude の回答から取り出した概念)のうち、
//       それより前から記憶にあったもの。ただし、依頼文にそのまま書いてある語、今の会話にすでに出てきた
//       情報、今の作業フォルダ(とそれを含む場所)は除く(Claude がすでに知っているので、差し込む価値がない)。
// 比べる方式(どれも過去のやり取りを5件思い出す): 記憶モデル / RAG(要旨の意味の近さだけ)/ 直近 / キーワード

import fs from 'node:fs';
import path from 'node:path';
import { ABLATE, ABLATIONS, atPlace, Brain, isPlace, P, recollectionText, type Trace } from './brain.ts';
import { DATA, RECORDS } from './config.ts';
import { Embedder } from './embed.ts';
import { Life, loadTurns, QUERY_INSTRUCT, recordKey, storeRecords } from './sessions.ts';
import { Store } from './store.ts';
import { initJapanese } from './text.ts';
import { config } from './config.ts';

const OUT = path.join(DATA, 'eval');
const METHODS = ['記憶モデル', 'RAG', '直近', 'キーワード'] as const;
// 最終確認用の期間の始まり(UTC)。日時で固定しておくと、新しいやり取りが増えても調整用の期間はずれない
const TEST_FROM = '2026-10-05T03:00';
const VALID_N = 300;
const CONF_BINS: [number, number][] = [[0, 0.3], [0.3, 0.5], [0.5, 0.7], [0.7, 0.9], [0.9, 1.01]];
const SECTIONS = ['todos', 'recent', 'episodes', 'places', 'concepts'] as const;
const CANDIDATES = 50;
const DEV_PLACES = ['brain-memory', 'munin'];   // このプロジェクト自体を作っているやり取り

const relevant = (concepts: string[], used: Set<string>) => concepts.filter((c) => used.has(c)).length >= 2;

/** 思い出したやり取りの集まりについて、[再現率, 適合率] */
function score(episodes: string[][], truth: Set<string>, used: Set<string>): [number, number] {
  const info = new Set(episodes.flat());
  const rec = [...truth].filter((c) => info.has(c)).length / truth.size;
  const prec = episodes.length ? episodes.filter((e) => relevant(e, used)).length / episodes.length : 0;
  return [rec, prec];
}

/** ACT-R の検索確率 P = 1 / (1 + exp(-(A - τ) / s)) の τ と s を、実際の当たり外れから推定する(ロジスティック回帰) */
function fitRetrieval(acts: number[], labels: boolean[]): [number, number] {
  const n = acts.length || 1;
  const mu = acts.reduce((s, x) => s + x, 0) / n;
  const sd = Math.sqrt(acts.reduce((s, x) => s + (x - mu) ** 2, 0) / n) || 1;
  const z = acts.map((x) => (x - mu) / sd);
  let w = 1, b = 0;
  for (let it = 0; it < 3000; it++) {
    let gw = 0, gb = 0;
    z.forEach((zi, i) => {
      const p = 1 / (1 + Math.exp(-(w * zi + b)));
      gw += (p - (labels[i] ? 1 : 0)) * zi;
      gb += p - (labels[i] ? 1 : 0);
    });
    w -= (0.5 * gw) / n;
    b -= (0.5 * gb) / n;
  }
  const s = w ? sd / w : P.EP_S;
  return [mu - b * s, s];
}

export async function evaluate(argv: string[]): Promise<void> {
  const arg = (name: string) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : undefined; };
  const mode = arg('--mode') ?? 'valid';
  const tag = arg('--tag') ?? '';
  const sets = argv.flatMap((a, i) => (argv[i - 1] === '--set' ? [a] : []));
  for (const a of (arg('--ablate') ?? '').split(',').filter(Boolean)) {
    if (!(a in ABLATIONS)) throw new Error(`知らない仕組み: ${a}`);
    ABLATE.add(a);
  }
  for (const kv of sets) {
    const [name, value] = kv.split('=');
    if (!(name in P)) throw new Error(`P に ${name} という値はない`);
    (P as any)[name] = JSON.parse(value);
  }
  fs.mkdirSync(OUT, { recursive: true });
  await initJapanese();
  const t0 = Date.now();
  const store = new Store(RECORDS);   // やり取りの記録は事実なので、本番と共有する
  let turns = await loadTurns(store);
  const nValid = turns.filter((t) => t.t < TEST_FROM).length;
  let train, test;
  if (mode === 'valid') {
    turns = turns.slice(0, nValid);   // 最終確認用の期間は見ない
    train = turns.slice(0, -VALID_N);
    test = turns.slice(-VALID_N);
  } else {
    train = turns.slice(0, nValid);
    test = turns.slice(nValid);
  }
  console.log(`学習 ${train.length}件 (${train[0].t.slice(0, 10)}〜${train.at(-1)!.t.slice(0, 10)}) / 評価 ${test.length}件 (${test[0].t.slice(0, 10)}〜${test.at(-1)!.t.slice(0, 10)})`);
  const embedder = new Embedder('meaning', config().embed_model || 'qwen3-embedding:8b', QUERY_INSTRUCT);
  await embedder.ensureGists(turns.map((t) => t.user), true);
  await storeRecords(store, turns);
  const state = path.join(OUT, 'state_memory.json');
  fs.rmSync(state, { force: true });
  const brain = new Brain(state, embedder, store);
  brain.checkPlaces = false;   // 過去の時点で思い出すので、今の時点で場所があるかは確かめない
  const life = new Life(brain, turns[0]);
  for (const [i, t] of train.entries()) {
    if (i % 50 === 0) process.stderr.write(`\r  学習 ${i}/${train.length}`);
    await life.experience(t);
  }
  process.stderr.write('\n');
  await embedder.ensure(brain.knownConcepts(), true);
  // 「それより前から記憶にあった」かどうかは、記憶モデルの状態(忘れたかどうか)ではなく、過去のやり取りに出てきたかどうかで決める
  const seen = new Set<string>([...brain.traces.flatMap((tr) => tr.concepts), ...brain.knownConcepts()]);

  const results: any[] = [];
  let prevSession: string | null = null;
  for (const [i, t] of test.entries()) {
    process.stderr.write(`\r  評価 ${i}/${test.length}`);
    life.advance(t);
    const first = t.session !== prevSession;
    prevSession = t.session;
    const actual = life.episode(t);
    const used = new Set(actual);
    const prompt = t.user.toLowerCase();
    const session = path.basename(t.session);
    const inConversation = new Set(brain.traces.filter((tr) => tr.session === session).flatMap((tr) => tr.concepts));
    // 今の作業フォルダ(とそれを含む場所)は、Claude がすでに知っているので、思い出す価値がない
    const here = new Set(actual.filter((c) => t.cwd && isPlace(c) && atPlace(t.cwd, c)));
    const truth = new Set(actual.filter((c) => seen.has(c) && !prompt.includes(c.toLowerCase()) && !inConversation.has(c) && !here.has(c)));

    const r = await brain.recollect(t.user, { session, cwd: t.cwd });
    const sims = brain.lastTraceSim;
    const traces = brain.traces.filter((tr) => tr.session !== session);
    const ragAll = [...traces].sort((a, b) => (sims.get(b) ?? 0) - (sims.get(a) ?? 0));
    const recent = [...traces].sort((a, b) => b.t - a.t).slice(0, P.EPISODES_SHOWN);
    const kwCount = (tr: Trace) => tr.concepts.filter((c) => prompt.includes(c.toLowerCase())).length;
    const kw = [...traces].sort((a, b) => kwCount(b) - kwCount(a) || b.t - a.t).slice(0, P.EPISODES_SHOWN);
    const picks: Record<string, string[][]> = {
      記憶モデル: r.ranked.slice(0, P.EPISODES_SHOWN).map((e) => e.concepts), RAG: ragAll.slice(0, P.EPISODES_SHOWN).map((tr) => tr.concepts),
      直近: recent.map((tr) => tr.concepts), キーワード: kw.map((tr) => tr.concepts),
    };
    const shown = [...r.episodes.map((e) => e.concepts), r.concepts.map((c) => c.name), r.places.map((p) => p.name), ...r.recent.map((e) => e.concepts)];
    const has = truth.size > 0;
    const lowText = r.text.toLowerCase();
    results.push({
      key: recordKey(t), prompt: t.user.slice(0, 150), first, truth: [...truth].sort(), dev: DEV_PLACES.some((d) => [...t.places!, t.cwd ?? ''].some((p) => p.toLowerCase().includes(d))),
      scores: has ? Object.fromEntries(METHODS.map((m) => [m, score(picks[m], truth, used)])) : null,
      candidate_recall: has ? {
        記憶モデル: score(brain.lastCandidates.slice(0, CANDIDATES).map((tr) => tr.concepts), truth, used)[0],
        RAG: score(ragAll.slice(0, CANDIDATES).map((tr) => tr.concepts), truth, used)[0],
      } : null,
      shown_recall: has ? score(shown, truth, used)[0] : null,
      episodes: r.ranked.map((e) => ({ activation: e.activation, relevant: relevant(e.concepts, used) })),
      text_len: r.text.length,
      text_parts: Object.fromEntries(SECTIONS.map((k) => [k, recollectionText({ todos: [], recent: [], episodes: [], places: [], concepts: [], [k]: (r as any)[k] }).length])),
      text_recall: has ? [...truth].filter((c) => lowText.includes(c.toLowerCase())).length / truth.size : null,
      cands: has ? brain.lastCandidates.slice(0, CANDIDATES).map((tr) => ({ f: brain.lastFeatures.get(tr), rel: relevant(tr.concepts, used),
        hits: tr.concepts.filter((c) => truth.has(c)).sort() })) : null,
    });
    await life.remember(t, actual);
    for (const c of actual) seen.add(c);
  }
  process.stderr.write('\n');
  const name = mode + (ABLATE.size ? '_' + [...ABLATE].sort().join('_') : '') + (tag ? '_' + tag : '');
  fs.writeFileSync(path.join(OUT, `memory_results_${name}.jsonl`), results.map((x) => JSON.stringify(x)).join('\n') + '\n');
  const report = makeReport(results, train, test, (Date.now() - t0) / 1000, mode, sets);
  fs.writeFileSync(path.join(OUT, `memory_report_${name}.md`), report);
  console.log(report);
}

function makeReport(results: any[], train: any[], test: any[], seconds: number, mode: string, sets: string[]): string {
  const scored = results.filter((x) => x.scores);
  const first = scored.filter((x) => x.first);
  const later = scored.filter((x) => !x.first);
  const mean = (xs: any[], m: string, k: number) => xs.reduce((s, x) => s + x.scores[m][k], 0) / Math.max(xs.length, 1);
  const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
  const lines = ['# 関連することを思い出せるかの自動評価', '',
    `- 期間: ${mode === 'valid' ? '調整用' : '最終確認用'}${ABLATE.size ? ` / 外した仕組み: ${[...ABLATE].join(', ')}` : ''}`,
    `- 変えた設定: ${sets.join(', ') || 'なし'}`,
    `- 学習: ${train.length}件(${train[0].t.slice(0, 10)}〜${train.at(-1).t.slice(0, 10)})`,
    `- 評価: ${test.length}件(${test[0].t.slice(0, 10)}〜${test.at(-1).t.slice(0, 10)})。正解の情報があるもの ${scored.length}件`,
    `- 所要時間: ${seconds.toFixed(0)}秒`, '',
    `## 再現率と適合率(どの方式も過去のやり取りを${P.EPISODES_SHOWN}件思い出す)`, '',
    '| 方式 | 再現率(全体) | 適合率(全体) | 再現率(会話の最初) | 再現率(会話の途中) |', '|---|---|---|---|---|'];
  for (const m of METHODS) lines.push(`| ${m} | ${pct(mean(scored, m, 0))} | ${pct(mean(scored, m, 1))} | ${pct(mean(first, m, 0))}(${first.length}件) | ${pct(mean(later, m, 0))}(${later.length}件) |`);
  const clean = scored.filter((x) => !x.dev);
  lines.push('', `このプロジェクト自体を作っているやり取り(${scored.length - clean.length}件)を除いた ${clean.length}件では:`, '', '| 方式 | 再現率 | 適合率 |', '|---|---|---|');
  for (const m of METHODS) lines.push(`| ${m} | ${pct(mean(clean, m, 0))} | ${pct(mean(clean, m, 1))} |`);
  const cand = (m: string) => scored.reduce((s, x) => s + x.candidate_recall[m], 0) / Math.max(scored.length, 1);
  const avg = (xs: any[], f: (x: any) => number) => xs.reduce((s, x) => s + f(x), 0) / Math.max(xs.length, 1);
  const idle = results.filter((x) => !x.scores && !x.first);
  const parts = SECTIONS.map((k, i) => `${['やること', '直前の作業', '出来事', '場所', '語'][i]} ${avg(results, (x) => x.text_parts[k]).toFixed(0)}`).join('、');
  lines.push('', `候補の段階(上位${CANDIDATES}件)での再現率: 記憶モデル ${pct(cand('記憶モデル'))} / RAG ${pct(cand('RAG'))}。`,
    '', `記憶モデルが実際に差し込む内容の再現率は ${pct(avg(scored, (x) => x.shown_recall))}。差し込む文章の中に、実際に使われた情報がそのまま書かれている割合は ${pct(avg(scored, (x) => x.text_recall))}。`
    + `差し込む文章の長さは平均 ${avg(results, (x) => x.text_len).toFixed(0)}文字(部分ごと: ${parts})。`,
    `過去の記憶が要らなかったやり取り(会話の途中、${idle.length}件)では、平均 ${avg(idle, (x) => x.text_len).toFixed(0)}文字を差し込み、${pct(avg(idle, (x) => (x.text_len === 0 ? 1 : 0)))}で何も差し込まなかった。`, '');
  // 確信度: 前半で ACT-R の検索確率の τ と s を推定し、後半で確かめる
  const half = Math.floor(results.length / 2);
  const fitData = results.slice(0, half).flatMap((x) => x.episodes);
  const [tau, s] = fitRetrieval(fitData.map((e: any) => e.activation), fitData.map((e: any) => e.relevant));
  const laterEps = results.slice(half).flatMap((x) => x.episodes);
  const prob = (a: number, t: number, s_: number) => 1 / (1 + Math.exp(-(a - t) / s_));
  const acc = (es: any[]) => (es.length ? pct(es.filter((e) => e.relevant).length / es.length) : '-');
  lines.push('## 確信度(思い出したやり取りが実際に関係していた割合)', '',
    `評価分の前半で τ と s を推定した(今の設定 τ=${P.EP_TAU}, s=${P.EP_S} → 推定 τ=${tau.toFixed(3)}, s=${s.toFixed(3)})。後半で当たり具合を比べる。`, '',
    '| 確信度 | 今の設定: 件数 | 今の設定: 正解率 | 推定後: 件数 | 推定後: 正解率 |', '|---|---|---|---|---|');
  for (const [lo, hi] of CONF_BINS) {
    const cur = laterEps.filter((e: any) => lo <= prob(e.activation, P.EP_TAU, P.EP_S) && prob(e.activation, P.EP_TAU, P.EP_S) < hi);
    const fit = laterEps.filter((e: any) => lo <= prob(e.activation, tau, s) && prob(e.activation, tau, s) < hi);
    lines.push(`| ${lo.toFixed(1)}〜${Math.min(hi, 1).toFixed(1)} | ${cur.length} | ${acc(cur)} | ${fit.length} | ${acc(fit)} |`);
  }
  return lines.join('\n') + '\n';
}
