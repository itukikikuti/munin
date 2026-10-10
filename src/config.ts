// 記憶のデータの置き場所と設定
//
// プログラム(プラグインとして入れると、更新のたびに入れ替わるフォルダに置かれる)とは分けて、
// ユーザーのフォルダの下(~/.munin)に置く。MUNIN_HOME で変えられる。
//
//   brain_state.json   記憶そのもの(皮質・海馬・痕跡・やること)
//   life_state.json    続きから覚えるための、体験の途中の状態
//   records.sqlite     やり取りの全文と検索用の索引
//   vectors/           意味のベクトル(作り直せるキャッシュ)
//   runtime/ models/   埋め込みを計算する llama.cpp とモデル(最初の準備で取ってくる)
//   layout.json        ネットワーク図の配置
//   logs/              使ったときの記録(差し込んだ内容・覚えた件数・エラー)
//   config.json        設定(使う埋め込みモデルなど)

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const HOME = process.env.MUNIN_HOME || path.join(os.homedir(), '.munin');
export const STATE = process.env.BRAIN_STATE || path.join(HOME, 'brain_state.json');
export const DATA = path.dirname(path.resolve(STATE));
export const LIFE = path.join(DATA, 'life_state.json');
export const RECORDS = path.join(DATA, 'records.sqlite');
export const VECTORS = path.join(DATA, 'vectors');
export const LOG_DIR = path.join(DATA, 'logs');
export const CONFIG = path.join(DATA, 'config.json');
/** プログラムの置き場所(src の1つ上) */
export const CODE = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
export const PORT = Number(process.env.PORT || 8770);

export interface Config {
  /** 概念と依頼の意味に使うモデル(GPU がなければ軽い 0.6b にできる) */
  embed_model?: string;
  /** 実際に使い始めた日時。分析(usage)は、指定がなければこれ以降の記録を集計する */
  trial_start?: string;
}

export function config(): Config {
  try {
    return JSON.parse(fs.readFileSync(CONFIG, 'utf8'));
  } catch {
    return {};
  }
}

export function saveConfig(c: Config): void {
  fs.mkdirSync(DATA, { recursive: true });
  fs.writeFileSync(CONFIG, JSON.stringify(c, null, 2));
}

/** 記録に1行足す(logs/ の下) */
export function logLine(file: string, message: string): void {
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.appendFileSync(path.join(LOG_DIR, file), `${stamp()} ${message}\n`);
  } catch {
    // 記録に失敗しても止めない
  }
}

export function appendJsonl(file: string, entry: unknown): void {
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.appendFileSync(path.join(LOG_DIR, file), JSON.stringify(entry) + '\n');
  } catch {
    // 同上
  }
}

/** 2026-10-10T12:34:56+0900 の形(その PC の時刻) */
export function stamp(d = new Date()): string {
  const off = -d.getTimezoneOffset();
  const pad = (n: number, w = 2) => String(Math.abs(n)).padStart(w, '0');
  const local = new Date(d.getTime() + off * 60000).toISOString().slice(0, 19);
  return `${local}${off >= 0 ? '+' : '-'}${pad(Math.trunc(off / 60))}${pad(off % 60)}`;
}
