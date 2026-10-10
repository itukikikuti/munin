// 記憶の鍵から、そのやり取りの記録を読める形にする(recall_detail・画面・コマンドで使う)

import type { Rec } from './store.ts';

const SHORT = 800;   // 全部を見ないときに表示する長さ

export function render(rec: Rec, full = false): string {
  const cut = (t: string) => (full || t.length <= SHORT ? t : `${t.slice(0, SHORT)}…(残り${t.length - SHORT}文字)`);
  const src = rec.src ?? {};
  const lines = [`日時: ${rec.time ?? ''}`, `作業フォルダ: ${rec.cwd ?? ''}`, `元の記録: ${src.file ?? ''} (発言のID ${src.uuid ?? ''})`,
    '', '## 依頼', cut(rec.prompt), '', '## やり取り'];
  for (const step of rec.steps) {
    if (step.kind === 'text') lines.push('', '### 返答', cut(step.text ?? ''));
    else if (step.kind === 'thinking') {
      if (full) lines.push('', '### 考えたこと', step.text ?? '');   // 考えた過程は長いので、全部を見るときだけ
    } else if (step.kind === 'interrupt') lines.push('', '### (ユーザーがここで止めた)');
    else {
      lines.push('', `### ${step.summary}`);
      if (step.said) lines.push(`ユーザーの言葉: ${cut(step.said)}`);
      if (step.result) lines.push((step.error ? '(エラー) ' : '') + cut(step.result));
    }
  }
  return lines.join('\n');
}
