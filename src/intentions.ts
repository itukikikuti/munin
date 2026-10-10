// 展望記憶: これからやること(やり残し・期限・返事待ち)を覚え、終わったら忘れる
//
// 過去の出来事を思い出す記憶(エピソード記憶)とは別に、人は「あとで〜する」「〜日までに〜する」という
// 未来の行動の記憶を持っている(展望記憶)。依頼の意味が近いかどうかでは思い出せない(「次なにする？」には
// 中身を示す語がない)ので、別の仕組みとして持つ。LLM は使わず、文の形から取り出す。
//
//   取り出す : Claude の返答と依頼から、期限のある文・保留や未対応の文・人への連絡を頼む文
//   完了     : 後のやり取りで、同じ件(概念が重なる)について「完了」「直しました」などが出てきたら完了にする
//              (思い出したときに記憶が書き換わる、再固定化にあたる)
//   思い出す : 新しい会話の最初と、「やり残し」「次なにする」「期限」のような依頼のときに、
//              期限の近さ・依頼との関係・新しさで並べる

// 期限: 「10月12日までに」「10/15 まで」「期限は10月20日」など
const DUE = /(?:(\d{4})[年/-])?(\d{1,2})(?:月|\/)(\d{1,2})日?(?:\([^)]*\))?\s*(?:まで|迄|が期限|が締め切り|締め切り|〆)|(?:期限|締め切り|〆切)\s*(?:は|:|：)?\s*(?:(\d{4})[年/-])?(\d{1,2})(?:月|\/)(\d{1,2})/;
// 保留・未対応・あとでやること
// 「残っている」は「振幅が残っている」のように、やり残し以外の意味でもよく使われるので条件にしない
// 「宿題」は研修の受講生の宿題、「未検証」は性質の説明でもよく使われるので条件にしない
const PENDING = /未対応|未完了|未着手|保留(?:に|の|中|して|します)|後回し|残タスク|やり残し|持ち越し|返事待ち|確認待ち|回答待ち|返信待ち|TODO|あとで(?:やる|対応|確認)|次回(?:やる|対応|確認)|まだ(?:終わって|できて|対応して|反映して|手を付けて|確認して)(?:い)?ません/;
const SUMMARY = /(?:決定|ToDo|保留)\s*\d+件/;   // 議事録の件数のまとめ(「決定9件・保留14件」)は、やることそのものではない
// 人への連絡・確認を頼む文(誰に、が書かれているものだけ。「画面を確認してください」のような軽いものは除く)
const ASK_PERSON = /[一-龥ぁ-んァ-ンA-Za-z]{1,8}(?:さん|課長|部長|係長|主任|先生|様)(?:に|へ)[^。]{0,40}(?:確認|連絡|相談|打診|依頼|報告|共有|提出|送付|聞いて|伝えて)/;
// 完了を表す言い方
const DONE = /完了|終わりました|終わった|直しました|直した|対応しました|対応済み|済みです|済んだ|反映しました|提出しました|送りました|送付しました|片付き|解決しました|できました|コミットしました|取り込みました/;
// 「やり残し」「次なにする」「期限」のような、これからやることを尋ねる依頼
const ASKS_TODO = /やり残|残って(?:る|い)|次(?:は|、)?(?:何|なに)|何(?:を)?(?:したら|すれば)|やること|やるべき|タスク|期限|締め切り|〆切|忘れて(?:る|い)|TODO|予定/;
const SENTENCE = /[^。\n!?！？]+[。!?！？]?/g;
const MAX_PER_TURN = 2;   // 1回のやり取りから取り出す数(多すぎると、どれも大事でなくなる)
const MAX_CHARS = 150;

export interface Intent {
  text: string;
  due: string | null;
  kind: '期限' | '保留' | '連絡';
  source: '依頼' | '返答';
}

export function clean(text: string): string {
  return text
    .replace(/```[\s\S]*?```/g, ' ')            // コードの部分は除く
    .replace(/`([^`]*)`/g, '$1')
    .replace(/\*\*|__|^#+\s*|^\s*[-*]\s+|^\s*\d+\.\s+/gm, '');
}

const iso = (y: number, m: number, d: number) =>
  `${String(y).padStart(4, '0')}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`;

/** 暦の日付の差(日)。a - b */
export function daysBetween(a: string, b: string): number {
  return Math.round((Date.parse(a + 'T00:00:00Z') - Date.parse(b + 'T00:00:00Z')) / 86400000);
}

function validDate(y: number, m: number, d: number): boolean {
  const t = new Date(Date.UTC(y, m - 1, d));
  return t.getUTCFullYear() === y && t.getUTCMonth() === m - 1 && t.getUTCDate() === d;
}

/** 期限の文字列から日付を作る(年がなければ、そのやり取りの日付から見て近い方の年) */
export function dueDate(m: RegExpMatchArray, base: string): string | null {
  const [year, month, day] = m[2] ? [m[1], m[2], m[3]] : [m[4], m[5], m[6]];
  let y = year ? Number(year) : Number(base.slice(0, 4));
  if (!validDate(y, Number(month), Number(day))) return null;
  let d = iso(y, Number(month), Number(day));
  if (!year && daysBetween(d, base) < -60) {   // 1月に「12月までに」とは言わないので、来年の話とみなす
    y += 1;
    if (!validDate(y, Number(month), Number(day))) return null;
    d = iso(y, Number(month), Number(day));
  }
  return d;
}

/** やり取りから、これからやることを取り出す。base はそのやり取りの日付(YYYY-MM-DD) */
export function extract(prompt: string, replies: string[], base: string): Intent[] {
  const found: Intent[] = [];
  const sources: [Intent['source'], string][] = [['依頼', prompt], ...replies.map((r): [Intent['source'], string] => ['返答', r])];
  for (const [source, text] of sources) {
    for (const raw of clean(text).match(SENTENCE) ?? []) {
      const s = raw.split(/\s+/).filter(Boolean).join(' ');
      if ([...s].length < 8 || s.endsWith('?') || s.endsWith('？') || s.includes('ますか')) continue;   // 質問や提案(〜しますか)は、やることではない
      const m = s.match(DUE);
      if (m) {
        const d = dueDate(m, base);
        if (d && daysBetween(d, base) >= -1) {   // 過去の日付は、やることではない
          found.push({ text: [...s].slice(0, MAX_CHARS).join(''), due: d, kind: '期限', source });
          continue;
        }
      }
      if (source === '返答' && PENDING.test(s) && !SUMMARY.test(s)) {
        found.push({ text: [...s].slice(0, MAX_CHARS).join(''), due: null, kind: '保留', source });
      } else if (source === '返答' && ASK_PERSON.test(s)) {
        found.push({ text: [...s].slice(0, MAX_CHARS).join(''), due: null, kind: '連絡', source });
      }
    }
  }
  const order = { 期限: 0, 保留: 1, 連絡: 2 };
  const unique = [...new Map(found.map((f) => [f.text, f])).values()];
  return unique.sort((a, b) => order[a.kind] - order[b.kind]).slice(0, MAX_PER_TURN);
}

export const mentionsDone = (text: string) => DONE.test(text);
export const asksTodo = (text: string) => ASKS_TODO.test(text);
