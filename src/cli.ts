// munin のコマンド
//
//   node src/cli.ts setup [--full|--light]   最初の準備(llama.cpp とモデルを取ってきて、最初の記憶を作り始める)
//   node src/cli.ts import                   すべてのセッション記録から記憶を作り直す
//   node src/cli.ts server                   常駐サーバーを起動する(http://127.0.0.1:8770 で記憶を見られる)
//   node src/cli.ts detail 鍵 [--full]       鍵から記録を開く
//   node src/cli.ts eval ...                 自動評価(evaluate.ts を参照)
//   node src/cli.ts compare 名前1 名前2       評価結果を同じ問題どうしで比べる
//   node src/cli.ts usage [--since 日付]      実際に使ったときに、差し込んだ記憶が使われたかを調べる

const [cmd, ...rest] = process.argv.slice(2);
switch (cmd) {
  case 'setup':
    process.exitCode = await (await import('./setup.ts')).setup(rest);
    break;
  case 'import':
    await (await import('./sessions.ts')).importAll();
    break;
  case 'server':
    await import('./server.ts');
    break;
  case 'detail': {
    const { Store } = await import('./store.ts');
    const { RECORDS } = await import('./config.ts');
    const { render } = await import('./detail.ts');
    const store = new Store(RECORDS);
    const key = store.find(String(rest[0] ?? '').replace(/[\[\]\s]/g, ''));
    console.log(key ? render(store.record(key)!, rest.includes('--full')) : `鍵 ${rest[0]} の記録が見つからない(または1件に決まらない)`);
    break;
  }
  case 'eval':
    await (await import('./evaluate.ts')).evaluate(rest);
    break;
  case 'compare':
    (await import('./compare.ts')).compare(rest);
    break;
  case 'usage':
    await (await import('./usage.ts')).usage(rest);
    break;
  default:
    console.log('使い方: node src/cli.ts setup | import | server | detail 鍵 | eval | compare 名前1 名前2 | usage');
}
