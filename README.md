# ollama-novel-bench

Ollamaで動くローカルLLMを、日本語小説の執筆適性で無人比較する再現可能なベンチマーク基盤です。対象モデルを実行時に検出し、生成本文・thinking・速度・機械採点・盲検Judge採点を同じrunに保存します。

## 最新結果

- [smoke-20260913c（予備試験・ランキング用途不可）](results/smoke-20260913c/README.md)

smokeは接続、モデル制御、生成、Judge、集計、グラフのエンドツーエンド確認用です。モデル間の結論には使わないでください。

## まず動かす

前提は Python 3.11 以上、ローカルOllama、認証済みHermes Agentです。対象Ollamaは既定で `http://127.0.0.1:11434`、変更時は `OLLAMA_HOST` または `--base-url` を使います。

```bash
cd /home/ws2/repos/ollama-novel-bench
uv sync --extra dev
uv run novelbench env
uv run novelbench models --probe-thinking
```

smokeは1モデルだけを選びます。モデル名は自分の `novelbench models` の出力に置き換えてください。

```bash
uv run novelbench run --profile smoke --run-id smoke-20260912 --models gemma4:e2b-it-qat
```

実行後は `results/smoke-20260912/README.md` を開きます。中断後は同じrunを次で続行できます。

```bash
uv run novelbench resume smoke-20260912
```

## quick / full / optional EQ-Bench

通常のscreeningを開始する正確なコマンドは次です。`quick`はインストール済みの全generation-capableモデルを対象に、固定seedの小規模サンプルでモデルごとのthinking `min` / `max`を直列測定します。長時間実行なので、この実装セッションでは開始していません。

```bash
uv run novelbench run --profile quick --run-id quick-20260912
```

候補を絞った本番比較は `full`、英語のEQ-Bench Creative Writing v3互換データを使う実験は `eqcw` です。

```bash
uv run novelbench run --profile full --run-id full-20260912 --models model-a,model-b
uv run novelbench run --profile eqcw --run-id eqcw-20260912 --models model-a
```

`eqcw`は公式仕様の温度 `0.7`、`min_p 0.1`、32プロンプト×3 iterationを使います。ただしJudgeが公式推奨Judgeと異なるため、公式leaderboard互換値ではありません。結果はこのリポジトリ内の比較値としてだけ解釈してください。

## ベンチマーク構成

| 区分 | 内容 | 採点 |
|---|---|---|
| objective-ja | JMMLU、JCommonsenseQA、JamC-QAを優先。quick/smokeは決定論的抽出、fullは大きいサンプル | 正解率（0–1）を機械採点。JMMLU/JCommonsenseQA/JamC-QAの出典・ライセンスは [docs/sources.md](docs/sources.md) |
| JCQ | Hugging Face `nlp-waseda/JCQ`。7タスク、700問。quick/fullは各タスクから固定seedで抽出 | `fluency`, `flexibility`, `originality`, `elaboration`をJudgeが各0–10 |
| novel-ja | このリポジトリの公開独自課題10問。会話、show-don't-tell、視点、人物一貫性、情景、伏線、改稿、余韻、ジャンル制約を含む | `instruction_following`, `japanese_naturalness`, `prose_quality`, `show_dont_tell`, `character_emotion`, `coherence`, `originality`, `ending_resonance`を各0–10 |
| EQCW（optional） | EQ-Bench Creative Writing v3の英語プロンプト | 公式値ではなく、固定Judgeによる参考値 |

問題とrubricは `data/novel_ja.json` にあります。外部データセット本体は再配布せず、ユーザーキャッシュへ取得し、選択したレコードだけをrunの `data/items.jsonl` に固定保存します。

smoke/screeningでは、JamC-QAは軽量な `dev` split、fullでは `test` splitを使います。サンプルseedはプロファイルに保存されます。

## モデル検出とthinking min/max

固定のモデル名表には依存しません。

1. `/api/tags` でインストール済みモデルを取得し、capabilitiesがembeddingだけのモデルを除外します。
2. 各候補について `/api/show` を読み、capabilities・digest・parameter size・quantization等をmanifestへ保存します。
3. `thinking` capabilityがあるモデルには短い実行probeを行います。`think=false/true`、`think`の各level、必要なら `reasoning_effort`を実際にOllamaへ送り、受理・エラー・thinking本文の有無・長さ・リクエストを保存します。

定義は次の通りです。

- thinking非対応: `none`を1設定。
- binary thinking: `min=think:false`、`max=think:true`。
- levelを受理: `min=none/off相当`、`max=そのOllamaが実際に受理してthinking本文を返す最高level`（`high`、`xhigh`、`max`等）。
- 判定不能: manifestの `kind` とprobe結果を確認し、推測で能力を補いません。

通常プロファイルでは `temperature`、`top_p`、`top_k`等を送らず、モデル既定値を尊重します。再現性のためseedと `num_predict` はプロファイルで明示します。生成は `model -> thinking mode -> item` の完全直列です。

生成予算の扱いは固定です。最初のリクエストは常にプロファイルの `num_predict` をそのまま使います。応答本文が空（空白だけを含む）の場合だけ、thinkingが予算を使い切った可能性への救済として `num_predict` を2倍ずつ増やして再試行し、profileの `empty_content_rescue.max_multiplier`（通常はbaseの4倍）を上限にします。smokeは現在のgemma4:e2b-it-qatの長いthinkingだけのobjective応答も有限に救済できるよう16倍を設定しています。本文が一文字でも非空なら、`done_reason=length` でも通常の成功として保存し、勝手に長文化しません。HTTP/network retryは同じbudgetで行う別経路です。救済で試した各budgetとresponse summaryはgeneration recordに保存され、CSV・README・manifestのmin/max比較にも救済件数と試行回数を出します。

## Judge

主観Judgeは常に次のHermes Agent設定です。GPT-5.6 Solは使用しません。

```text
hermes --safe-mode --provider openai-codex -m gpt-5.6-luna --reasoning max -z <prompt>
```

Judge promptには対象モデル名、サイズ、thinking modeを渡しません。作品はdata boundaryで囲み、作品内の命令・採点要求・役割指定・メタ発言をJudgeへの命令として扱わないよう指定します。長さは統計化し、極端な長さだけでは加点しません。

出力は指定キーだけのJSONを要求します。parse失敗時は修復を1回試し、それでも失敗したら有限回retryし、最終的にerrorをJSONLへ記録して先へ進みます。Judge model/provider/reasoning、Hermes binary、prompt version、prompt SHA256、usage、stdout/stderrの相対パスを保存します。生成とjudgmentは別JSONLなので、`novelbench judge <run-id>` で再採点できます。

full候補のpairwise比較では、同じA/BをA/B順とB/A順の両方で採点し、position biasを抑えます。

## 結果の読み方

`results/<run-id>/README.md` に以下を生成します。

- novel-jaを主目的としたランキング（モデルごとにmaxを優先。ただしスコアがないmaxよりスコアのあるminを優先）
- モデル×thinking設定の品質、正解率、出力文字数、thinking文字数、tok/s、失敗率
- thinking maxの品質向上量と速度低下量
- objective / creativity（JCQ）/ novel / performanceを分離したCSVと可視化
- generationのplanned / pending / error、Judgeのpending / error
- 生成全文、thinking、Ollamaメトリクス、Judge JSONを追跡できる相対リンク

グラフはPNGとSVGで保存します。色だけに依存せず、軸ラベル、凡例、値、モデル名を表示します。`novel-ja`の平均は便利な主目的指標ですが、単一の恣意的な総合点で全能力を置き換えるものではありません。作品本文と `axis_scores.csv` も確認してください。Judgeは人間の文学的評価そのものではなく、サンプル数が少ない結果、文字数、モデルの欠測、Judgeの癖に影響されます。

## CLI

```text
novelbench env
novelbench models [--probe-thinking]
novelbench run --profile smoke|quick|full|eqcw --run-id ID [--models NAME,...] [--publish]
novelbench resume ID [--publish]
novelbench judge ID
novelbench report ID
novelbench publish ID
novelbench shortlist ID --top N
```

- `run`: モデル検出、probe、items固定、直列生成、Judge、reportを実行。
- `resume`: manifestとitemsを再利用し、terminal recordのあるitemを再実行しない。
- `judge`: 生成済みcreative itemを固定Judgeで再採点（既存judgmentを置き換えず追記）。
- `report`: JSONLからCSV、README、PNG/SVGを再生成。
- `shortlist`: ranking上位を `data/shortlist.json` に保存。pairwise自体はライブラリの `pairwise_compare` から実行できます。
- `publish`: report生成→対象runだけcommit→`main`へpush→remote SHAをread-back検証。run外のdirty tree、commit、push、検証失敗は明示的に記録します。

## ファイルと再現性

```text
configs/profiles.json       プロファイル、seed、件数、生成設定
 data/novel_ja.json         公開独自課題とrubric
src/novelbench/             Python 3.11+ src layout
results/<run-id>/manifest.json
results/<run-id>/data/items.jsonl
data/generations.jsonl       生成本文・thinking・API request・メトリクス
results/<run-id>/data/judgments.jsonl
results/<run-id>/data/*.csv
results/<run-id>/charts/*.png, *.svg
results/<run-id>/logs/      Hermes Judge stdout/stderr、usage
```

manifestにはhost/GPU、Python/Ollama/Hermes情報、container環境の安全なsubset、Ollama version、model digest/details、dataset snapshot SHA256、git commit、設定、probe結果を記録します。credential/API key/tokenは保存しません。環境変数のうち保存するのはGPU選択やOllama並列性などの非秘密値だけです。

## 開発

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
uv run python -m compileall -q src tests
```

## ライセンス

このリポジトリのコードと `data/novel_ja.json` はMITです。外部benchmark/datasetはそれぞれの原ライセンス、出典、利用条件に従ってください。詳細は [docs/sources.md](docs/sources.md) と [LICENSE](LICENSE) を参照してください。
