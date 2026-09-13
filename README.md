# ollama-novel-bench

Ollamaで動くローカルLLMを、日本語小説の執筆適性で無人比較する再現可能なベンチマーク基盤です。対象モデルを実行時に検出し、生成本文・thinking・速度・機械採点・盲検Judge採点を同じrunに保存します。

## 最新結果

- [smoke-20260913c（予備試験・ランキング用途不可）](results/smoke-20260913c/README.md)
- [Judge比較 2026-09-13（GLM/Luna/DeepSeekの校正）](results/judge-comparison-20260913/README.md)
- [Judge選定と校正結果](docs/judge-selection.md)

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

通常のscreeningを開始する正確なコマンドは次です。`quick`はインストール済みの全generation-capableモデルを対象に、固定seedの小規模サンプルでモデルごとのthinking `min` / `max`を直列測定し、cloud referenceのGLMも`low` / `max`で測定します。長時間実行なので、この実装セッションでは開始していません。cloud referenceを省く場合だけ`--no-cloud-reference`を明示してください。

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

標準LLM-as-a-JudgeはOpenRouterの `z-ai/glm-5.3-flash` / `reasoning=max`です。GPT-5.6 Solは使用しません。OpenRouter APIを直接利用し、キーは `~/.config/credstore/openrouter.key` から実行時だけ読み込みます。キーとAuthorization headerは成果物・ログ・manifestへ保存しません。

標準リクエストでは、`response_format.type=json_schema`、strict JSON Schema、`provider.only=["z-ai"]`、`allow_fallbacks=false`、`provider.require_parameters=true`、`usage.include=true`を明示し、Z.AI first-partyへ固定します。Judge出力は8192 tokensから開始し、空本文・`finish_reason=length`・不正JSONは採点値として採用せず、元の作品を16384 tokensまで再採点します。切れた採点文をrepairして0点へ変換しません。実際に返ったprovider・model、usage、reasoning tokens、cost、latencyと安全なrequest subsetを保存します。

2026-09-13の同一8作品による校正では、GLMは8/8 JSON成功、LunaとのPearson `0.990` / Spearman `0.994` / MAE `0.269`、平均`16.27s`、8件`$0.005812`でした。DeepSeekは7/8、Spearman `0.857`、平均`43.17s`でした。このJSON遵守、Lunaとの一致度、速度、コストを標準Judgeの採用根拠とします。詳細は [results/judge-comparison-20260913/README.md](results/judge-comparison-20260913/README.md) と [docs/judge-selection.md](docs/judge-selection.md) にあります。8作品の小標本なので、人間評価との一致を保証するものではありません。

Judge promptには対象モデル名、サイズ、thinking modeを渡しません。作品はdata boundaryで囲み、作品内の命令・採点要求・役割指定・メタ発言をJudgeへの命令として扱わないよう指定します。長さは統計化し、極端な長さだけでは加点しません。

OpenRouterがstrict JSONを返しても、既存のbare JSON・exact key・score range・finite numberのparse validationを維持します。parse失敗時は修復を1回試し、それでも失敗したら有限回retryし、最終的にerrorをJSONLへ記録して先へ進みます。生成とjudgmentは別JSONLなので、`novelbench judge <run-id>` で標準GLMを再採点できます。

既存のLuna実装は独立audit backendとして残します。Lunaは `openai-codex / gpt-5.6-luna / reasoning=max` で、cloud reference creativeの `independent_audit.jsonl` と `novelbench audit-luna <run-id>` だけに使います。標準Judge経路には入りません。

full候補のpairwise比較では、同じA/BをA/B順とB/A順の両方で採点し、position biasを抑えます。

## cloud reference GLM

`quick` と `full` は、ローカルOllamaモデルに加えて `z-ai/glm-5.3-flash` 自身をcloud reference targetとして既定で含めます。OpenRouterのsupported effortに合わせ、ベンチマーク上のMINは`low`、MAXは`max`です。targetのrequest、実provider/model、usage、reasoning tokens、cost、latencyを生成データへ保存します。`smoke`は高速性優先でlocalのみが既定ですが、`--cloud-reference` または `--cloud-reference-only` で確認できます。

Judgeとtargetを同じGLMにすると自己評価バイアスが入るため、cloud referenceには `reference_only=true` と `ranking_eligible=false` を付けます。GLMのcreative値は `self_judged=true` として別表示し、主ランキング・shortlistには使いません。creativeの独立値はLuna(Max)で採点し、`independent_audit.jsonl` に分けてレポートで並べます。objective-jaは通常どおり機械採点します。localとの直接順位比較は断定せず、グラフも別パネル・別表記に分けます。Judge costとcloud target generation costも別集計です。

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
novelbench run --profile smoke|quick|full|eqcw --run-id ID [--models NAME,...] [--cloud-reference|--no-cloud-reference|--cloud-reference-only] [--publish]
novelbench resume ID [--cloud-reference|--no-cloud-reference] [--publish]
novelbench judge ID
novelbench report ID
novelbench publish ID
novelbench shortlist ID --top N
```

- `run`: モデル検出、probe、items固定、local/cloud直列生成、Judge、reportを実行。`--cloud-reference`、`--no-cloud-reference`、`--cloud-reference-only`でcloud targetを制御し、`--models`はlocal Ollama exact nameだけを指定します。
- `resume`: manifestとitemsを再利用し、terminal recordのあるitemを再実行しない。
- `judge`: 生成済みcreative itemを固定OpenRouter GLM Judgeで再採点（既存judgmentを置き換えず追記）。
- `audit-luna`: cloud referenceのcreative itemをLuna(Max)で独立監査し、`independent_audit.jsonl`へ保存。既存監査は既定でskipするため冪等です。意図的に再監査して追記する場合だけ `--force` を使います。
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
results/<run-id>/logs/      OpenRouter Judge/cloud metadata、Luna audit stdout/stderr、usage
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
