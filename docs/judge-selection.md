# Judge選定と校正結果

## 結論

標準のLLM-as-a-Judgeは OpenRouter の `z-ai/glm-5.3-flash`、`reasoning.max` に固定する。GPT-5.6 Solは使用しない。

OpenRouterは直接HTTPで呼び出し、認証キーは実行時に `~/.config/credstore/openrouter.key` から読み込む。キーとAuthorization headerはJSONL、manifest、ログ、README、比較結果へ保存しない。Judgeリクエストは strict JSON Schema の `response_format`、`provider.only=["z-ai"]`、`allow_fallbacks=false`、`provider.require_parameters=true`、`usage.include=true` を明示し、Z.AI first-partyへ固定する。出力予算は8192 tokensから開始し、空本文・`finish_reason=length`・不正JSONは採点値として採用せず、元の作品を16384 tokensまで再採点する。requested/actual model・provider、usage、reasoning tokens、cost、latencyと安全なrequest subsetを保存し、prompt本文はhashだけを安全なrequest metadataへ残す。

## 2026-09-13 校正試験

`results/judge-comparison-20260913/` に、`smoke-20260913c` の同じ8作品を同じrubric・blind条件で再採点した実測を保存している。生成は再実行していない。OpenRouter候補はstrict JSON Schemaと`reasoning=max`、必要パラメータ対応endpointのprice routingで実行した。

| Judge | JSON成功 | LunaとのPearson | LunaとのSpearman | LunaとのMAE | 平均latency | 8件の実コスト |
|---|---:|---:|---:|---:|---:|---:|
| GLM `z-ai/glm-5.3-flash` | 8/8 | 0.990 | 0.994 | 0.269 | 16.27s | $0.005812 |
| DeepSeek `deepseek/deepseek-v4.1-flash` | 7/8 | 0.927 | 0.857 | 0.821 | 43.17s | $0.005945 |
| Luna `gpt-5.6-luna` | 8/8 | — | — | — | 18.26s | Included |

GLMは8件すべてで厳密JSONを返し、Lunaとの相関・誤差が最も良く、DeepSeekより速く、8件の実コストも低かった。この校正結果を標準Judgeの採用根拠とする。ただし8作品の小標本なので、人間の文学的評価との一致や大規模runでの安定性を証明するものではない。quick/fullでは欠測、retry、prompt injection、blind境界、結果本文も引き続き確認する。

比較結果の詳細は [results/judge-comparison-20260913/README.md](../results/judge-comparison-20260913/README.md) を参照する。再実行スクリプトは [`scripts/compare_judges.py`](../scripts/compare_judges.py) で、次のコマンドを使う。

```text
cd /home/ws2/repos/ollama-novel-bench
uv run python scripts/compare_judges.py
```

スクリプトは `~/.config/credstore/openrouter.key` をメモリへ読み込み、キーやAuthorization headerを結果・ログへ書き出さない。比較の再実行はOpenRouter APIの利用料金を発生させる。

## 固定する役割

| 役割 | backend/provider | model | reasoning | 用途 | ranking |
|---|---|---|---|---|---|
| 標準Judge | OpenRouter / `openrouter` | `z-ai/glm-5.3-flash` | `max` | localとcloud referenceのprimary creative採点 | localのみ使用 |
| Luna監査 | Hermes / `openai-codex` | `gpt-5.6-luna` | `max` | cloud reference creativeの独立監査。旧比較の基準 | primary rankingには使用しない |
| Sol | — | — | — | 使用禁止 | — |

既存のLuna Judge実装は `audit-luna` とcloud referenceの `independent_audit.jsonl` 用に独立して残す。標準の `judge` 経路はOpenRouter GLMだけを使う。

## cloud referenceの自己評価バイアス

quick/fullはデフォルトで、ローカルOllamaモデルとは別に `z-ai/glm-5.3-flash` をcloud reference targetとして測定する。OpenRouter metadataで対応するeffortは `low`、`high`、`max` なので、ベンチマーク上のMINを`low`、MAXを`max`としてリクエストする。target側のrequest、実際のprovider/model、usage、reasoning tokens、cost、latency、空本文budget救済履歴を生成recordに保存する。

GLMをJudgeとtargetの両方に使うと自己評価バイアスが入る。そのためcloud referenceには常に `reference_only=true`、`ranking_eligible=false` を付け、主ランキング・shortlist・localモデルの順位へ混ぜない。creativeについてはGLMの値を `self_judged=true` と明示して保存し、独立した `independent_audit.jsonl` にはLuna(Max)の値を保存する。レポートではGLM自己採点とLuna監査値を並べ、別図・別行で表示する。

objective-jaはGLM自己評価を使わず、target本文を通常の機械採点へ通す。cloud referenceとlocalの直接順位比較はJudgeが同じでない場合も含めて断定しない。`--models` は従来どおりlocal Ollamaのexact nameだけを指定し、cloud referenceは `--cloud-reference`、`--no-cloud-reference`、`--cloud-reference-only` で独立に制御する。

## 再現性と再採点

- `quick` と `full` はcloud referenceを既定で有効化する。`smoke` はlocalのみを既定とし、必要時に `--cloud-reference` または `--cloud-reference-only` を付ける。
- `novelbench judge RUN_ID` は標準GLMでcreative generationを再採点する。既存JSONLを上書きせず追記する。
- `novelbench audit-luna RUN_ID` はcloud reference creativeだけをLuna(Max)で独立監査する。既存監査は既定でskipし、明示的な再監査だけ `--force` で追記する。
- Judge costとcloud target generation costはreportの `costs.primary_judge_usd` と `costs.cloud_reference_generation_usd` で分離する。
- `response_format`によるstrict schemaを使っても、bare JSON、exact key、score range、finite numberのvalidationを省略しない。不完全なprimary Judge出力はrepairで推測補完せず、元のblind採点を再実行する。
