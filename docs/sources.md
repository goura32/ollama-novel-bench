# Sources, licenses, and provenance

このリポジトリのコードと独自課題はMITです。以下の外部資料・データセットはMITではありません。各原典のライセンス、帰属表示、非商用・改変禁止等の条件を優先してください。

## Ollama

- [Thinking capability documentation](https://docs.ollama.com/capabilities/thinking) — `think` のboolean/level制御、chat responseの `message.thinking` と本文フィールドの一次資料。
- [OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility) — Ollamaが提供する互換APIの説明。本プロジェクトは互換層ではなく、監査しやすいnative `/api/tags`、`/api/show`、`/api/version`、`/api/chat`を使う。

## 日本語の客観・創造性ベンチマーク

### JCQ

- [Hugging Face dataset card: nlp-waseda/JCQ](https://huggingface.co/datasets/nlp-waseda/JCQ) — 7タスク各100問、合計700問、データフィールド、評価軸、ライセンス。
- [ACL paper](https://aclanthology.org/2025.acl-srw.69/) — JCQを含む日本語創造性ベンチマークの研究論文。
- [JCQ judge repository](https://github.com/nlp-waseda/JCQ-judge) — 原典の評価実装へのリンク。
- ライセンス: dataset cardの記載に従い **CC BY 4.0**。runで使ったsnapshotのURLとSHA256は `manifest.json` に保存する。

### JMMLU

- [JMMLU repository](https://github.com/nlp-waseda/JMMLU) — 日本語4択ベンチマークとデータ形式、課題一覧、ライセンス注記。
- [JMMLU Hugging Face dataset](https://huggingface.co/datasets/nlp-waseda/JMMLU) — 実行時に取得する配布元。
- ライセンス: upstreamのREADMEにある通り、**CC BY-SA 4.0のタスク群**と、**CC BY-NC-ND 4.0のタスク群**がある。配布物全体を一律にCC BY-SAとして扱わない。商用利用・改変・再配布が必要な場合は、該当タスクの原条件を確認する。

### JCommonsenseQA / JGLUE

- [JGLUE repository](https://github.com/yahoojapan/JGLUE) — JCommonsenseQAの仕様、データ形式、引用、ライセンス。
- [JCommonsenseQA valid snapshot](https://github.com/yahoojapan/JGLUE/blob/v1.2.0/datasets/jcommonsenseqa-v1.2/valid-v1.2.json) — 本プロジェクトの固定tag URL。
- ライセンス: JGLUEの記載に従い **CC BY-SA 4.0**。元データ由来の条件がある場合はJGLUEの原典を優先する。

### JamC-QA

- [Hugging Face dataset card: sbintuitions/JamC-QA](https://huggingface.co/datasets/sbintuitions/JamC-QA) — Japan-specific multiple-choice QA、split、項目数、形式、評価方法、引用。
- [JamC-QA paper information](https://lrec.elra.info/lrec2026-main-356) — 原論文へのリンク。
- ライセンス: dataset cardの記載に従い **CC BY-SA 4.0**。

### llm-jp-eval

- [llm-jp/llm-jp-eval](https://github.com/llm-jp/llm-jp-eval) — JMMLU/JCommonsenseQA/JamC-QAを採用する背景と、日本語評価データの横断利用方法を確認するための一次資料。

## Optional: EQ-Bench Creative Writing v3

- [EQ-Bench Creative Writing v3 repository](https://github.com/EQ-bench/creative-writing-bench) — 32 prompts×3 iterations、temperature `0.7`、`min_p 0.1`、公式のrubric/Elo手順の説明。
- ライセンス: upstream repositoryと同梱データのライセンス表記を実行時に確認する。ここではプロンプト本体を再配布せず、ユーザーキャッシュから取得する。
- 注意: 本プロジェクトの標準Judgeは OpenRouter `z-ai/glm-5.3-flash` / `reasoning=max` で、Luna(Max)はcloud referenceの独立監査専用です。したがって `eqcw` の値は公式leaderboard互換ではありません。

## 本プロジェクトの取得・保存方針

外部データは `~/.cache/ollama-novel-bench/datasets`（または `--cache-dir`）へ取得します。HTTP取得時のresolved URL、取得時刻、サイズ、SHA256を保存し、run manifestにはsource URL・version・license・snapshot SHA256を記録します。再開時にはrun内の `data/items.jsonl` を使い、設問の再抽出をしません。

生データ全体はリポジトリへコミットしません。監査と再現性のため、実際に選択した最小限の設問レコードだけをrunごとに保存します。利用者は各外部datasetの帰属表示・共有条件・商用利用条件を確認してください。
