"""Machine-readable aggregation and target-aware charts for a benchmark run."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from .cloud import CLOUD_REFERENCE_MODEL
from .judge import NOVEL_CRITERIA
from .store import RunStore
from .util import utc_now

SUMMARY_FIELDS = [
    "backend",
    "reference_only",
    "ranking_eligible",
    "model",
    "parameter_size",
    "mode",
    "thinking_kind",
    "provider",
    "provider_actual",
    "model_actual",
    "generation_total",
    "generation_ok",
    "generation_failed",
    "failure_rate",
    "target_generation_cost_usd",
    "target_latency_s_mean",
    "budget_rescue_n",
    "budget_rescue_attempts",
    "budget_rescue_rate",
    "output_chars_mean",
    "thinking_chars_mean",
    "tok_s_mean",
    "judge_backend",
    "judge_provider",
    "judge_model",
    "judge_reasoning",
    "judge_n",
    "judge_cost_usd",
    "self_judged_score",
    "self_judged_n",
    "luna_audit_score",
    "luna_audit_n",
    "independent_audit_pending",
    "objective_ja_score",
    "objective_ja_n",
    "jcq_score",
    "jcq_n",
    "novel_primary_score",
    "novel_n",
    "eqcw_score",
    "eqcw_n",
]


def _mean(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    return mean(numeric) if numeric else None


def _sum_known(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    ]
    return sum(numeric) if numeric else None


def _rescue_attempt_count(record: dict[str, Any]) -> int:
    rescue = record.get("empty_content_rescue")
    if not isinstance(rescue, dict) or not rescue.get("used"):
        return 0
    count = rescue.get("rescue_request_count")
    if isinstance(count, int) and not isinstance(count, bool):
        return max(count, 0)
    summaries = rescue.get("response_summaries")
    return max(len(summaries) - 1, 0) if isinstance(summaries, list) else 0


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _short_model(name: str, width: int = 24) -> str:
    return name if len(name) <= width else f"…{name[-(width - 1) :]}"


def _is_reference(record: dict[str, Any]) -> bool:
    return bool(record.get("reference_only", False))


def _is_ranking_eligible(record: dict[str, Any]) -> bool:
    return bool(record.get("ranking_eligible", not _is_reference(record)))


def _plot_empty(path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.text(0.5, 0.5, "データなし", ha="center", va="center", fontsize=16)
    axis.set_title(title)
    axis.axis("off")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _configure_plot_font(matplotlib: Any) -> None:
    from matplotlib import font_manager

    preferred = (
        "Noto Sans CJK JP",
        "Noto Serif CJK JP",
        "TakaoGothic",
        "TakaoMincho",
        "Droid Sans Fallback",
    )
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in available), None)
    if selected:
        matplotlib.rcParams["font.family"] = [selected, "DejaVu Sans"]


def _charts(
    run_dir: Path,
    ranking: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    axes: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    reference_summary: list[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chart_dir = run_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    _configure_plot_font(matplotlib)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 13, "figure.titlesize": 14})

    ranking_valid = [row for row in ranking if row.get("novel_primary_score") is not None]
    if ranking_valid:
        rows = list(reversed(ranking_valid))
        fig, axis = plt.subplots(figsize=(10, max(4.5, len(rows) * 0.45)))
        values = [float(row["novel_primary_score"]) for row in rows]
        labels = [f"{row['model']} [{row['mode']}]" for row in rows]
        bars = axis.barh(range(len(rows)), values, color="#4c78a8", edgecolor="black")
        axis.set_yticks(range(len(rows)), labels)
        axis.set_xlim(0, 10)
        axis.set_xlabel("novel-ja 平均 (0–10)")
        axis.set_title("A. local novel-ja 主目的ランキング（GLM Judge）")
        for bar, value in zip(bars, values, strict=True):
            axis.text(min(value + 0.08, 9.7), bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
        fig.tight_layout()
        fig.savefig(chart_dir / "novel_ranking.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "novel_ranking.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "novel_ranking", "A. local novel-ja 主目的ランキング")
    generated += ["charts/novel_ranking.png", "charts/novel_ranking.svg"]

    if paired:
        rows = paired
        x = np.arange(len(rows))
        width = 0.36
        fig, (quality, speed) = plt.subplots(2, 1, figsize=(11, max(6, len(rows) * 0.8)), sharex=True)
        min_quality = [row.get("novel_min") if row.get("novel_min") is not None else np.nan for row in rows]
        max_quality = [row.get("novel_max") if row.get("novel_max") is not None else np.nan for row in rows]
        quality.bar(x - width / 2, min_quality, width, label="min", color="#9ecae1", edgecolor="black")
        quality.bar(x + width / 2, max_quality, width, label="max", color="#2171b5", edgecolor="black")
        quality.set_ylim(0, 10)
        quality.set_ylabel("novel-ja (0–10)")
        quality.set_title("B. local thinking 最小 vs 最大（paired）")
        quality.legend()
        for index, row in enumerate(rows):
            if row.get("quality_delta") is not None:
                quality.text(index, min(float(row.get("novel_min") or 0), 9.2), f"Δ{float(row['quality_delta']):+.2f}", ha="center", va="bottom", fontsize=9)
        min_speed = [row.get("tok_s_min") if row.get("tok_s_min") is not None else np.nan for row in rows]
        max_speed = [row.get("tok_s_max") if row.get("tok_s_max") is not None else np.nan for row in rows]
        speed.bar(x - width / 2, min_speed, width, label="min", color="#fdd0a2", edgecolor="black")
        speed.bar(x + width / 2, max_speed, width, label="max", color="#e6550d", edgecolor="black")
        speed.set_ylabel("生成速度 (tok/s)")
        speed.set_xticks(x, [_short_model(str(row["model"])) for row in rows], rotation=25, ha="right")
        speed.set_title("local生成速度（値は paired.csv）")
        speed.legend()
        fig.tight_layout()
        fig.savefig(chart_dir / "thinking_paired.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "thinking_paired.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "thinking_paired", "B. local thinking 最小 vs 最大")
    generated += ["charts/thinking_paired.png", "charts/thinking_paired.svg"]

    scatter_rows = [
        row for row in summary if _is_ranking_eligible(row) and row.get("novel_primary_score") is not None and row.get("tok_s_mean") is not None
    ]
    if scatter_rows:
        fig, axis = plt.subplots(figsize=(10, 6))
        for row in scatter_rows:
            axis.scatter(float(row["tok_s_mean"]), float(row["novel_primary_score"]), s=85, edgecolor="black", label=f"{row['model']} [{row['mode']}]")
            axis.annotate(f"{_short_model(str(row['model']), 14)}\n{row['mode']}", (float(row["tok_s_mean"]), float(row["novel_primary_score"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
        axis.set_xlabel("生成速度 (tok/s)")
        axis.set_ylabel("novel-ja 平均 (0–10)")
        axis.set_title("C. local品質スコア vs 生成速度（cloud referenceは別図）")
        if len(scatter_rows) <= 12:
            axis.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(chart_dir / "quality_speed_scatter.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "quality_speed_scatter.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "quality_speed_scatter", "C. local品質スコア vs 生成速度")
    generated += ["charts/quality_speed_scatter.png", "charts/quality_speed_scatter.svg"]

    top = ranking_valid[:5]
    top_keys = {(row["model"], row["mode"]) for row in top}
    grouped = [row for row in axes if _is_ranking_eligible(row) and row["benchmark"] == "novel-ja" and (row["model"], row["mode"]) in top_keys]
    if grouped:
        labels = [f"{row['model']} [{row['mode']}]" for row in top]
        values_by_key = {(row["model"], row["mode"], row["axis"]): row.get("score") for row in grouped}
        x = np.arange(len(labels))
        axis_names = list(NOVEL_CRITERIA)
        width = 0.8 / len(axis_names)
        fig, axis = plt.subplots(figsize=(max(11, len(labels) * 2.1), 6))
        for offset, axis_name in enumerate(axis_names):
            values = [values_by_key.get((row["model"], row["mode"], axis_name), np.nan) for row in top]
            axis.bar(x - 0.4 + width / 2 + offset * width, values, width, label=axis_name, edgecolor="black")
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylim(0, 10)
        axis.set_ylabel("スコア (0–10)")
        axis.set_title("D. local novel-ja 評価軸（上位5設定）")
        axis.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / "novel_axes.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "novel_axes.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "novel_axes", "D. local novel-ja 評価軸")
    generated += ["charts/novel_axes.png", "charts/novel_axes.svg"]

    category_rows = [row for row in categories if _is_ranking_eligible(row) and row.get("score") is not None]
    if category_rows:
        benchmarks = ["objective-ja", "jcq", "novel-ja"]
        fig, axes_plot = plt.subplots(1, 3, figsize=(17, 5), squeeze=False)
        for plot_axis, benchmark in zip(axes_plot[0], benchmarks, strict=True):
            subset = [row for row in category_rows if row["benchmark"] == benchmark]
            if not subset:
                plot_axis.text(0.5, 0.5, "データなし", ha="center", va="center")
                plot_axis.set_title(benchmark)
                plot_axis.axis("off")
                continue
            categories_unique = sorted({str(row["category"]) for row in subset})
            models_unique = sorted({(str(row["model"]), str(row["mode"])) for row in subset})[:5]
            x = np.arange(len(categories_unique))
            width = 0.8 / max(1, len(models_unique))
            for index, key in enumerate(models_unique):
                values = []
                for category in categories_unique:
                    match = next((row for row in subset if (row["model"], row["mode"]) == key and row["category"] == category), None)
                    value = match["score"] if match else np.nan
                    values.append(float(value) * 10 if benchmark == "objective-ja" else float(value))
                plot_axis.bar(x - 0.4 + width / 2 + index * width, values, width, label=f"{key[0]} [{key[1]}]", edgecolor="black")
            plot_axis.set_xticks(x, [_short_model(category, 14) for category in categories_unique], rotation=35, ha="right")
            plot_axis.set_ylim(0, 10)
            plot_axis.set_title(benchmark)
            plot_axis.set_ylabel("可視化スケール (0–10)")
            if len(models_unique) <= 5:
                plot_axis.legend(fontsize=7)
        fig.suptitle("E. local objective-ja / JCQ / novel-ja のカテゴリ別比較")
        fig.tight_layout()
        fig.savefig(chart_dir / "category_comparison.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "category_comparison.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "category_comparison", "E. localカテゴリ別比較")
    generated += ["charts/category_comparison.png", "charts/category_comparison.svg"]

    ref_rows = [
        row
        for row in reference_summary
        if row.get("self_judged_score") is not None
        or row.get("luna_audit_score") is not None
        or row.get("objective_ja_score") is not None
    ]
    if ref_rows:
        labels = [str(row["mode"]) for row in ref_rows]
        x = np.arange(len(labels))
        width = 0.25
        fig, (quality, objective) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        self_values = [row.get("self_judged_score") if row.get("self_judged_score") is not None else np.nan for row in ref_rows]
        audit_values = [row.get("luna_audit_score") if row.get("luna_audit_score") is not None else np.nan for row in ref_rows]
        quality.bar(x - width / 2, self_values, width, label="GLM self-judged", color="#756bb1", edgecolor="black")
        quality.bar(x + width / 2, audit_values, width, label="Luna(Max) independent audit", color="#31a354", edgecolor="black")
        quality.set_ylim(0, 10)
        quality.set_ylabel("creative score (0–10)")
        quality.set_title("F. cloud reference: GLM自己採点 vs Luna監査（ランキング外）")
        quality.legend()
        objective_values = [float(row["objective_ja_score"]) * 10 if row.get("objective_ja_score") is not None else np.nan for row in ref_rows]
        objective.bar(x, objective_values, width=0.5, color="#e6550d", edgecolor="black")
        objective.set_ylim(0, 10)
        objective.set_ylabel("objective accuracy × 10")
        objective.set_title("cloud reference objective-ja（機械採点）")
        objective.set_xticks(x, labels)
        fig.tight_layout()
        fig.savefig(chart_dir / "cloud_reference.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "cloud_reference.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "cloud_reference", "F. cloud reference")
    generated += ["charts/cloud_reference.png", "charts/cloud_reference.svg"]
    return generated


def _judge_model(record: dict[str, Any]) -> str | None:
    judge = record.get("judge")
    return judge.get("model") if isinstance(judge, dict) else None


def _judgment_key(record: dict[str, Any]) -> str:
    return str(record.get("generation_key") or record.get("key") or "")


def _legacy_scores_are_primary(manifest: dict[str, Any], record: dict[str, Any]) -> bool:
    if record.get("evaluator_role") == "primary":
        return True
    if record.get("evaluator_role") == "independent_audit":
        return False
    configured = (manifest.get("judge") or {}).get("model")
    if configured:
        return _judge_model(record) == configured
    return True


def _reference_audit_rows(
    generations: list[dict[str, Any]],
    primary_by_key: dict[str, dict[str, Any]],
    audit_by_key: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for generation in generations:
        if not _is_reference(generation) or generation.get("benchmark") not in {"jcq", "novel-ja", "eqcw"}:
            continue
        key = str(generation.get("key"))
        primary = primary_by_key.get(key, {})
        audit = audit_by_key.get(key, {})
        rows.append(
            {
                "generation_key": key,
                "model": generation.get("model"),
                "mode": generation.get("mode"),
                "benchmark": generation.get("benchmark"),
                "item_id": generation.get("item_id"),
                "self_judged": primary.get("self_judged") is True,
                "self_judged_status": primary.get("status"),
                "self_judged_overall": primary.get("overall"),
                "self_judged_model": _judge_model(primary),
                "independent_audit_status": audit.get("status"),
                "luna_audit_overall": audit.get("overall"),
                "independent_audit_model": _judge_model(audit),
            }
        )
    return rows


def build_report(run_dir: Path) -> dict[str, Any]:
    store = RunStore(run_dir)
    manifest = store.load_manifest()
    generations = list(store.read_jsonl("generations.jsonl"))
    all_judgments = list(store.read_jsonl("judgments.jsonl"))
    all_audits = list(store.read_jsonl("independent_audit.jsonl"))

    primary_by_key: dict[str, dict[str, Any]] = {}
    for record in all_judgments:
        key = _judgment_key(record)
        if key and _legacy_scores_are_primary(manifest, record):
            primary_by_key[key] = record
    audit_by_key: dict[str, dict[str, Any]] = {}
    for record in all_audits:
        key = _judgment_key(record)
        if key:
            audit_by_key[key] = record

    model_info = {str(model.get("name")): model for model in manifest.get("models", [])}
    mode_keys = {
        (str(record.get("model")), str(record.get("mode")))
        for record in generations
        if record.get("model") and record.get("mode")
    }
    for model in manifest.get("models", []):
        for mode in (model.get("thinking") or {}).get("modes", []):
            mode_keys.add((str(model.get("name")), str(mode.get("label"))))
    cloud_plan = manifest.get("cloud_reference") or {}
    if cloud_plan.get("enabled"):
        for mode in cloud_plan.get("modes", []):
            mode_keys.add((CLOUD_REFERENCE_MODEL, str(mode.get("label"))))

    item_ids = [str(item_id) for item_id in manifest.get("item_ids", [])]
    item_count = int(manifest.get("item_count") or len(item_ids) or len({str(record.get("item_id")) for record in generations if record.get("item_id")}))
    generation_planned = len(mode_keys) * item_count
    generation_keys = {str(record["key"]) for record in generations if record.get("key")}
    generation_pending = max(generation_planned - len(generation_keys), 0)
    creative_generation_keys = {
        str(record["key"])
        for record in generations
        if record.get("key") and record.get("status") == "ok" and record.get("benchmark") in {"jcq", "novel-ja", "eqcw"}
    }
    primary_keys = set(primary_by_key)
    judge_pending = len(creative_generation_keys - primary_keys)
    audit_pending = len(creative_generation_keys.intersection({str(record["key"]) for record in generations if record.get("key") and _is_reference(record)}) - set(audit_by_key))

    summary: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []
    category_accumulator: dict[tuple[str, str, str, str, bool, bool, str], list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for model, mode in sorted(mode_keys):
        rows = [record for record in generations if str(record.get("model")) == model and str(record.get("mode")) == mode]
        successful = [record for record in rows if record.get("status") == "ok"]
        reference_only = any(_is_reference(record) for record in rows) or model == CLOUD_REFERENCE_MODEL
        ranking_eligible = all(_is_ranking_eligible(record) for record in rows) if rows else not reference_only
        objective = [record for record in successful if record.get("benchmark") == "objective-ja" and isinstance(record.get("objective_score"), (int, float))]
        creative_by_benchmark: dict[str, list[dict[str, Any]]] = {}
        for benchmark in ("jcq", "novel-ja", "eqcw"):
            creative_by_benchmark[benchmark] = [
                primary_by_key[str(record.get("key"))]
                for record in successful
                if record.get("benchmark") == benchmark and str(record.get("key")) in primary_by_key and primary_by_key[str(record.get("key"))].get("status") == "ok"
            ]
        ref_generations = [record for record in successful if _is_reference(record)]
        self_judgments = [record for record in creative_by_benchmark.get("jcq", []) + creative_by_benchmark.get("novel-ja", []) + creative_by_benchmark.get("eqcw", []) if record.get("self_judged") is True]
        audits = [audit_by_key[str(record.get("key"))] for record in ref_generations if str(record.get("key")) in audit_by_key and audit_by_key[str(record.get("key"))].get("status") == "ok"]
        for judgment in creative_by_benchmark.get("jcq", []) + creative_by_benchmark.get("novel-ja", []) + creative_by_benchmark.get("eqcw", []):
            for axis, score in (judgment.get("scores") or {}).items():
                if isinstance(score, (int, float)):
                    axis_rows.append({
                        "benchmark": judgment.get("benchmark"),
                        "model": model,
                        "mode": mode,
                        "axis": axis,
                        "score": float(score),
                        "n": 1,
                        "backend": "openrouter" if reference_only else "ollama",
                        "reference_only": reference_only,
                        "ranking_eligible": ranking_eligible,
                        "evaluator": _judge_model(judgment),
                    })
                    category_accumulator[(str(judgment.get("benchmark")), model, mode, str(judgment.get("category", "all")), reference_only, ranking_eligible, str(_judge_model(judgment) or "unknown"))].append(float(score))
        for record in objective:
            category_accumulator[("objective-ja", model, mode, str(record.get("category", "all")), reference_only, ranking_eligible, "machine")].append(float(record["objective_score"]))

        info = model_info.get(model, {})
        judge_records = [primary_by_key[str(record.get("key"))] for record in successful if str(record.get("key")) in primary_by_key]
        audit_scores = [record.get("overall") for record in audits]
        summary.append({
            "backend": "openrouter" if reference_only else "ollama",
            "reference_only": reference_only,
            "ranking_eligible": ranking_eligible,
            "model": model,
            "parameter_size": info.get("parameter_size") or ("cloud" if reference_only else None),
            "mode": mode,
            "thinking_kind": (info.get("thinking") or {}).get("kind") or ("levels" if reference_only else None),
            "provider": next((record.get("provider") for record in rows if record.get("provider")), "openrouter" if reference_only else "ollama"),
            "provider_actual": ",".join(sorted({str(record.get("provider_actual")) for record in rows if record.get("provider_actual")})) or None,
            "model_actual": ",".join(sorted({str(record.get("model_actual")) for record in rows if record.get("model_actual")})) or (model if not reference_only else None),
            "generation_total": len(rows),
            "generation_ok": len(successful),
            "generation_failed": len(rows) - len(successful),
            "failure_rate": (len(rows) - len(successful)) / len(rows) if rows else None,
            "target_generation_cost_usd": _sum_known(record.get("cost_usd") for record in successful),
            "target_latency_s_mean": _mean(record.get("latency_s") for record in successful),
            "budget_rescue_n": sum(1 for record in rows if isinstance(record.get("empty_content_rescue"), dict) and record["empty_content_rescue"].get("used")),
            "budget_rescue_attempts": sum(_rescue_attempt_count(record) for record in rows),
            "budget_rescue_rate": sum(1 for record in rows if isinstance(record.get("empty_content_rescue"), dict) and record["empty_content_rescue"].get("used")) / len(rows) if rows else None,
            "output_chars_mean": _mean(record.get("content_length", len(str(record.get("content", "")))) for record in successful),
            "thinking_chars_mean": _mean(record.get("thinking_length", len(str(record.get("thinking", "")))) for record in successful),
            "tok_s_mean": _mean(record.get("tok_s") for record in successful),
            "judge_backend": next((_record.get("backend") for _record in judge_records if _record.get("backend")), "hermes" if judge_records and _judge_model(judge_records[0]) == "gpt-5.6-luna" else "openrouter" if judge_records else None),
            "judge_provider": next((_record.get("judge", {}).get("provider") for _record in judge_records if isinstance(_record.get("judge"), dict)), None),
            "judge_model": next((_judge_model(_record) for _record in judge_records if _judge_model(_record)), None),
            "judge_reasoning": next((_record.get("judge", {}).get("reasoning") for _record in judge_records if isinstance(_record.get("judge"), dict)), None),
            "judge_n": len(judge_records),
            "judge_cost_usd": _sum_known(record.get("cost_usd") for record in judge_records),
            "self_judged_score": _mean(record.get("overall") for record in self_judgments),
            "self_judged_n": len(self_judgments),
            "luna_audit_score": _mean(audit_scores),
            "luna_audit_n": len(audits),
            "independent_audit_pending": sum(1 for record in ref_generations if str(record.get("key")) not in audit_by_key),
            "objective_ja_score": _mean(record.get("objective_score") for record in objective),
            "objective_ja_n": len(objective),
            "jcq_score": _mean(record.get("overall") for record in creative_by_benchmark["jcq"]),
            "jcq_n": len(creative_by_benchmark["jcq"]),
            "novel_primary_score": _mean(record.get("overall") for record in creative_by_benchmark["novel-ja"]),
            "novel_n": len(creative_by_benchmark["novel-ja"]),
            "eqcw_score": _mean(record.get("overall") for record in creative_by_benchmark["eqcw"]),
            "eqcw_n": len(creative_by_benchmark["eqcw"]),
        })
        for record in rows:
            if record.get("status") != "ok":
                failures.append({"kind": "generation", "key": record.get("key"), "model": model, "mode": mode, "benchmark": record.get("benchmark"), "item_id": record.get("item_id"), "error": record.get("error")})
            judgment = primary_by_key.get(str(record.get("key")))
            if judgment and judgment.get("status") != "ok":
                failures.append({"kind": "judge", "key": record.get("key"), "model": model, "mode": mode, "benchmark": record.get("benchmark"), "item_id": record.get("item_id"), "error": judgment.get("error")})
            audit = audit_by_key.get(str(record.get("key")))
            if audit and audit.get("status") != "ok":
                failures.append({"kind": "independent_audit", "key": record.get("key"), "model": model, "mode": mode, "benchmark": record.get("benchmark"), "item_id": record.get("item_id"), "error": audit.get("error")})

    ranking: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary if row.get("ranking_eligible")}):
        candidates = [row for row in summary if row["model"] == model and row.get("ranking_eligible")]
        preferred = sorted(candidates, key=lambda row: (row.get("novel_primary_score") is not None, row["mode"] == "max", row.get("novel_primary_score") if row.get("novel_primary_score") is not None else -1), reverse=True)[0]
        ranking.append(dict(preferred))
    ranking.sort(key=lambda row: row.get("novel_primary_score") if row.get("novel_primary_score") is not None else -1, reverse=True)
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index

    by_model_mode = {(row["model"], row["mode"]): row for row in summary}
    paired: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary if row.get("ranking_eligible")}):
        minimum = by_model_mode.get((model, "min"))
        maximum = by_model_mode.get((model, "max"))
        if not minimum or not maximum or not minimum.get("ranking_eligible") or not maximum.get("ranking_eligible"):
            continue
        min_quality, max_quality = minimum.get("novel_primary_score"), maximum.get("novel_primary_score")
        min_speed, max_speed = minimum.get("tok_s_mean"), maximum.get("tok_s_mean")
        paired.append({
            "model": model,
            "parameter_size": maximum.get("parameter_size") or minimum.get("parameter_size"),
            "novel_min": min_quality,
            "novel_max": max_quality,
            "quality_delta": max_quality - min_quality if min_quality is not None and max_quality is not None else None,
            "tok_s_min": min_speed,
            "tok_s_max": max_speed,
            "speed_delta": max_speed - min_speed if min_speed is not None and max_speed is not None else None,
            "speed_delta_pct": ((max_speed / min_speed) - 1) * 100 if min_speed and max_speed is not None else None,
            "budget_rescue_min_n": minimum.get("budget_rescue_n", 0),
            "budget_rescue_max_n": maximum.get("budget_rescue_n", 0),
            "budget_rescue_min_rate": minimum.get("budget_rescue_rate"),
            "budget_rescue_max_rate": maximum.get("budget_rescue_rate"),
            "jcq_min": minimum.get("jcq_score"),
            "jcq_max": maximum.get("jcq_score"),
            "jcq_delta": maximum.get("jcq_score") - minimum.get("jcq_score") if minimum.get("jcq_score") is not None and maximum.get("jcq_score") is not None else None,
        })

    category_rows = [
        {
            "benchmark": benchmark,
            "model": model,
            "mode": mode,
            "category": category,
            "score": _mean(values),
            "n": len(values),
            "reference_only": reference_only,
            "ranking_eligible": ranking_eligible,
            "evaluator": evaluator,
        }
        for (benchmark, model, mode, category, reference_only, ranking_eligible, evaluator), values in sorted(category_accumulator.items())
    ]
    reference_summary = [row for row in summary if row.get("reference_only")]
    reference_rows = _reference_audit_rows(generations, primary_by_key, audit_by_key)
    budget_rescue_records_total = sum(int(row.get("budget_rescue_n", 0)) for row in summary)
    budget_rescue_attempts_total = sum(int(row.get("budget_rescue_attempts", 0)) for row in summary)
    budget_rescue_by_mode = {f"{row['model']}|{row['mode']}": {"records": row.get("budget_rescue_n", 0), "attempts": row.get("budget_rescue_attempts", 0), "rate": row.get("budget_rescue_rate")} for row in summary}
    data_dir = run_dir / "data"
    _write_csv(data_dir / "summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(data_dir / "ranking.csv", ranking, ["rank", *SUMMARY_FIELDS])
    _write_csv(data_dir / "reference_summary.csv", reference_summary, SUMMARY_FIELDS)
    _write_csv(data_dir / "reference_audit.csv", reference_rows, ["generation_key", "model", "mode", "benchmark", "item_id", "self_judged", "self_judged_status", "self_judged_overall", "self_judged_model", "independent_audit_status", "luna_audit_overall", "independent_audit_model"])
    _write_csv(data_dir / "paired.csv", paired, ["model", "parameter_size", "novel_min", "novel_max", "quality_delta", "tok_s_min", "tok_s_max", "speed_delta", "speed_delta_pct", "budget_rescue_min_n", "budget_rescue_max_n", "budget_rescue_min_rate", "budget_rescue_max_rate", "jcq_min", "jcq_max", "jcq_delta"])
    _write_csv(data_dir / "axis_scores.csv", axis_rows, ["benchmark", "model", "mode", "axis", "score", "n", "backend", "reference_only", "ranking_eligible", "evaluator"])
    _write_csv(data_dir / "category_scores.csv", category_rows, ["benchmark", "model", "mode", "category", "score", "n", "reference_only", "ranking_eligible", "evaluator"])
    _write_csv(data_dir / "failures.csv", failures, ["kind", "key", "model", "mode", "benchmark", "item_id", "error"])

    chart_paths = _charts(run_dir, ranking, paired, summary, axis_rows, category_rows, reference_summary)
    smoke_warning = manifest.get("smoke_warning") if manifest.get("profile") == "smoke" else None
    judge_errors = sum(1 for record in all_judgments if _legacy_scores_are_primary(manifest, record) and record.get("status") != "ok")
    audit_errors = sum(1 for record in all_audits if record.get("status") != "ok")
    judge_cost_total = _sum_known(record.get("cost_usd") for record in all_judgments if _legacy_scores_are_primary(manifest, record))
    cloud_cost_total = _sum_known(record.get("cost_usd") for record in generations if _is_reference(record) and record.get("status") == "ok")
    lines = [
        f"# Run `{manifest.get('run_id', run_dir.name)}`",
        "",
        f"- profile: `{manifest.get('profile', 'unknown')}`",
        f"- status: `{manifest.get('status', 'unknown')}`",
        f"- generation rows: {len(generations)} / planned: {generation_planned} / pending: {generation_pending} / errors: {sum(1 for record in generations if record.get('status') != 'ok')}",
        f"- budget rescue: {budget_rescue_records_total} records / {budget_rescue_attempts_total} additional budget requests (empty-content rescue)",
        f"- primary GLM Judge rows: {len(primary_by_key)} / pending: {judge_pending} / errors: {judge_errors}",
        f"- independent Luna audit rows: {len(audit_by_key)} / pending: {audit_pending} / errors: {audit_errors}",
        f"- cost separation: primary Judge={_fmt(judge_cost_total, 6)} USD / cloud target generation={_fmt(cloud_cost_total, 6)} USD",
        "",
    ]
    if smoke_warning:
        lines += [f"> **注意: {smoke_warning}。この結果はランキング用途不可です。**", ""]
    lines += [
        "## 主目的ランキング: local novel-ja",
        "",
        "ランキング対象は `ranking_eligible=true` のローカルOllama設定だけです。cloud reference GLMは自己評価バイアスを避けるため別集計・別図にし、localとの直接順位比較は断定しません。",
        "",
        "| rank | model | thinking | novel-ja | JCQ | objective-ja | tok/s | ok/total |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(f"| {row.get('rank')} | `{row['model']}` | `{row['mode']}` | {_fmt(row.get('novel_primary_score'))} | {_fmt(row.get('jcq_score'))} | {_fmt(row.get('objective_ja_score'), 3)} | {_fmt(row.get('tok_s_mean'))} | {row.get('generation_ok', 0)}/{row.get('generation_total', 0)} |")
    lines += ["", "## cloud reference: GLM-5.3-Flash（ランキング外）", "", "`z-ai/glm-5.3-flash` はtarget自身をGLM Judgeが採点するため、自己採点値は `self_judged=true` と明示し、主ランキングには混ぜません。creativeの独立参照値は `independent_audit.jsonl` の Luna(Max) です。機械採点objective-jaは通常どおり保存します。", "", "| mode | objective-ja | GLM self-judged | Luna audit | target cost USD | target latency |", "|---|---:|---:|---:|---:|---:|"]
    for row in reference_summary:
        lines.append(f"| `{row['mode']}` | {_fmt(row.get('objective_ja_score'), 3)} | {_fmt(row.get('self_judged_score'))} | {_fmt(row.get('luna_audit_score'))} | {_fmt(row.get('target_generation_cost_usd'), 6)} | {_fmt(row.get('target_latency_s_mean'))} |")
    lines += ["", "## モデル × thinking 設定", "", "`summary.csv` はlocalとcloudを同じファイルに保持します。`backend`、`reference_only`、`ranking_eligible`、requested/actual provider・modelを必ず確認してください。", "", "| backend | model | mode | reference | rank eligible | novel-ja | self | Luna audit | objective-ja | Judge cost | target cost |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| `{row['backend']}` | `{row['model']}` | `{row['mode']}` | {row['reference_only']} | {row['ranking_eligible']} | {_fmt(row.get('novel_primary_score'))} | {_fmt(row.get('self_judged_score'))} | {_fmt(row.get('luna_audit_score'))} | {_fmt(row.get('objective_ja_score'), 3)} | {_fmt(row.get('judge_cost_usd'), 6)} | {_fmt(row.get('target_generation_cost_usd'), 6)} |")
    lines += ["", "## thinking max の差分（localのみ）", "", "`paired.csv` はランキング対象localの同じモデルの min/max を対応付けます。cloud referenceのlow/maxは `reference_summary.csv` と `cloud_reference` 図で別表示します。"]
    if paired:
        lines += ["", "| model | novel min | novel max | quality Δ | tok/s min | tok/s max | speed Δ% | rescue min n | rescue max n |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in paired:
            lines.append(f"| `{row['model']}` | {_fmt(row.get('novel_min'))} | {_fmt(row.get('novel_max'))} | {_fmt(row.get('quality_delta'))} | {_fmt(row.get('tok_s_min'))} | {_fmt(row.get('tok_s_max'))} | {_fmt(row.get('speed_delta_pct'))} | {row.get('budget_rescue_min_n', 0)} | {row.get('budget_rescue_max_n', 0)} |")
    else:
        lines += ["", "対応する local min/max の組はありません。"]
    lines += ["", "## Judge評価内訳", "", "主観採点の標準経路は OpenRouter `z-ai/glm-5.3-flash` / `reasoning=max` です。OpenRouterのrequested model/providerと実際の `provider_actual` / `model_actual`、usage、reasoning tokens、cost、latencyはJSONLとログへ保存します。GPT-5.6 Solは使用しません。旧runのLuna記録はlegacy互換としてのみ表示されます。", "", "## 失敗・未完了", "", f"- generation planned: {generation_planned}", f"- generation pending: {generation_pending}", f"- generation errors: {sum(1 for record in generations if record.get('status') != 'ok')}", f"- primary Judge pending: {judge_pending}", f"- primary Judge errors: {judge_errors}", f"- Luna audit pending: {audit_pending}", f"- Luna audit errors: {audit_errors}", "- 詳細: [failures.csv](data/failures.csv)", "", "## 可視化", ""]
    for chart in chart_paths:
        if chart.endswith(".png"):
            lines += [f"![{chart}]({chart})", ""]
    lines += ["## 追跡可能なデータ", "", "- 生成全文・thinking・local/cloud request・usage/cost/latency: [generations.jsonl](data/generations.jsonl)", "- 標準GLM Judge JSONとusage/response参照: [judgments.jsonl](data/judgments.jsonl)", "- cloud creativeのLuna(Max)独立監査: [independent_audit.jsonl](data/independent_audit.jsonl)", "- 選択した設問: [items.jsonl](data/items.jsonl)", "- machine-readable集計: [summary.csv](data/summary.csv), [ranking.csv](data/ranking.csv), [reference_summary.csv](data/reference_summary.csv), [reference_audit.csv](data/reference_audit.csv), [paired.csv](data/paired.csv), [axis_scores.csv](data/axis_scores.csv), [category_scores.csv](data/category_scores.csv)", ""]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest["report"] = {
        "generated_at": utc_now(),
        "summary_rows": len(summary),
        "ranking_rows": len(ranking),
        "reference_rows": len(reference_summary),
        "generation_planned": generation_planned,
        "generation_pending": generation_pending,
        "generation_errors": sum(1 for record in generations if record.get("status") != "ok"),
        "budget_rescue_records": budget_rescue_records_total,
        "budget_rescue_attempts": budget_rescue_attempts_total,
        "budget_rescue_by_mode": budget_rescue_by_mode,
        "judge_pending": judge_pending,
        "judge_errors": judge_errors,
        "independent_audit_pending": audit_pending,
        "independent_audit_errors": audit_errors,
        "costs": {"primary_judge_usd": judge_cost_total, "cloud_reference_generation_usd": cloud_cost_total},
        "charts": chart_paths,
    }
    store.write_manifest(manifest)
    return {
        "rows": len(summary),
        "ranking_rows": len(ranking),
        "reference_rows": len(reference_summary),
        "generation_planned": manifest["report"]["generation_planned"],
        "generation_pending": manifest["report"]["generation_pending"],
        "generation_errors": manifest["report"]["generation_errors"],
        "budget_rescue_records": manifest["report"]["budget_rescue_records"],
        "budget_rescue_attempts": manifest["report"]["budget_rescue_attempts"],
        "judge_pending": manifest["report"]["judge_pending"],
        "judge_errors": manifest["report"]["judge_errors"],
        "independent_audit_pending": manifest["report"]["independent_audit_pending"],
        "independent_audit_errors": manifest["report"]["independent_audit_errors"],
        "costs": manifest["report"]["costs"],
        "charts": chart_paths,
    }
