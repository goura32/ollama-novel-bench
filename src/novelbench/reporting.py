"""Machine-readable aggregation and matplotlib charts for a run."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any

from .judge import NOVEL_CRITERIA
from .store import RunStore
from .util import utc_now

SUMMARY_FIELDS = [
    "model",
    "parameter_size",
    "mode",
    "thinking_kind",
    "generation_total",
    "generation_ok",
    "generation_failed",
    "failure_rate",
    "budget_rescue_n",
    "budget_rescue_attempts",
    "budget_rescue_rate",
    "output_chars_mean",
    "thinking_chars_mean",
    "tok_s_mean",
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
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return mean(numeric) if numeric else None


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

    # A. Novel primary ranking.  Text labels and values make the chart usable
    # without color perception.
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
        axis.set_title("A. novel-ja 主目的ランキング（最大思考設定を優先）")
        for bar, value in zip(bars, values, strict=True):
            axis.text(
                min(value + 0.08, 9.7),
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
            )
        fig.tight_layout()
        fig.savefig(chart_dir / "novel_ranking.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "novel_ranking.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "novel_ranking", "A. novel-ja 主目的ランキング")
    generated += ["charts/novel_ranking.png", "charts/novel_ranking.svg"]

    # B. Paired min/max quality and speed changes.
    if paired:
        rows = paired
        x = np.arange(len(rows))
        width = 0.36
        fig, (quality, speed) = plt.subplots(
            2, 1, figsize=(11, max(6, len(rows) * 0.8)), sharex=True
        )
        min_quality = [
            row.get("novel_min") if row.get("novel_min") is not None else np.nan for row in rows
        ]
        max_quality = [
            row.get("novel_max") if row.get("novel_max") is not None else np.nan for row in rows
        ]
        quality.bar(
            x - width / 2, min_quality, width, label="min", color="#9ecae1", edgecolor="black"
        )
        quality.bar(
            x + width / 2, max_quality, width, label="max", color="#2171b5", edgecolor="black"
        )
        quality.set_ylim(0, 10)
        quality.set_ylabel("novel-ja (0–10)")
        quality.set_title("B. thinking 最小 vs 最大（paired）")
        quality.legend()
        for index, row in enumerate(rows):
            if row.get("quality_delta") is not None:
                quality.text(
                    index,
                    min(float(row.get("novel_min") or 0), 9.2),
                    f"Δ{float(row['quality_delta']):+.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        min_speed = [
            row.get("tok_s_min") if row.get("tok_s_min") is not None else np.nan for row in rows
        ]
        max_speed = [
            row.get("tok_s_max") if row.get("tok_s_max") is not None else np.nan for row in rows
        ]
        speed.bar(x - width / 2, min_speed, width, label="min", color="#fdd0a2", edgecolor="black")
        speed.bar(x + width / 2, max_speed, width, label="max", color="#e6550d", edgecolor="black")
        speed.set_ylabel("生成速度 (tok/s)")
        speed.set_xticks(
            x, [_short_model(str(row["model"])) for row in rows], rotation=25, ha="right"
        )
        speed.set_title("生成速度（高いほど速い；数値は paired.csv）")
        speed.legend()
        fig.tight_layout()
        fig.savefig(chart_dir / "thinking_paired.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "thinking_paired.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "thinking_paired", "B. thinking 最小 vs 最大")
    generated += ["charts/thinking_paired.png", "charts/thinking_paired.svg"]

    # C. Quality vs speed scatter.
    scatter_rows = [
        row
        for row in summary
        if row.get("novel_primary_score") is not None and row.get("tok_s_mean") is not None
    ]
    if scatter_rows:
        fig, axis = plt.subplots(figsize=(10, 6))
        for _index, row in enumerate(scatter_rows):
            axis.scatter(
                float(row["tok_s_mean"]),
                float(row["novel_primary_score"]),
                s=85,
                edgecolor="black",
                label=f"{row['model']} [{row['mode']}]",
            )
            axis.annotate(
                f"{_short_model(str(row['model']), 14)}\n{row['mode']}",
                (float(row["tok_s_mean"]), float(row["novel_primary_score"])),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("生成速度 (tok/s)")
        axis.set_ylabel("novel-ja 平均 (0–10)")
        axis.set_title("C. 品質スコア vs 生成速度")
        if len(scatter_rows) <= 12:
            axis.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(chart_dir / "quality_speed_scatter.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "quality_speed_scatter.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "quality_speed_scatter", "C. 品質スコア vs 生成速度")
    generated += ["charts/quality_speed_scatter.png", "charts/quality_speed_scatter.svg"]

    # D. Axes for the top models. Grouped bars are more readable than a dense
    # radar chart and preserve each axis instead of hiding it in one score.
    top = ranking_valid[:5]
    top_keys = {(row["model"], row["mode"]) for row in top}
    axis_names = list(NOVEL_CRITERIA)
    grouped = [
        row
        for row in axes
        if row["benchmark"] == "novel-ja" and (row["model"], row["mode"]) in top_keys
    ]
    if grouped:
        labels = [f"{row['model']} [{row['mode']}]" for row in top]
        values_by_key = {
            (row["model"], row["mode"], row["axis"]): row.get("score") for row in grouped
        }
        x = np.arange(len(labels))
        width = 0.8 / len(axis_names)
        fig, axis = plt.subplots(figsize=(max(11, len(labels) * 2.1), 6))
        for offset, axis_name in enumerate(axis_names):
            values = [
                values_by_key.get((row["model"], row["mode"], axis_name), np.nan) for row in top
            ]
            axis.bar(
                x - 0.4 + width / 2 + offset * width,
                values,
                width,
                label=axis_name,
                edgecolor="black",
            )
        axis.set_xticks(x, labels, rotation=20, ha="right")
        axis.set_ylim(0, 10)
        axis.set_ylabel("スコア (0–10)")
        axis.set_title("D. novel-ja 評価軸（上位5設定）")
        axis.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(chart_dir / "novel_axes.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "novel_axes.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "novel_axes", "D. novel-ja 評価軸")
    generated += ["charts/novel_axes.png", "charts/novel_axes.svg"]

    # E. Category comparison. Objective accuracy is shown on the same 0..10
    # visual scale only; raw values remain in category_scores.csv.
    category_rows = [row for row in categories if row.get("score") is not None]
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
                    match = next(
                        (
                            row
                            for row in subset
                            if (row["model"], row["mode"]) == key and row["category"] == category
                        ),
                        None,
                    )
                    value = match["score"] if match else np.nan
                    values.append(
                        float(value) * 10 if benchmark == "objective-ja" else float(value)
                    )
                plot_axis.bar(
                    x - 0.4 + width / 2 + index * width,
                    values,
                    width,
                    label=f"{key[0]} [{key[1]}]",
                    edgecolor="black",
                )
            plot_axis.set_xticks(
                x,
                [_short_model(category, 14) for category in categories_unique],
                rotation=35,
                ha="right",
            )
            plot_axis.set_ylim(0, 10)
            plot_axis.set_title(benchmark)
            plot_axis.set_ylabel("可視化スケール (0–10)")
            if len(models_unique) <= 5:
                plot_axis.legend(fontsize=7)
        fig.suptitle("E. objective-ja / JCQ / novel-ja のカテゴリ別比較（objectiveは正解率×10）")
        fig.tight_layout()
        fig.savefig(chart_dir / "category_comparison.png", dpi=150, bbox_inches="tight")
        fig.savefig(chart_dir / "category_comparison.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        _plot_empty(chart_dir / "category_comparison", "E. カテゴリ別比較")
    generated += ["charts/category_comparison.png", "charts/category_comparison.svg"]
    return generated


def build_report(run_dir: Path) -> dict[str, Any]:
    store = RunStore(run_dir)
    manifest = store.load_manifest()
    generations = list(store.read_jsonl("generations.jsonl"))
    judgments_by_key: dict[str, dict[str, Any]] = {}
    for record in store.read_jsonl("judgments.jsonl"):
        key = str(record.get("key") or record.get("generation_key") or "")
        if key:
            judgments_by_key[key] = record

    model_info = {str(model.get("name")): model for model in manifest.get("models", [])}
    mode_keys = {
        (str(record.get("model")), str(record.get("mode")))
        for record in generations
        if record.get("model")
    }
    if not mode_keys:
        for model in manifest.get("models", []):
            for mode in (model.get("thinking") or {}).get("modes", []):
                mode_keys.add((str(model.get("name")), str(mode.get("label"))))

    item_ids = [str(item_id) for item_id in manifest.get("item_ids", [])]
    item_count = int(
        manifest.get("item_count")
        or len(item_ids)
        or len({str(record.get("item_id")) for record in generations if record.get("item_id")})
    )
    generation_planned = len(mode_keys) * item_count
    generation_keys = {str(record["key"]) for record in generations if record.get("key")}
    generation_pending = max(generation_planned - len(generation_keys), 0)
    creative_generation_keys = {
        str(record["key"])
        for record in generations
        if record.get("key")
        and record.get("status") == "ok"
        and record.get("benchmark") in {"jcq", "novel-ja", "eqcw"}
    }
    judgment_keys = {
        str(record.get("key") or record.get("generation_key"))
        for record in store.read_jsonl("judgments.jsonl")
        if record.get("key") or record.get("generation_key")
    }
    judge_pending = len(creative_generation_keys - judgment_keys)

    summary: list[dict[str, Any]] = []
    axis_rows: list[dict[str, Any]] = []
    category_accumulator: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for model, mode in sorted(mode_keys):
        rows = [
            record
            for record in generations
            if str(record.get("model")) == model and str(record.get("mode")) == mode
        ]
        successful = [record for record in rows if record.get("status") == "ok"]
        objective = [
            record
            for record in successful
            if record.get("benchmark") == "objective-ja"
            and isinstance(record.get("objective_score"), (int, float))
        ]
        jcq_judgments = [
            judgments_by_key.get(str(record.get("key")))
            for record in successful
            if record.get("benchmark") == "jcq"
        ]
        jcq_judgments = [
            record for record in jcq_judgments if record and record.get("status") == "ok"
        ]
        novel_judgments = [
            judgments_by_key.get(str(record.get("key")))
            for record in successful
            if record.get("benchmark") == "novel-ja"
        ]
        novel_judgments = [
            record for record in novel_judgments if record and record.get("status") == "ok"
        ]
        eqcw_judgments = [
            judgments_by_key.get(str(record.get("key")))
            for record in successful
            if record.get("benchmark") == "eqcw"
        ]
        eqcw_judgments = [
            record for record in eqcw_judgments if record and record.get("status") == "ok"
        ]
        for judgment in [*jcq_judgments, *novel_judgments, *eqcw_judgments]:
            for axis, score in (judgment.get("scores") or {}).items():
                if isinstance(score, (int, float)):
                    axis_rows.append(
                        {
                            "benchmark": judgment.get("benchmark"),
                            "model": model,
                            "mode": mode,
                            "axis": axis,
                            "score": float(score),
                            "n": 1,
                        }
                    )
                    category_accumulator[
                        (
                            str(judgment.get("benchmark")),
                            model,
                            mode,
                            str(judgment.get("category", "all")),
                        )
                    ].append(float(score))
        for record in objective:
            category_accumulator[
                ("objective-ja", model, mode, str(record.get("category", "all")))
            ].append(float(record["objective_score"]))
        info = model_info.get(model, {})
        budget_rescue_records = [
            record
            for record in rows
            if isinstance(record.get("empty_content_rescue"), dict)
            and record["empty_content_rescue"].get("used")
        ]
        budget_rescue_n = len(budget_rescue_records)
        budget_rescue_attempts = sum(
            _rescue_attempt_count(record) for record in budget_rescue_records
        )
        summary.append(
            {
                "model": model,
                "parameter_size": info.get("parameter_size"),
                "mode": mode,
                "thinking_kind": (info.get("thinking") or {}).get("kind"),
                "generation_total": len(rows),
                "generation_ok": len(successful),
                "generation_failed": len(rows) - len(successful),
                "failure_rate": (len(rows) - len(successful)) / len(rows) if rows else None,
                "budget_rescue_n": budget_rescue_n,
                "budget_rescue_attempts": budget_rescue_attempts,
                "budget_rescue_rate": budget_rescue_n / len(rows) if rows else None,
                "output_chars_mean": _mean(
                    record.get("content_length", len(str(record.get("content", ""))))
                    for record in successful
                ),
                "thinking_chars_mean": _mean(
                    record.get("thinking_length", len(str(record.get("thinking", ""))))
                    for record in successful
                ),
                "tok_s_mean": _mean(record.get("tok_s") for record in successful),
                "objective_ja_score": _mean(record.get("objective_score") for record in objective),
                "objective_ja_n": len(objective),
                "jcq_score": _mean(record.get("overall") for record in jcq_judgments),
                "jcq_n": len(jcq_judgments),
                "novel_primary_score": _mean(record.get("overall") for record in novel_judgments),
                "novel_n": len(novel_judgments),
                "eqcw_score": _mean(record.get("overall") for record in eqcw_judgments),
                "eqcw_n": len(eqcw_judgments),
            }
        )
        for record in rows:
            if record.get("status") != "ok":
                failures.append(
                    {
                        "kind": "generation",
                        "key": record.get("key"),
                        "model": model,
                        "mode": mode,
                        "benchmark": record.get("benchmark"),
                        "item_id": record.get("item_id"),
                        "error": record.get("error"),
                    }
                )
            judgment = judgments_by_key.get(str(record.get("key")))
            if judgment and judgment.get("status") != "ok":
                failures.append(
                    {
                        "kind": "judge",
                        "key": record.get("key"),
                        "model": model,
                        "mode": mode,
                        "benchmark": record.get("benchmark"),
                        "item_id": record.get("item_id"),
                        "error": judgment.get("error"),
                    }
                )

    ranking: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary}):
        candidates = [row for row in summary if row["model"] == model]
        preferred = sorted(
            candidates,
            key=lambda row: (
                row.get("novel_primary_score") is not None,
                row["mode"] == "max",
                row.get("novel_primary_score") or -1,
            ),
            reverse=True,
        )[0]
        ranking.append(dict(preferred))
    ranking.sort(
        key=lambda row: (
            row.get("novel_primary_score") if row.get("novel_primary_score") is not None else -1
        ),
        reverse=True,
    )
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index

    by_model_mode = {(row["model"], row["mode"]): row for row in summary}
    paired: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in summary}):
        minimum = by_model_mode.get((model, "min"))
        maximum = by_model_mode.get((model, "max"))
        if not minimum or not maximum:
            continue
        min_quality = minimum.get("novel_primary_score")
        max_quality = maximum.get("novel_primary_score")
        min_speed = minimum.get("tok_s_mean")
        max_speed = maximum.get("tok_s_mean")
        paired.append(
            {
                "model": model,
                "parameter_size": maximum.get("parameter_size") or minimum.get("parameter_size"),
                "novel_min": min_quality,
                "novel_max": max_quality,
                "quality_delta": max_quality - min_quality
                if min_quality is not None and max_quality is not None
                else None,
                "tok_s_min": min_speed,
                "tok_s_max": max_speed,
                "speed_delta": max_speed - min_speed
                if min_speed is not None and max_speed is not None
                else None,
                "speed_delta_pct": ((max_speed / min_speed) - 1) * 100
                if min_speed and max_speed is not None
                else None,
                "budget_rescue_min_n": minimum.get("budget_rescue_n", 0),
                "budget_rescue_max_n": maximum.get("budget_rescue_n", 0),
                "budget_rescue_min_rate": minimum.get("budget_rescue_rate"),
                "budget_rescue_max_rate": maximum.get("budget_rescue_rate"),
                "jcq_min": minimum.get("jcq_score"),
                "jcq_max": maximum.get("jcq_score"),
                "jcq_delta": maximum.get("jcq_score") - minimum.get("jcq_score")
                if minimum.get("jcq_score") is not None and maximum.get("jcq_score") is not None
                else None,
            }
        )

    category_rows = [
        {
            "benchmark": benchmark,
            "model": model,
            "mode": mode,
            "category": category,
            "score": _mean(values),
            "n": len(values),
        }
        for (benchmark, model, mode, category), values in sorted(category_accumulator.items())
    ]
    budget_rescue_records_total = sum(int(row.get("budget_rescue_n", 0)) for row in summary)
    budget_rescue_attempts_total = sum(
        int(row.get("budget_rescue_attempts", 0)) for row in summary
    )
    budget_rescue_by_mode = {
        f"{row['model']}|{row['mode']}": {
            "records": row.get("budget_rescue_n", 0),
            "attempts": row.get("budget_rescue_attempts", 0),
            "rate": row.get("budget_rescue_rate"),
        }
        for row in summary
    }
    data_dir = run_dir / "data"
    _write_csv(data_dir / "summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(data_dir / "ranking.csv", ranking, ["rank", *SUMMARY_FIELDS])
    _write_csv(
        data_dir / "paired.csv",
        paired,
        [
            "model",
            "parameter_size",
            "novel_min",
            "novel_max",
            "quality_delta",
            "tok_s_min",
            "tok_s_max",
            "speed_delta",
            "speed_delta_pct",
            "budget_rescue_min_n",
            "budget_rescue_max_n",
            "budget_rescue_min_rate",
            "budget_rescue_max_rate",
            "jcq_min",
            "jcq_max",
            "jcq_delta",
        ],
    )
    _write_csv(
        data_dir / "axis_scores.csv",
        axis_rows,
        ["benchmark", "model", "mode", "axis", "score", "n"],
    )
    _write_csv(
        data_dir / "category_scores.csv",
        category_rows,
        ["benchmark", "model", "mode", "category", "score", "n"],
    )
    _write_csv(
        data_dir / "failures.csv",
        failures,
        ["kind", "key", "model", "mode", "benchmark", "item_id", "error"],
    )

    chart_paths = _charts(run_dir, ranking, paired, summary, axis_rows, category_rows)
    smoke_warning = manifest.get("smoke_warning") if manifest.get("profile") == "smoke" else None
    lines = [
        f"# Run `{manifest.get('run_id', run_dir.name)}`",
        "",
        f"- profile: `{manifest.get('profile', 'unknown')}`",
        f"- status: `{manifest.get('status', 'unknown')}`",
        f"- generation rows: {len(generations)} / planned: {generation_planned} / pending: {generation_pending} / errors: {sum(1 for record in generations if record.get('status') != 'ok')}",
        f"- budget rescue: {budget_rescue_records_total} records / {budget_rescue_attempts_total} additional budget requests (empty content only)",
        f"- judge rows: {len(judgments_by_key)} / pending: {judge_pending} / errors: {sum(1 for record in judgments_by_key.values() if record.get('status') != 'ok')}",
        "",
    ]
    if smoke_warning:
        lines += [f"> **注意: {smoke_warning}。この結果はランキング用途不可です。**", ""]
    lines += [
        "## 主目的ランキング: novel-ja",
        "",
        "モデルごとに `max` 設定を優先（非thinkingモデルは `none`）。主目的を一つの総合点へ潰さず、objective / creativity / novel / performance を別列で示します。",
        "",
        "| rank | model | thinking | novel-ja | JCQ | objective-ja | tok/s | ok/total |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        lines.append(
            f"| {row.get('rank')} | `{row['model']}` | `{row['mode']}` | {_fmt(row.get('novel_primary_score'))} | {_fmt(row.get('jcq_score'))} | {_fmt(row.get('objective_ja_score'), 3)} | {_fmt(row.get('tok_s_mean'))} | {row.get('generation_ok', 0)}/{row.get('generation_total', 0)} |"
        )
    lines += [
        "",
        "## モデル × thinking 設定",
        "",
        "| model | size | mode | novel-ja | JCQ | objective-ja | tok/s | output chars | thinking chars | failure rate | budget rescue n/rate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['model']}` | {row.get('parameter_size') or '—'} | `{row['mode']}` | {_fmt(row.get('novel_primary_score'))} | {_fmt(row.get('jcq_score'))} | {_fmt(row.get('objective_ja_score'), 3)} | {_fmt(row.get('tok_s_mean'))} | {_fmt(row.get('output_chars_mean'))} | {_fmt(row.get('thinking_chars_mean'))} | {_fmt(row.get('failure_rate'), 3)} | {row.get('budget_rescue_n', 0)}/{_fmt(row.get('budget_rescue_rate'), 3)} |"
        )
    lines += [
        "",
        "## thinking max の差分",
        "",
        "`paired.csv` は同じモデルの min/max を対応付けます。品質差は `max - min`、速度差は tok/s の差と低下率です。",
        "",
    ]
    if paired:
        lines += [
            "| model | novel min | novel max | quality Δ | tok/s min | tok/s max | speed Δ% | rescue min n | rescue max n |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in paired:
            lines.append(
                f"| `{row['model']}` | {_fmt(row.get('novel_min'))} | {_fmt(row.get('novel_max'))} | {_fmt(row.get('quality_delta'))} | {_fmt(row.get('tok_s_min'))} | {_fmt(row.get('tok_s_max'))} | {_fmt(row.get('speed_delta_pct'))} | {row.get('budget_rescue_min_n', 0)} | {row.get('budget_rescue_max_n', 0)} |"
            )
    else:
        lines.append("対応する min/max の組はありません。")
    lines += [
        "",
        "## Judge評価内訳",
        "",
        "`axis_scores.csv` に全軸を保存しています。novel-ja は instruction_following / japanese_naturalness / prose_quality / show_dont_tell / character_emotion / coherence / originality / ending_resonance、JCQ は fluency / flexibility / originality / elaboration です。",
        "",
        "## 失敗・未完了",
        "",
        f"- generation planned: {generation_planned}",
        f"- generation pending: {generation_pending}",
        f"- generation errors: {sum(1 for record in generations if record.get('status') != 'ok')}",
        f"- judge pending: {judge_pending}",
        f"- judge errors: {sum(1 for record in judgments_by_key.values() if record.get('status') != 'ok')}",
        "- 詳細: [failures.csv](data/failures.csv)",
        "",
        "## 可視化",
        "",
    ]
    for chart in chart_paths:
        if chart.endswith(".png"):
            lines.append(f"![{chart}]({chart})")
            lines.append("")
    lines += [
        "## 追跡可能なデータ",
        "",
        "- 生成全文・thinking・Ollamaメトリクス: [generations.jsonl](data/generations.jsonl)",
        "- Judge JSONとusage/stdio参照: [judgments.jsonl](data/judgments.jsonl)",
        "- 選択した設問: [items.jsonl](data/items.jsonl)",
        "- machine-readable集計: [summary.csv](data/summary.csv), [ranking.csv](data/ranking.csv), [paired.csv](data/paired.csv), [axis_scores.csv](data/axis_scores.csv), [category_scores.csv](data/category_scores.csv)",
        "",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest["report"] = {
        "generated_at": utc_now(),
        "summary_rows": len(summary),
        "ranking_rows": len(ranking),
        "generation_planned": generation_planned,
        "generation_pending": generation_pending,
        "generation_errors": sum(1 for record in generations if record.get("status") != "ok"),
        "budget_rescue_records": budget_rescue_records_total,
        "budget_rescue_attempts": budget_rescue_attempts_total,
        "budget_rescue_by_mode": budget_rescue_by_mode,
        "judge_pending": judge_pending,
        "judge_errors": sum(
            1 for record in judgments_by_key.values() if record.get("status") != "ok"
        ),
        "charts": chart_paths,
    }
    store.write_manifest(manifest)
    return {
        "rows": len(summary),
        "ranking_rows": len(ranking),
        "generation_planned": manifest["report"]["generation_planned"],
        "generation_pending": manifest["report"]["generation_pending"],
        "generation_errors": manifest["report"]["generation_errors"],
        "budget_rescue_records": manifest["report"]["budget_rescue_records"],
        "budget_rescue_attempts": manifest["report"]["budget_rescue_attempts"],
        "judge_pending": manifest["report"]["judge_pending"],
        "judge_errors": manifest["report"]["judge_errors"],
        "charts": chart_paths,
    }
