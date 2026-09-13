"""Re-run the 2026-09-13 Judge calibration without persisting credentials.

Re-run with ``uv run python scripts/compare_judges.py`` after the source smoke
run exists and ``~/.config/credstore/openrouter.key`` is readable.  The key is
loaded only into memory by :class:`OpenRouterClient`; neither it nor request
headers are written to results or logs.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from novelbench.datasets import BenchmarkItem
from novelbench.judge import build_judge_prompt, parse_judge_json
from novelbench.openrouter import (
    OpenRouterClient,
    extract_cost_usd,
    extract_reasoning_tokens,
    safe_response_subset,
)
from novelbench.util import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "results/smoke-20260913c"
OUT = ROOT / "results/judge-comparison-20260913"
MODELS = [
    ("glm-5.3-flash", "z-ai/glm-5.3-flash"),
    ("deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def schema_for(criteria: tuple[str, ...]) -> dict[str, Any]:
    props = {axis: {"type": "integer", "minimum": 0, "maximum": 10} for axis in criteria}
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": props,
                "required": list(criteria),
                "additionalProperties": False,
            },
            "overall": {"type": "number", "minimum": 0, "maximum": 10},
            "brief_rationale": {"type": "string"},
        },
        "required": ["scores", "overall", "brief_rationale"],
        "additionalProperties": False,
    }


def iso_seconds(start: str, end: str) -> float:
    return (
        datetime.fromisoformat(end.replace("Z", "+00:00"))
        - datetime.fromisoformat(start.replace("Z", "+00:00"))
    ).total_seconds()


def call_openrouter(
    client: OpenRouterClient, label: str, model: str, generation: dict[str, Any]
) -> dict[str, Any]:
    item = BenchmarkItem.from_dict(generation["item"])
    criteria = tuple(item.rubric)
    prompt = build_judge_prompt(
        benchmark=item.benchmark, task=item.prompt, work=generation["content"], criteria=criteria
    )
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "reasoning": {"effort": "max", "exclude": True},
        "max_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judge_result", "strict": True, "schema": schema_for(criteria)},
        },
        "provider": {"sort": "price", "require_parameters": True},
        "usage": {"include": True},
    }
    last_error = ""
    for attempt in range(1, 3):
        started = time.perf_counter()
        try:
            raw = client.chat(body)
            latency = time.perf_counter() - started
            content = raw["choices"][0]["message"].get("content") or ""
            parsed = parse_judge_json(content, criteria=criteria)
            usage = raw.get("usage") or {}
            cost_usd = extract_cost_usd(usage)
            reasoning_tokens = extract_reasoning_tokens(usage)
            log_path = (
                OUT
                / "logs"
                / f"{label}-{generation['key'].replace(':', '_').replace('|', '_')}.json"
            )
            log_path.write_text(json.dumps(safe_response_subset(raw), ensure_ascii=False, indent=2) + "\n")
            return {
                "generation_key": generation["key"],
                "benchmark": item.benchmark,
                "category": item.category,
                "judge": label,
                "model": model,
                "reasoning": "max",
                "provider": raw.get("provider"),
                "model_actual": raw.get("model"),
                "latency_s": latency,
                "prompt_sha256": sha256_bytes(prompt.encode()),
                "status": "ok",
                "scores": parsed["scores"],
                "overall": parsed["overall"],
                "brief_rationale": parsed["brief_rationale"],
                "usage": usage,
                "cost_usd": cost_usd,
                "reasoning_tokens": reasoning_tokens,
                "attempts": attempt,
                "raw_log": str(log_path.relative_to(OUT)),
            }
        except (TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        except Exception as exc:
            # OpenRouterClient deliberately omits HTTP response bodies here;
            # this keeps a reflected credential out of the comparison JSONL.
            last_error = str(exc)[:1000]
    return {
        "generation_key": generation["key"],
        "benchmark": item.benchmark,
        "category": item.category,
        "judge": label,
        "model": model,
        "reasoning": "max",
        "status": "error",
        "error": last_error,
        "attempts": 2,
    }


def rankdata(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else None


def spearman(a: list[float], b: list[float]) -> float | None:
    return pearson(rankdata(a), rankdata(b))


def pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def normalize_luna(row: dict[str, Any]) -> dict[str, Any]:
    usage = ((row.get("io") or {}).get("primary") or {}).get("usage") or {}
    return {
        "generation_key": row["generation_key"],
        "benchmark": row["benchmark"],
        "category": row["category"],
        "judge": "gpt-5.6-luna",
        "model": "gpt-5.6-luna",
        "reasoning": "max",
        "provider": "openai-codex",
        "latency_s": iso_seconds(row["started_at"], row["finished_at"]),
        "status": row["status"],
        "scores": row["scores"],
        "overall": row["overall"],
        "brief_rationale": row["brief_rationale"],
        "usage": usage,
        "cost_usd": extract_cost_usd(usage),
        "reasoning_tokens": extract_reasoning_tokens(usage),
        "attempts": row["attempts"],
    }


def main() -> None:
    OUT.joinpath("data").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("charts").mkdir(parents=True, exist_ok=True)
    generations = [
        r
        for r in read_jsonl(SOURCE_RUN / "data/generations.jsonl")
        if r.get("status") == "ok" and r.get("benchmark") in {"jcq", "novel-ja"}
    ]
    luna = [
        normalize_luna(r)
        for r in read_jsonl(SOURCE_RUN / "data/judgments.jsonl")
        if r.get("status") == "ok"
    ]
    existing = {r["generation_key"] for r in luna}
    generations = [g for g in generations if g["key"] in existing]
    client = OpenRouterClient(timeout=180)
    rows = list(luna)
    for label, model in MODELS:
        for n, g in enumerate(generations, 1):
            print(f"[{label}] {n}/{len(generations)} {g['key']}", flush=True)
            r = call_openrouter(client, label, model, g)
            rows.append(r)
            print(
                f"  {r['status']} overall={r.get('overall')} latency={r.get('latency_s')}",
                flush=True,
            )
    with (OUT / "data/judgments.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r["status"] == "ok"]
    judges = ["gpt-5.6-luna", "glm-5.3-flash", "deepseek-v4.1-flash"]
    by = {(r["judge"], r["generation_key"]): r for r in ok}
    keys = [g["key"] for g in generations]
    metrics = []
    for j in judges:
        jr = [by[(j, k)] for k in keys if (j, k) in by]
        lat = [float(r["latency_s"]) for r in jr if r.get("latency_s") is not None]
        costs = [
            float(r["cost_usd"])
            if isinstance(r.get("cost_usd"), (int, float))
            else float((r.get("usage") or {}).get("cost") or (r.get("usage") or {}).get("estimated_cost_usd") or 0)
            for r in jr
        ]
        reason = [
            int(r["reasoning_tokens"])
            if isinstance(r.get("reasoning_tokens"), int)
            else int(((r.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or (r.get("usage") or {}).get("reasoning_tokens") or 0)
            for r in jr
        ]
        metrics.append(
            {
                "judge": j,
                "success": len(jr),
                "json_success_rate": len(jr) / len(keys),
                "mean_overall": statistics.mean([r["overall"] for r in jr]) if jr else None,
                "mean_latency_s": statistics.mean(lat) if lat else None,
                "p95_latency_s": pct(lat, 0.95),
                "total_cost_usd": sum(costs),
                "mean_reasoning_tokens": statistics.mean(reason) if reason else 0,
            }
        )
    correlations = []
    for a in judges:
        for b in judges:
            if a >= b:
                continue
            common = [k for k in keys if (a, k) in by and (b, k) in by]
            av = [float(by[(a, k)]["overall"]) for k in common]
            bv = [float(by[(b, k)]["overall"]) for k in common]
            correlations.append(
                {
                    "judge_a": a,
                    "judge_b": b,
                    "n": len(common),
                    "pearson": pearson(av, bv),
                    "spearman": spearman(av, bv),
                    "mae": statistics.mean(abs(x - y) for x, y in zip(av, bv, strict=True))
                    if common
                    else None,
                }
            )
    winner = []
    item_ids = sorted({k.split("|", 2)[2] for k in keys})
    for j in judges:
        for iid in item_ids:
            mn = next(
                (
                    by[(j, k)]["overall"]
                    for k in keys
                    if k.endswith("|" + iid) and "|min|" in k and (j, k) in by
                ),
                None,
            )
            mx = next(
                (
                    by[(j, k)]["overall"]
                    for k in keys
                    if k.endswith("|" + iid) and "|max|" in k and (j, k) in by
                ),
                None,
            )
            outcome = (
                None
                if mn is None or mx is None
                else ("max" if mx > mn else "min" if mn > mx else "tie")
            )
            winner.append({"judge": j, "item_id": iid, "min": mn, "max": mx, "winner": outcome})
    (OUT / "data/summary.json").write_text(
        json.dumps(
            {
                "source_run": "smoke-20260913c",
                "reasoning": "max",
                "metrics": metrics,
                "correlations": correlations,
                "min_max_winners": winner,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    with (OUT / "data/judge_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=metrics[0])
        w.writeheader()
        w.writerows(metrics)
    with (OUT / "data/correlations.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=correlations[0])
        w.writeheader()
        w.writerows(correlations)
    make_charts(ok, judges, keys, metrics)
    write_readme(metrics, correlations, winner, rows)


def make_charts(
    rows: list[dict[str, Any]], judges: list[str], keys: list[str], metrics: list[dict[str, Any]]
) -> None:
    import matplotlib.pyplot as plt

    short = {
        "gpt-5.6-luna": "Luna(Max)",
        "glm-5.3-flash": "GLM-5.3-Flash",
        "deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    }
    by = {(r["judge"], r["generation_key"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(keys))
    width = 0.25
    for i, j in enumerate(judges):
        vals = [by[(j, k)]["overall"] if (j, k) in by else float("nan") for k in keys]
        ax.bar([v + (i - 1) * width for v in x], vals, width, label=short[j])
    ax.set_ylim(0, 10)
    ax.set_ylabel("Overall score (0-10)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [k.split("|", 2)[2] + ("/MAX" if "|max|" in k else "/MIN") for k in keys],
        rotation=45,
        ha="right",
    )
    ax.legend()
    ax.set_title("Judge score comparison on the same 8 works")
    fig.tight_layout()
    fig.savefig(OUT / "charts/score_comparison.png", dpi=160)
    fig.savefig(OUT / "charts/score_comparison.svg")
    plt.close(fig)
    api = [m for m in metrics if m["judge"] != "gpt-5.6-luna"]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([short[m["judge"]] for m in api], [m["total_cost_usd"] for m in api])
    ax.set_ylabel("Actual OpenRouter cost (USD)")
    ax.set_title("Cost for judging 8 works at reasoning=max")
    fig.tight_layout()
    fig.savefig(OUT / "charts/cost.png", dpi=160)
    fig.savefig(OUT / "charts/cost.svg")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([short[m["judge"]] for m in metrics], [m["mean_latency_s"] for m in metrics])
    ax.set_ylabel("Mean latency per judgment (s)")
    ax.set_title("Judge latency")
    fig.tight_layout()
    fig.savefig(OUT / "charts/latency.png", dpi=160)
    fig.savefig(OUT / "charts/latency.svg")
    plt.close(fig)


def write_readme(
    metrics: list[dict[str, Any]],
    correlations: list[dict[str, Any]],
    winners: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    m = {x["judge"]: x for x in metrics}

    def fmt(v: Any, n: int = 3) -> str:
        return "—" if v is None else f"{v:.{n}f}"

    lines = [
        "# Judge comparison — 2026-09-13",
        "",
        "`smoke-20260913c` の同じ8作品を、同一rubric・blind条件で再採点した小規模な校正試験です。生成モデルは再実行していません。OpenRouter側2モデルは `reasoning=max` と strict JSON Schemaを使用し、対応パラメータを満たすendpointのうちprice routingを使用しました。Lunaも既存の `reasoning=max` 採点を再利用しています。",
        "",
        "## 実測サマリー",
        "",
        "| Judge | JSON成功 | 平均overall | 平均latency | reasoning tokens/件 | 実コスト(8件) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for j in ["gpt-5.6-luna", "glm-5.3-flash", "deepseek-v4.1-flash"]:
        x = m[j]
        cost = "Included" if j == "gpt-5.6-luna" else f"${x['total_cost_usd']:.6f}"
        lines.append(
            f"| {j} | {x['success']}/8 | {x['mean_overall']:.3f} | {x['mean_latency_s']:.2f}s | {x['mean_reasoning_tokens']:.1f} | {cost} |"
        )
    lines += [
        "",
        "## Luna(Max)との一致度",
        "",
        "| Judge | Pearson | Spearman | MAE |",
        "|---|---:|---:|---:|",
    ]
    for j in ["glm-5.3-flash", "deepseek-v4.1-flash"]:
        x = next(x for x in correlations if {x["judge_a"], x["judge_b"]} == {"gpt-5.6-luna", j})
        lines.append(f"| {j} | {fmt(x['pearson'])} | {fmt(x['spearman'])} | {fmt(x['mae'])} |")
    lines += [
        "",
        "![Score comparison](charts/score_comparison.png)",
        "",
        "![Cost](charts/cost.png)",
        "",
        "![Latency](charts/latency.png)",
        "",
        "## 注意",
        "",
        "8作品だけのsmoke校正なので、文学的Judgeの最終結論には小さすぎます。ここではJSON遵守、相対順位、速度、実請求コストを確認します。quick/fullのJudge選定では、相関だけでなくMIN/MAX勝敗の一致と再現性も重視します。",
        "",
        "生データ: `data/judgments.jsonl` / `data/summary.json` / `data/judge_metrics.csv` / `data/correlations.csv`。OpenRouterのraw responseは `logs/` に保存しています。",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
