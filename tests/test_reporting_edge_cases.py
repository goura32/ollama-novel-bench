from pathlib import Path

from novelbench.reporting import build_report
from novelbench.store import RunStore


def test_ranking_uses_scored_min_when_max_judgment_is_missing(tmp_path: Path):
    run_dir = tmp_path / "results" / "r2"
    store = RunStore(run_dir)
    store.write_manifest(
        {
            "run_id": "r2",
            "profile": "quick",
            "status": "complete_with_errors",
            "item_count": 1,
            "item_ids": ["novel-ja-01"],
            "models": [
                {
                    "name": "demo",
                    "parameter_size": "1B",
                    "thinking": {
                        "kind": "binary",
                        "modes": [
                            {"label": "min", "field": "think", "value": False},
                            {"label": "max", "field": "think", "value": True},
                        ],
                    },
                }
            ],
        }
    )
    store.append(
        "generations.jsonl",
        {
            "key": "demo|min|novel-ja-01",
            "model": "demo",
            "mode": "min",
            "benchmark": "novel-ja",
            "item_id": "novel-ja-01",
            "status": "ok",
            "content": "文章",
            "content_length": 2,
            "thinking_length": 0,
            "tok_s": 10.0,
        },
    )
    store.append(
        "generations.jsonl",
        {
            "key": "demo|max|novel-ja-01",
            "model": "demo",
            "mode": "max",
            "benchmark": "novel-ja",
            "item_id": "novel-ja-01",
            "status": "ok",
            "content": "文章",
            "content_length": 2,
            "thinking_length": 5,
            "tok_s": 5.0,
        },
    )
    store.append(
        "judgments.jsonl",
        {
            "key": "demo|min|novel-ja-01",
            "model": "demo",
            "mode": "min",
            "benchmark": "novel-ja",
            "category": "会話・サブテキスト",
            "item_id": "novel-ja-01",
            "status": "ok",
            "scores": {"prose_quality": 6},
            "overall": 6.0,
        },
    )

    result = build_report(run_dir)

    assert result["generation_errors"] == 0
    assert result["judge_errors"] == 0
    assert result["generation_pending"] == 0
    assert result["judge_pending"] == 1
    ranking = (run_dir / "data" / "ranking.csv").read_text(encoding="utf-8")
    assert "6.0" in ranking
    assert "min" in ranking
