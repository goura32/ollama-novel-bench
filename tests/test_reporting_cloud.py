import csv
import json
from pathlib import Path

from novelbench.cloud import CLOUD_REFERENCE_MODEL
from novelbench.judge import JUDGE_MODEL
from novelbench.reporting import build_report
from novelbench.store import RunStore


def _judgment(key: str, *, model: str, self_judged: bool, score: float):
    return {
        "key": key,
        "generation_key": key,
        "model": model,
        "mode": "max",
        "benchmark": "novel-ja",
        "category": "test",
        "item_id": "novel-ja-01",
        "status": "ok",
        "evaluator_role": "primary",
        "self_judged": self_judged,
        "reference_only": self_judged,
        "ranking_eligible": not self_judged,
        "judge": {"backend": "openrouter", "provider": "openrouter", "model": JUDGE_MODEL, "reasoning": "max"},
        "scores": {"prose_quality": score},
        "overall": score,
        "cost_usd": 0.004 if self_judged else 0.002,
    }


def test_mixed_local_and_cloud_report_separates_ranking_and_audit(tmp_path: Path):
    run_dir = tmp_path / "run"
    store = RunStore(run_dir)
    store.write_manifest(
        {
            "run_id": "mixed",
            "profile": "quick",
            "status": "complete",
            "item_count": 1,
            "item_ids": ["novel-ja-01"],
            "models": [
                {
                    "name": "local-model",
                    "parameter_size": "1B",
                    "thinking": {"modes": [{"label": "max"}]},
                }
            ],
            "judge": {"backend": "openrouter", "provider": "openrouter", "model": JUDGE_MODEL},
            "cloud_reference": {
                "enabled": True,
                "model": CLOUD_REFERENCE_MODEL,
                "modes": [{"label": "min", "effort": "low"}, {"label": "max", "effort": "max"}],
            },
        }
    )
    local_key = "local-model|max|novel-ja-01"
    cloud_key = f"cloud:{CLOUD_REFERENCE_MODEL}|max|novel-ja-01"
    store.append(
        "generations.jsonl",
        {
            "key": local_key,
            "model": "local-model",
            "mode": "max",
            "benchmark": "novel-ja",
            "category": "test",
            "item_id": "novel-ja-01",
            "status": "ok",
            "backend": "ollama",
            "provider": "ollama",
            "reference_only": False,
            "ranking_eligible": True,
            "content": "local",
            "content_length": 5,
            "thinking_length": 2,
            "tok_s": 10.0,
        },
    )
    store.append(
        "generations.jsonl",
        {
            "key": cloud_key,
            "model": CLOUD_REFERENCE_MODEL,
            "mode": "max",
            "benchmark": "novel-ja",
            "category": "test",
            "item_id": "novel-ja-01",
            "status": "ok",
            "backend": "openrouter",
            "provider": "openrouter",
            "provider_actual": "Together",
            "reference_only": True,
            "ranking_eligible": False,
            "content": "cloud",
            "content_length": 5,
            "thinking_length": 3,
            "latency_s": 1.5,
            "cost_usd": 0.01,
        },
    )
    store.append("judgments.jsonl", _judgment(local_key, model="local-model", self_judged=False, score=6.0))
    store.append("judgments.jsonl", _judgment(cloud_key, model=CLOUD_REFERENCE_MODEL, self_judged=True, score=8.0))
    store.append(
        "independent_audit.jsonl",
        {
            "key": cloud_key,
            "generation_key": cloud_key,
            "model": CLOUD_REFERENCE_MODEL,
            "mode": "max",
            "benchmark": "novel-ja",
            "item_id": "novel-ja-01",
            "status": "ok",
            "evaluator_role": "independent_audit",
            "independent_audit": True,
            "judge": {"backend": "hermes", "provider": "openai-codex", "model": "gpt-5.6-luna", "reasoning": "max"},
            "scores": {"prose_quality": 7},
            "overall": 7.0,
        },
    )

    result = build_report(run_dir)

    ranking = list(csv.DictReader((run_dir / "data" / "ranking.csv").open(encoding="utf-8")))
    reference = list(csv.DictReader((run_dir / "data" / "reference_summary.csv").open(encoding="utf-8")))
    report_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["report"]
    readme = (run_dir / "README.md").read_text(encoding="utf-8")

    assert [row["model"] for row in ranking] == ["local-model"]
    assert {row["model"] for row in reference} == {CLOUD_REFERENCE_MODEL}
    assert reference[0]["self_judged_score"] == "8.0"
    assert reference[0]["luna_audit_score"] == "7.0"
    assert report_manifest["costs"]["primary_judge_usd"] == 0.006
    assert report_manifest["costs"]["cloud_reference_generation_usd"] == 0.01
    assert "self_judged" in readme
    assert result["reference_rows"] == 2
