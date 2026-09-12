import csv
import json
from pathlib import Path

from novelbench.reporting import build_report
from novelbench.store import RunStore


def test_report_tracks_empty_content_rescue_per_mode_and_manifest(tmp_path: Path):
    run_dir = tmp_path / "results" / "r1"
    store = RunStore(run_dir)
    store.write_manifest(
        {
            "run_id": "r1",
            "profile": "smoke",
            "status": "complete",
            "smoke_warning": "予備試験・ランキング用途不可",
            "item_count": 1,
            "item_ids": ["novel-ja-01"],
            "models": [{"name": "demo", "parameter_size": "1B"}],
            "generation_policy": {
                "initial_num_predict": 384,
                "empty_content_rescue": {"enabled": True, "multiplier": 2, "max_multiplier": 4},
            },
        }
    )
    for mode in ("min", "max"):
        rescue = {
            "enabled": True,
            "trigger": "empty_content",
            "base_num_predict": 384,
            "multiplier": 2,
            "max_multiplier": 4,
            "max_num_predict": 1536,
            "used": mode == "max",
            "exhausted": False,
            "rescue_request_count": 1 if mode == "max" else 0,
            "response_summaries": [],
        }
        store.append(
            "generations.jsonl",
            {
                "key": f"demo|{mode}|novel-ja-01",
                "model": "demo",
                "mode": mode,
                "benchmark": "novel-ja",
                "item_id": "novel-ja-01",
                "status": "ok",
                "content": "文章",
                "content_length": 2,
                "thinking_length": 0,
                "tok_s": 10.0,
                "empty_content_rescue": rescue,
            },
        )

    result = build_report(run_dir)
    summary = list(csv.DictReader((run_dir / "data" / "summary.csv").open(encoding="utf-8")))
    paired = list(csv.DictReader((run_dir / "data" / "paired.csv").open(encoding="utf-8")))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    readme = (run_dir / "README.md").read_text(encoding="utf-8")

    assert result["budget_rescue_records"] == 1
    assert result["budget_rescue_attempts"] == 1
    assert {row["mode"]: row["budget_rescue_n"] for row in summary} == {"min": "0", "max": "1"}
    assert paired[0]["budget_rescue_min_n"] == "0"
    assert paired[0]["budget_rescue_max_n"] == "1"
    assert manifest["report"]["budget_rescue_records"] == 1
    assert "budget rescue" in readme
    assert "rescue min" in readme
