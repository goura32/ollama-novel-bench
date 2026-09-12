from pathlib import Path

from novelbench.reporting import build_report
from novelbench.store import RunStore


def test_report_writes_machine_readable_summary_and_ranked_readme(tmp_path: Path):
    run_dir = tmp_path / "results" / "r1"
    store = RunStore(run_dir)
    store.write_manifest(
        {
            "run_id": "r1",
            "profile": "smoke",
            "status": "complete",
            "smoke_warning": "予備試験・ランキング用途不可",
            "models": [{"name": "demo", "parameter_size": "1B"}],
        }
    )
    for mode, score, speed in (("min", 5.0, 10.0), ("max", 7.0, 6.0)):
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
                "eval_count": 10,
                "eval_duration": int(1e9 / speed * 10),
                "tok_s": speed,
            },
        )
        store.append(
            "judgments.jsonl",
            {
                "key": f"demo|{mode}|novel-ja-01",
                "model": "demo",
                "mode": mode,
                "benchmark": "novel-ja",
                "item_id": "novel-ja-01",
                "status": "ok",
                "scores": {
                    "instruction_following": score,
                    "japanese_naturalness": score,
                    "prose_quality": score,
                    "show_dont_tell": score,
                    "character_emotion": score,
                    "coherence": score,
                    "originality": score,
                    "ending_resonance": score,
                },
                "overall": score,
            },
        )

    result = build_report(run_dir)

    assert result["rows"] == 2
    assert (run_dir / "data" / "summary.csv").exists()
    assert (run_dir / "README.md").read_text().find("予備試験・ランキング用途不可") >= 0
