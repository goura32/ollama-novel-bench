from pathlib import Path
from typing import Any

import pytest

from novelbench.cloud import (
    CLOUD_REFERENCE_MODEL,
    build_cloud_target_payload,
    cloud_generation_key,
    cloud_reference_config,
    run_cloud_reference,
)
from novelbench.datasets import BenchmarkItem
from novelbench.store import RunStore


class SequenceClient:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        return next(self.responses)


def response(content: str, thinking: str) -> dict[str, Any]:
    return {
        "id": "req-1",
        "model": CLOUD_REFERENCE_MODEL,
        "provider": "Together",
        "choices": [{"finish_reason": "stop", "message": {"content": content, "reasoning": thinking}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 4},
            "cost": 0.002,
        },
    }


def creative_item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="novel-ja-01",
        benchmark="novel-ja",
        category="test",
        prompt="短い話を書いてください。",
        rubric=("prose_quality",),
    )


def objective_item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="objective-01",
        benchmark="objective-ja",
        category="test",
        prompt="正解は？",
        choices=("A", "B"),
        answer=1,
    )


def test_cloud_reference_config_maps_min_max_to_low_max_and_is_separate():
    config = cloud_reference_config({"cloud_reference": {"enabled": True}}, "quick")
    assert config["enabled"] is True
    assert config["model"] == CLOUD_REFERENCE_MODEL
    assert config["reference_only"] is True
    assert config["ranking_eligible"] is False
    assert config["max_tokens_max"] >= 8192
    assert config["request_timeout_seconds"] == 180
    assert [(mode["label"], mode["effort"]) for mode in config["modes"]] == [
        ("min", "low"),
        ("max", "max"),
    ]

    assert cloud_reference_config({}, "smoke")["enabled"] is False
    assert cloud_reference_config({}, "smoke", force_enable=True)["enabled"] is True


def test_cloud_reference_defaults_follow_profile_and_can_be_disabled():
    assert cloud_reference_config({}, "quick")["enabled"] is True
    assert cloud_reference_config({}, "full")["enabled"] is True
    assert cloud_reference_config({}, "smoke")["enabled"] is False
    assert cloud_reference_config({}, "quick", disable=True)["enabled"] is False
    assert cloud_reference_config({}, "smoke", force_only=True)["only"] is True


def test_cloud_target_payload_uses_openrouter_reasoning_not_ollama_controls():
    payload = build_cloud_target_payload(creative_item(), effort="low", max_tokens=321)
    assert payload["model"] == CLOUD_REFERENCE_MODEL
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_tokens"] == 321
    assert payload["provider"] == {"only": ["z-ai"], "allow_fallbacks": False, "require_parameters": True}
    assert payload["usage"] == {"include": True}
    assert "options" not in payload
    assert "think" not in payload


def test_cloud_reference_records_both_modes_and_machine_scores(tmp_path: Path):
    client = SequenceClient(
        [
            response("B", "low thought"),
            response("創作本文", "max thought"),
            response("B", "low thought 2"),
            response("創作本文2", "max thought 2"),
        ]
    )
    store = RunStore(tmp_path / "run")
    config = cloud_reference_config(
        {"cloud_reference": {"enabled": True, "max_tokens": 321}}, "quick"
    )
    counts = run_cloud_reference(
        client,
        store,
        [objective_item(), creative_item()],
        config,
        max_retries=0,
    )

    assert counts == {"planned": 4, "skipped": 0, "ok": 4, "error": 0}
    records = list(store.read_jsonl("generations.jsonl"))
    assert {record["mode"] for record in records} == {"min", "max"}
    assert all(record["backend"] == "openrouter" for record in records)
    assert all(record["reference_only"] is True for record in records)
    assert all(record["ranking_eligible"] is False for record in records)
    assert {record["thinking_effort"] for record in records} == {"low", "max"}
    objective = next(record for record in records if record["benchmark"] == "objective-ja")
    assert objective["objective_score"] == 1.0
    assert all(record["cost_usd"] == 0.002 for record in records)
    assert cloud_generation_key(creative_item(), "min").startswith("cloud:")
    assert len(client.calls) == 4


def test_cloud_creative_self_judge_and_luna_audit_are_separate_lanes(tmp_path: Path, monkeypatch):
    target_client = SequenceClient(
        [
            response("B", "low"),
            response("本文", "max"),
            response("B", "low"),
            response("本文", "max"),
        ]
    )
    store = RunStore(tmp_path / "run")
    config = cloud_reference_config({"cloud_reference": {"enabled": True}}, "smoke")
    run_cloud_reference(target_client, store, [objective_item(), creative_item()], config, max_retries=0)

    class JudgeClient:
        def chat(self, _payload):
            return {
                "id": "judge-1",
                "model": CLOUD_REFERENCE_MODEL,
                "provider": "Together",
                "choices": [
                    {
                        "message": {
                            "content": '{"scores":{"prose_quality":6},"overall":6.0,"brief_rationale":"根拠"}'
                        }
                    }
                ],
                "usage": {"cost": 0.003, "reasoning_tokens": 8},
            }

    from novelbench import judge as judge_module

    primary_counts = judge_module.judge_run(
        tmp_path,
        store,
        [objective_item(), creative_item()],
        client=JudgeClient(),
        max_retries=0,
    )
    assert primary_counts == {"considered": 2, "judged": 2, "ok": 2, "error": 0, "skipped": 0}
    primary = list(store.read_jsonl("judgments.jsonl"))
    assert all(row["self_judged"] is True for row in primary)
    assert all(row["reference_only"] is True and row["ranking_eligible"] is False for row in primary)

    class FakeLuna:
        def __init__(self, *_args, **_kwargs):
            pass

        def judge_generation(self, generation, *, item):
            return {
                "key": generation["key"],
                "generation_key": generation["key"],
                "model": generation["model"],
                "mode": generation["mode"],
                "benchmark": item.benchmark,
                "item_id": item.item_id,
                "status": "ok",
                "evaluator_role": "independent_audit",
                "independent_audit": True,
                "reference_only": True,
                "ranking_eligible": False,
                "judge": {
                    "backend": "hermes",
                    "provider": "openai-codex",
                    "model": "gpt-5.6-luna",
                    "reasoning": "max",
                },
                "scores": {"prose_quality": 7},
                "overall": 7.0,
            }

    monkeypatch.setattr(judge_module, "LunaAuditRunner", FakeLuna)
    audit_counts = judge_module.luna_audit_run(
        tmp_path,
        store,
        [objective_item(), creative_item()],
        max_retries=0,
    )
    audits = list(store.read_jsonl("independent_audit.jsonl"))
    assert audit_counts == {"considered": 2, "audited": 2, "ok": 2, "error": 0, "skipped": 0}
    assert len(audits) == 2
    assert all(row["independent_audit"] is True for row in audits)


def test_cloud_empty_content_uses_finite_budget_rescue_and_aggregates_telemetry(tmp_path: Path):
    client = SequenceClient(
        [
            response("", "budget consumed"),
            response("本文", "rescued"),
            response("本文2", "max thought"),
        ]
    )
    store = RunStore(tmp_path / "run")
    config = cloud_reference_config(
        {"cloud_reference": {"enabled": True, "max_tokens": 10}}, "quick"
    )

    counts = run_cloud_reference(client, store, [creative_item()], config, max_retries=0)

    assert counts == {"planned": 2, "skipped": 0, "ok": 2, "error": 0}
    records = list(store.read_jsonl("generations.jsonl"))
    minimum = next(record for record in records if record["mode"] == "min")
    assert minimum["empty_content_rescue"]["used"] is True
    assert minimum["empty_content_rescue"]["rescue_request_count"] == 1
    assert minimum["final_max_tokens"] == 20
    assert minimum["attempts"] == 2
    assert minimum["reasoning_tokens"] == 8
    assert minimum["cost_usd"] == pytest.approx(0.004)
    assert [request["max_tokens"] for request in minimum["request_history"]] == [10, 20]
    assert len(client.calls) == 3
