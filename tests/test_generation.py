from pathlib import Path
from typing import Any

from novelbench.datasets import BenchmarkItem
from novelbench.generation import run_generations
from novelbench.ollama import OllamaError
from novelbench.store import RunStore


class SequenceClient:
    def __init__(self, responses: list[Any]):
        self.responses = iter(responses)
        self.calls: list[dict[str, Any]] = []

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response


def item() -> BenchmarkItem:
    return BenchmarkItem(
        item_id="novel-ja-01",
        benchmark="novel-ja",
        category="test",
        prompt="短い物語を書いてください。",
        rubric=("prose_quality",),
    )


def response(*, content: str, thinking: str = "考え中", done_reason: str = "length") -> dict[str, Any]:
    return {
        "message": {"content": content, "thinking": thinking},
        "done": True,
        "done_reason": done_reason,
        "eval_count": 10,
        "eval_duration": 1_000_000_000,
    }


def test_empty_content_uses_doubling_budget_rescue_and_records_summaries(tmp_path: Path):
    client = SequenceClient(
        [
            response(content=""),
            response(content="途中までの本文", done_reason="length"),
            response(content="完成した本文", done_reason="stop"),
        ]
    )
    store = RunStore(tmp_path / "run")

    counts = run_generations(
        client,
        store,
        [{"name": "demo", "thinking": {"kind": "levels", "modes": [
            {"label": "max", "field": "think", "value": "max"}
        ]}}],
        [item()],
        {"num_predict": 384, "max_retries": 0},
    )

    record = next(store.read_jsonl("generations.jsonl"))
    rescue = record["empty_content_rescue"]
    assert counts == {"planned": 1, "skipped": 0, "ok": 1, "error": 0}
    assert [call["options"]["num_predict"] for call in client.calls] == [384, 768, 1536]
    assert record["api_request"]["options"]["num_predict"] == 384
    assert record["final_num_predict"] == 1536
    assert rescue["used"] is True
    assert [summary["num_predict"] for summary in rescue["response_summaries"]] == [384, 768, 1536]
    assert rescue["response_summaries"][0]["content_length"] == 0
    assert rescue["response_summaries"][1]["done_reason"] == "length"


def test_nonempty_subjective_length_response_uses_budget_rescue(tmp_path: Path):
    client = SequenceClient([
        response(content="途中本文", done_reason="length"),
        response(content="完成本文", done_reason="stop"),
    ])
    store = RunStore(tmp_path / "run")

    counts = run_generations(
        client,
        store,
        [{"name": "demo", "thinking": {"modes": [
            {"label": "max", "field": "think", "value": "max"}
        ]}}],
        [item()],
        {"num_predict": 384, "max_retries": 0},
    )

    record = next(store.read_jsonl("generations.jsonl"))
    assert counts["ok"] == 1
    assert len(client.calls) == 2
    assert record["status"] == "ok"
    assert record["response"]["done_reason"] == "stop"
    assert record["empty_content_rescue"]["used"] is True
    assert [s["done_reason"] for s in record["empty_content_rescue"]["response_summaries"]] == ["length", "stop"]


def test_transport_retry_keeps_budget_and_is_recorded_separately(tmp_path: Path):
    client = SequenceClient(
        [OllamaError("temporary outage", status=503), response(content="本文", done_reason="stop")]
    )
    store = RunStore(tmp_path / "run")

    counts = run_generations(
        client,
        store,
        [{"name": "demo", "thinking": {"modes": [
            {"label": "max", "field": "think", "value": "max"}
        ]}}],
        [item()],
        {"num_predict": 384, "max_retries": 1},
    )

    record = next(store.read_jsonl("generations.jsonl"))
    assert counts["ok"] == 1
    assert [call["options"]["num_predict"] for call in client.calls] == [384, 384]
    assert record["network_retries"] == 1
    assert record["empty_content_rescue"]["used"] is False
    assert record["retry_history"][0]["kind"] == "http_or_network"


def test_empty_content_rescue_has_finite_profile_cap(tmp_path: Path):
    client = SequenceClient([response(content=""), response(content=""), response(content="")])
    store = RunStore(tmp_path / "run")

    counts = run_generations(
        client,
        store,
        [{"name": "demo", "thinking": {"modes": [
            {"label": "max", "field": "think", "value": "max"}
        ]}}],
        [item()],
        {"num_predict": 384, "max_retries": 0},
    )

    record = next(store.read_jsonl("generations.jsonl"))
    assert counts["error"] == 1
    assert [call["options"]["num_predict"] for call in client.calls] == [384, 768, 1536]
    assert record["empty_content_rescue"]["max_num_predict"] == 1536
    assert record["empty_content_rescue"]["exhausted"] is True
