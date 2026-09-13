from pathlib import Path
from typing import Any

import pytest

from novelbench.datasets import BenchmarkItem
from novelbench.generation import build_chat_payload, run_generations
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
        rubric=("prose_quality",),    )


def model(context_length: int = 131072) -> dict[str, Any]:
    return {
        "name": "demo",
        "context_length": context_length,
        "thinking": {
            "kind": "levels",
            "modes": [{"label": "max", "field": "think", "value": "max"}],
        },
    }


def response(*, content: str, thinking: str = "考え中", done_reason: str = "stop") -> dict[str, Any]:
    return {
        "message": {"content": content, "thinking": thinking},
        "done": True,
        "done_reason": done_reason,
        "eval_count": 10,
        "eval_duration": 1_000_000_000,
    }


def test_native_context_is_used_and_num_predict_is_omitted(tmp_path: Path):
    client = SequenceClient([response(content="完成した本文")])
    store = RunStore(tmp_path / "run")
    counts = run_generations(client, store, [model(262144)], [item()], {"max_retries": 0})
    record = next(store.read_jsonl("generations.jsonl"))
    assert counts == {"planned": 1, "skipped": 0, "ok": 1, "error": 0}
    assert client.calls[0]["options"]["num_ctx"] == 262144
    assert "num_predict" not in client.calls[0]["options"]
    assert record["native_context_length"] == 262144
    assert record["effective_num_ctx"] == 262144
    assert record["num_predict"] is None
    assert record["context_exhausted"] is False


def test_context_exhaustion_is_recorded_without_budget_rescue(tmp_path: Path):
    client = SequenceClient([response(content="途中本文", done_reason="length")])
    store = RunStore(tmp_path / "run")
    counts = run_generations(client, store, [model()], [item()], {"max_retries": 0})
    record = next(store.read_jsonl("generations.jsonl"))
    assert counts["error"] == 1
    assert len(client.calls) == 1
    assert record["context_exhausted"] is True
    assert "model context" in record["error"]


def test_empty_content_is_recorded_without_regeneration(tmp_path: Path):
    client = SequenceClient([response(content="")])
    store = RunStore(tmp_path / "run")
    counts = run_generations(client, store, [model()], [item()], {"max_retries": 0})
    record = next(store.read_jsonl("generations.jsonl"))
    assert counts["error"] == 1
    assert len(client.calls) == 1
    assert record["error"] == "Ollama returned an empty content field"


def test_transport_retry_repeats_exact_native_context_request(tmp_path: Path):
    client = SequenceClient([
        OllamaError("temporary outage", status=503),
        response(content="本文"),
    ])
    store = RunStore(tmp_path / "run")
    counts = run_generations(client, store, [model()], [item()], {"max_retries": 1})
    record = next(store.read_jsonl("generations.jsonl"))
    assert counts["ok"] == 1
    assert client.calls[0] == client.calls[1]
    assert client.calls[0]["options"]["num_ctx"] == 131072
    assert "num_predict" not in client.calls[0]["options"]
    assert record["network_retries"] == 1
    assert record["retry_history"][0]["kind"] == "http_or_network"


def test_explicit_profile_num_predict_is_kept_for_external_benchmark():
    payload = build_chat_payload(
        model(),
        item(),
        {"label": "max", "field": "think", "value": "max"},
        {"num_predict": 1400},
    )
    assert payload["options"]["num_ctx"] == 131072
    assert payload["options"]["num_predict"] == 1400


def test_missing_native_context_is_rejected():
    bad_model = model()
    bad_model.pop("context_length")
    with pytest.raises(ValueError, match="native context length"):
        build_chat_payload(
            bad_model,
            item(),
            {"label": "max", "field": "think", "value": "max"},
            {},
        )
