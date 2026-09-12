from novelbench.models import (
    detect_thinking_modes,
    filter_generation_models,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def chat(self, payload):
        self.calls.append(payload)
        key = (payload.get("think"), payload.get("reasoning_effort"))
        return self.responses.get(key, {"message": {"content": "ok"}})


def test_embedding_only_models_are_excluded():
    models = [
        {"name": "text", "capabilities": ["completion"]},
        {"name": "embed", "capabilities": ["embedding"]},
        {"name": "both", "capabilities": ["embedding", "completion"]},
    ]

    assert [m["name"] for m in filter_generation_models(models)] == ["text", "both"]


def test_binary_thinking_uses_false_and_true_modes():
    client = FakeClient(
        {
            (False, None): {"message": {"content": "off"}},
            (True, None): {"message": {"thinking": "考える", "content": "on"}},
        }
    )

    result = detect_thinking_modes(
        client,
        {"name": "toy", "capabilities": ["completion", "thinking"]},
    )

    assert result["kind"] == "binary"
    assert result["min"]["value"] is False
    assert result["max"]["value"] is True
    assert result["probe_results"]
    assert any(p["thinking_length"] > 0 for p in result["probe_results"])


def test_level_thinking_prefers_highest_observed_level():
    client = FakeClient(
        {
            (False, None): {"message": {"content": "off"}},
            (True, None): {"message": {"thinking": "default", "content": "on"}},
            ("low", None): {"message": {"thinking": "短い", "content": "low"}},
            ("high", None): {"message": {"thinking": "長い思考", "content": "high"}},
            ("max", None): {"message": {"thinking": "最大", "content": "max"}},
        }
    )

    result = detect_thinking_modes(
        client,
        {"name": "toy", "capabilities": ["completion", "thinking"]},
    )

    assert result["kind"] == "levels"
    assert result["max"]["value"] == "max"
    assert result["min"]["value"] in (False, "none", "off")


def test_non_thinking_model_has_one_none_mode():
    client = FakeClient({})
    result = detect_thinking_modes(
        client,
        {"name": "plain", "capabilities": ["completion"]},
    )
    assert result["kind"] == "none"
    assert result["modes"] == [{"label": "none", "field": None, "value": None}]
    assert client.calls == []
