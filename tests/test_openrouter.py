import json
from pathlib import Path
from typing import Any

import pytest

from novelbench.openrouter import (
    OPENROUTER_API_URL,
    OpenRouterClient,
    OpenRouterError,
    extract_cost_usd,
    extract_reasoning_tokens,
    message_parts,
    safe_request_subset,
    safe_response_subset,
)


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RecordingOpener:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return FakeHTTPResponse(self.payload)


def test_openrouter_client_posts_directly_without_persisting_auth(tmp_path: Path):
    key = "sk-test-secret-value"
    key_path = tmp_path / "openrouter.key"
    key_path.write_text(key + "\n", encoding="utf-8")
    opener = RecordingOpener(
        {
            "id": "req-1",
            "model": "z-ai/glm-5.3-flash",
            "provider": "Together",
            "choices": [{"message": {"content": "{}"}}],
        }
    )

    client = OpenRouterClient(key_path=key_path, opener=opener)
    response = client.chat({"model": "z-ai/glm-5.3-flash", "messages": []})

    assert response["id"] == "req-1"
    assert opener.request.full_url == OPENROUTER_API_URL
    assert opener.request.headers["Authorization"] == f"Bearer {key}"
    assert key not in json.dumps(response)
    assert opener.timeout == client.timeout


def test_openrouter_client_rejects_empty_key(tmp_path: Path):
    key_path = tmp_path / "openrouter.key"
    key_path.write_text(" \n", encoding="utf-8")

    with pytest.raises(OpenRouterError, match="empty"):
        OpenRouterClient(key_path=key_path, opener=RecordingOpener({}))


def test_openrouter_safe_metadata_and_reasoning_cost_adapters():
    payload = {
        "model": "z-ai/glm-5.3-flash",
        "reasoning": {"effort": "max"},
        "provider": {"sort": "price", "require_parameters": True},
        "response_format": {"type": "json_schema", "json_schema": {"strict": True}},
        "messages": [{"role": "user", "content": "secret prompt"}],
        "Authorization": "Bearer sk-do-not-store",
    }
    response = {
        "id": "req-1",
        "model": "z-ai/glm-5.3-flash",
        "provider": "Together",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": "本文",
                    "reasoning": "思考",
                    "reasoning_details": [{"type": "text", "text": "思考"}],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "cost": 0.00123,
        },
    }

    request_safe = safe_request_subset(payload, prompt_sha256="abc")
    response_safe = safe_response_subset(response)
    content, thinking = message_parts(response)

    serialized = json.dumps({"request": request_safe, "response": response_safe})
    assert "Authorization" not in serialized
    assert "sk-do-not-store" not in serialized
    assert "secret prompt" not in serialized
    assert request_safe["model"] == "z-ai/glm-5.3-flash"
    assert request_safe["reasoning"] == {"effort": "max"}
    assert request_safe["provider"]["require_parameters"] is True
    assert request_safe["response_format"]["strict"] is True
    assert response_safe["provider"] == "Together"
    assert content == "本文"
    assert thinking == "思考"
    assert extract_reasoning_tokens(response["usage"]) == 12
    assert extract_cost_usd(response["usage"]) == pytest.approx(0.00123)
