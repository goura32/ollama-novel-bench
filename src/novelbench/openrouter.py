"""Direct OpenRouter transport and secret-safe response metadata helpers."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_KEY_PATH = Path.home() / ".config" / "credstore" / "openrouter.key"
OPENROUTER_USER_AGENT = "ollama-novel-bench/0.2"

_SECRET_KEY_NAMES = {
    "authorization",
    "api_key",
    "apikey",
    "key",
    "password",
    "secret",
    "token",
}


class OpenRouterError(RuntimeError):
    """An OpenRouter transport or response error with no credential material."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message)
        self.status = status


def load_openrouter_key(path: Path | None = None) -> str:
    """Read the API key from the configured local credential file."""
    key_path = Path(path or OPENROUTER_KEY_PATH).expanduser()
    try:
        value = key_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OpenRouterError(f"OpenRouter key file is unavailable: {key_path}") from exc
    if not value:
        raise OpenRouterError(f"OpenRouter key file is empty: {key_path}")
    return value


def _sanitize(value: Any, *, secret: str | None = None, depth: int = 0) -> Any:
    """Return JSON-safe diagnostics with credential-looking fields removed."""
    if depth > 12:
        return "<truncated>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower().replace("-", "_") in _SECRET_KEY_NAMES:
                continue
            result[key_text] = _sanitize(item, secret=secret, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [_sanitize(item, secret=secret, depth=depth + 1) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item, secret=secret, depth=depth + 1) for item in value]
    if isinstance(value, str):
        text = value
        if secret:
            text = text.replace(secret, "<redacted>")
            text = text.replace(f"Bearer {secret}", "Bearer <redacted>")
        text = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~-]{12,}", "Bearer <redacted>", text)
        return text
    return value


def _text_part(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return "" if value is None else str(value)


def message_parts(response: dict[str, Any]) -> tuple[str, str]:
    """Extract OpenAI-compatible content and reasoning/thinking text."""
    choices = response.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}
    content = _text_part(message.get("content", ""))
    reasoning = _text_part(message.get("reasoning", message.get("thinking", "")))
    if not reasoning and isinstance(message.get("reasoning_details"), list):
        reasoning = _text_part(message["reasoning_details"])
    return content, reasoning


def extract_usage(response_or_usage: dict[str, Any]) -> dict[str, Any]:
    """Return the usage object with nested secret-looking fields removed."""
    usage = response_or_usage.get("usage", response_or_usage)
    return dict(_sanitize(usage if isinstance(usage, dict) else {}) or {})


def extract_reasoning_tokens(usage: dict[str, Any] | None) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    candidates = [
        usage.get("reasoning_tokens"),
        details.get("reasoning_tokens") if isinstance(details, dict) else None,
        usage.get("reasoning_token_count"),
    ]
    for value in candidates:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def extract_cost_usd(usage: dict[str, Any] | None) -> float | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("cost_details")
    candidates = [
        usage.get("cost"),
        usage.get("total_cost"),
        usage.get("estimated_cost_usd"),
        details.get("upstream_inference_cost") if isinstance(details, dict) else None,
    ]
    for value in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return None


def actual_provider(response: dict[str, Any]) -> str | None:
    if isinstance(response.get("provider"), str):
        return response["provider"]
    if isinstance(response.get("provider_name"), str):
        return response["provider_name"]
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        provider = choices[0].get("provider")
        if isinstance(provider, str):
            return provider
    return None


def actual_model(response: dict[str, Any]) -> str | None:
    value = response.get("model")
    return value if isinstance(value, str) else None


def safe_request_subset(payload: dict[str, Any], *, prompt_sha256: str | None = None) -> dict[str, Any]:
    """Keep only auditable request controls; never persist prompt or headers."""
    result: dict[str, Any] = {
        "endpoint": OPENROUTER_API_URL,
        "model": payload.get("model"),
        "reasoning": _sanitize(payload.get("reasoning")),
        "provider": _sanitize(payload.get("provider")),
        "max_tokens": payload.get("max_tokens"),
        "response_format": {
            "type": (payload.get("response_format") or {}).get("type")
            if isinstance(payload.get("response_format"), dict)
            else None,
            "name": (
                ((payload.get("response_format") or {}).get("json_schema") or {}).get("name")
                if isinstance(payload.get("response_format"), dict)
                else None
            ),
            "strict": (
                ((payload.get("response_format") or {}).get("json_schema") or {}).get("strict")
                if isinstance(payload.get("response_format"), dict)
                else None
            ),
        },
        "usage": _sanitize(payload.get("usage")),
        "message_count": len(payload.get("messages", []))
        if isinstance(payload.get("messages"), list)
        else None,
    }
    if prompt_sha256:
        result["prompt_sha256"] = prompt_sha256
    return result


def safe_response_subset(response: dict[str, Any]) -> dict[str, Any]:
    """Persist response routing/usage without duplicating model text."""
    content, reasoning = message_parts(response)
    choices = response.get("choices")
    choice_metadata: list[dict[str, Any]] = []
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            choice_metadata.append(
                {
                    "index": choice.get("index"),
                    "finish_reason": choice.get("finish_reason"),
                    "native_finish_reason": choice.get("native_finish_reason"),
                }
            )
    usage = extract_usage(response)
    return {
        "id": response.get("id"),
        "model": actual_model(response),
        "provider": actual_provider(response),
        "created": response.get("created"),
        "system_fingerprint": response.get("system_fingerprint"),
        "choices": choice_metadata,
        "content_length": len(content),
        "reasoning_length": len(reasoning),
        "usage": usage,
    }


class OpenRouterClient:
    """Small direct client; the key stays in memory and never enters payload metadata."""

    def __init__(
        self,
        *,
        api_url: str = OPENROUTER_API_URL,
        key_path: Path | None = None,
        timeout: float = 180,
        opener: Callable[..., Any] | None = None,
        key: str | None = None,
    ):
        self.api_url = api_url
        self.timeout = timeout
        self._opener = opener or urllib.request.urlopen
        self._key = key if key is not None else load_openrouter_key(key_path)
        if not self._key.strip():
            raise OpenRouterError("OpenRouter key is empty")

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
                "User-Agent": OPENROUTER_USER_AGENT,
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            try:
                exc.read()
            except OSError:
                pass
            raise OpenRouterError(
                f"OpenRouter returned HTTP {exc.code}", status=exc.code
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OpenRouterError("OpenRouter request failed") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenRouterError("OpenRouter returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise OpenRouterError("OpenRouter returned a non-object JSON value")
        return value
