"""Minimal native Ollama HTTP client.

Only the native endpoints needed by the benchmark are implemented.  Keeping the
client small makes the request payload and the stored response fields auditable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class OllamaError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        body = None
        headers = {"Accept": "application/json", "User-Agent": "ollama-novel-bench/0.1"}
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise OllamaError(
                f"Ollama {method} {path} returned HTTP {exc.code}",
                status=exc.code,
                body=detail[:2000],
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaError(f"Ollama {method} {path} failed: {exc}") from exc
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama {method} {path} returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise OllamaError(f"Ollama {method} {path} returned a non-object JSON value")
        return value

    def version(self) -> dict[str, Any]:
        return self._request("GET", "/api/version")

    def tags(self) -> dict[str, Any]:
        return self._request("GET", "/api/tags")

    def show(self, model: str) -> dict[str, Any]:
        return self._request("POST", "/api/show", {"model": model})

    def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = dict(payload)
        request.setdefault("stream", False)
        return self._request("POST", "/api/chat", request)
