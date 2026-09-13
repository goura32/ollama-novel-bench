"""Strictly serial target generation and append-safe persistence."""

from __future__ import annotations

import time
from typing import Any

from .datasets import BenchmarkItem, render_item_prompt, score_multiple_choice
from .models import _message_parts
from .ollama import OllamaClient, OllamaError
from .store import RunStore
from .util import utc_now

_METRIC_FIELDS = (
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
    "prompt_eval_count",
    "eval_count",
)


def generation_key(model_name: str, mode_label: str, item_id: str) -> str:
    return f"{model_name}|{mode_label}|{item_id}"


def _mode_payload(payload: dict[str, Any], mode: dict[str, Any]) -> None:
    field = mode.get("field")
    if field:
        payload[str(field)] = mode.get("value")


def build_chat_payload(
    model: dict[str, Any],
    item: BenchmarkItem,
    mode: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    model_name = str(model["name"])
    context_length = model.get("context_length")
    if not isinstance(context_length, int) or context_length <= 0:
        raise ValueError(f"native context length is unavailable for {model_name}")
    options: dict[str, Any] = dict(profile.get("ollama_options") or {})
    options["seed"] = int(profile.get("seed", 20260912))
    options["num_ctx"] = context_length
    # Normal novel-writing profiles intentionally omit num_predict so generation
    # can run until the model naturally stops or its native context is exhausted.
    # A profile may still set num_predict when an external benchmark specification
    # requires a fixed generation cap (currently EQ-CW only).
    if profile.get("num_predict") is not None:
        options["num_predict"] = int(profile["num_predict"])
    # Temperature/min_p are deliberately absent for all normal profiles.  The
    # optional EQ-CW profile is the only profile that requests upstream settings.
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": render_item_prompt(item)}],
        "stream": False,
        "options": options,
    }
    _mode_payload(payload, mode)
    return payload


def _metrics(response: dict[str, Any]) -> dict[str, Any]:
    metrics = {field: response.get(field) for field in _METRIC_FIELDS if field in response}
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    if (
        isinstance(eval_count, (int, float))
        and isinstance(eval_duration, (int, float))
        and eval_duration > 0
    ):
        metrics["tok_s"] = float(eval_count) / (float(eval_duration) / 1_000_000_000)
    else:
        metrics["tok_s"] = None
    return metrics


def _response_record(response: dict[str, Any]) -> dict[str, Any]:
    content, thinking = _message_parts(response)
    metrics = _metrics(response)
    return {
        "content": content,
        "thinking": thinking,
        "thinking_length": len(thinking),
        "content_length": len(content),
        "metrics": metrics,
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
    }


def run_generations(
    client: OllamaClient,
    store: RunStore,
    models: list[dict[str, Any]],
    items: list[BenchmarkItem],
    profile: dict[str, Any],
) -> dict[str, int]:
    """Run model × thinking mode × item in a single deterministic loop.

    Each model runs with its discovered native maximum context length. Normal
    novel-writing profiles do not set ``num_predict``: the model must stop
    naturally, otherwise context exhaustion is recorded as a model behavior.
    HTTP/network retries repeat the exact same request.
    """
    finished = store.completed_keys("generations.jsonl")
    counts = {"planned": 0, "skipped": 0, "ok": 0, "error": 0}
    modes_by_model: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for model in models:
        modes = list((model.get("thinking") or {}).get("modes") or [])
        if not modes:
            modes = [{"label": "none", "field": None, "value": None}]
        for mode in modes:
            modes_by_model.append((model, mode))
    counts["planned"] = len(modes_by_model) * len(items)

    for model, mode in modes_by_model:
        model_name = str(model["name"])
        for item in items:
            key = generation_key(model_name, str(mode["label"]), item.item_id)
            if key in finished:
                counts["skipped"] += 1
                continue
            payload = build_chat_payload(model, item, mode, profile)
            max_retries = int(profile.get("max_retries", 2))
            network_retries = 0
            retry_history: list[dict[str, Any]] = []
            attempt = 0
            last_error = ""
            record: dict[str, Any] = {
                "key": key,
                "model": model_name,
                "mode": mode.get("label"),
                "thinking_kind": (model.get("thinking") or {}).get("kind"),
                "model_digest": model.get("digest"),
                "parameter_size": model.get("parameter_size"),
                "native_context_length": model.get("context_length"),
                "effective_num_ctx": payload["options"].get("num_ctx"),
                "num_predict": payload["options"].get("num_predict"),
                "backend": "ollama",
                "provider": "ollama",
                "provider_actual": "ollama",
                "model_actual": model_name,
                "reference_only": False,
                "ranking_eligible": True,
                "target_role": "local",
                "benchmark": item.benchmark,
                "category": item.category,
                "item_id": item.item_id,
                "item": item.to_dict(),
                "api_request": payload,
                "started_at": utc_now(),
            }
            response_data: dict[str, Any] | None = None
            while True:
                attempt += 1
                try:
                    response = client.chat(payload)
                    response_data = _response_record(response)
                    break
                except Exception as exc:
                    last_error = str(exc)[:2000]
                    network_retries += 1
                    retry_history.append(
                        {
                            "kind": "http_or_network"
                            if isinstance(exc, OllamaError)
                            else "client_error",
                            "attempt": attempt,
                            "retry_index": network_retries,
                            "http_status": getattr(exc, "status", None),
                            "error": last_error,
                        }
                    )
                    if network_retries <= max_retries:
                        time.sleep(min(2.0, 0.25 * network_retries))
                        continue
                    break

            if response_data is not None:
                content_present = bool(str(response_data["content"]).strip())
                context_exhausted = response_data.get("done_reason") == "length"
                record.update(
                    {
                        "attempts": attempt,
                        "network_retries": network_retries,
                        "retry_history": retry_history,
                        "context_exhausted": context_exhausted,
                        "content": response_data["content"],
                        "thinking": response_data["thinking"],
                        "thinking_length": response_data["thinking_length"],
                        "content_length": response_data["content_length"],
                        **response_data["metrics"],
                        "response": response_data,
                        "finished_at": utc_now(),
                    }
                )
                if content_present and not context_exhausted:
                    record["status"] = "ok"
                    if item.benchmark == "objective-ja":
                        score, prediction = score_multiple_choice(response_data["content"], item)
                        record["objective_score"] = score
                        record["prediction"] = prediction
                    counts["ok"] += 1
                else:
                    record["status"] = "error"
                    record["error"] = (
                        "Ollama exhausted the model context before natural completion"
                        if context_exhausted
                        else "Ollama returned an empty content field"
                    )
                    counts["error"] += 1
            else:
                record.update(
                    {
                        "status": "error",
                        "attempts": attempt,
                        "network_retries": network_retries,
                        "retry_history": retry_history,
                        "context_exhausted": False,
                        "error": last_error or "unknown generation failure",
                        "finished_at": utc_now(),
                    }
                )
                counts["error"] += 1

            store.append("generations.jsonl", record)
            finished.add(key)
    return counts
