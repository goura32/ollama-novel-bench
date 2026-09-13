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
    model_name: str,
    item: BenchmarkItem,
    mode: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    options: dict[str, Any] = dict(profile.get("ollama_options") or {})
    options["seed"] = int(profile.get("seed", 20260912))
    options["num_predict"] = int(profile.get("num_predict", 768))
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


def _response_summary(response_data: dict[str, Any], num_predict: int) -> dict[str, Any]:
    return {
        "num_predict": num_predict,
        "content_length": response_data["content_length"],
        "thinking_length": response_data["thinking_length"],
        "content_present": bool(str(response_data["content"]).strip()),
        "done": response_data["done"],
        "done_reason": response_data["done_reason"],
        "metrics": response_data["metrics"],
    }


def _empty_content_rescue_policy(profile: dict[str, Any]) -> dict[str, Any]:
    configured = profile.get("empty_content_rescue")
    settings = configured if isinstance(configured, dict) else {}
    multiplier = max(2, int(settings.get("multiplier", 2)))
    max_multiplier = max(multiplier, int(settings.get("max_multiplier", 4)))
    return {
        "enabled": bool(settings.get("enabled", True)),
        "multiplier": multiplier,
        "max_multiplier": max_multiplier,
    }


def _with_num_predict(payload: dict[str, Any], num_predict: int) -> dict[str, Any]:
    request = dict(payload)
    request["options"] = dict(payload["options"])
    request["options"]["num_predict"] = num_predict
    return request


def _budget_rescue_record(base_num_predict: int, policy: dict[str, Any]) -> dict[str, Any]:
    max_num_predict = (
        base_num_predict * policy["max_multiplier"]
        if policy["enabled"]
        else base_num_predict
    )
    return {
        "enabled": policy["enabled"],
        "trigger": "empty_content_or_subjective_length",
        "base_num_predict": base_num_predict,
        "multiplier": policy["multiplier"],
        "max_multiplier": policy["max_multiplier"],
        "max_num_predict": max_num_predict,
        "used": False,
        "exhausted": False,
        "rescue_request_count": 0,
        "response_summaries": [],
    }


def run_generations(
    client: OllamaClient,
    store: RunStore,
    models: list[dict[str, Any]],
    items: list[BenchmarkItem],
    profile: dict[str, Any],
) -> dict[str, int]:
    """Run model × thinking mode × item in a single deterministic loop.

    Output-budget rescue is handled separately from HTTP/network retries. The
    first request always uses the profile budget. Empty output, or a subjective
    creative response ending with done_reason=length, triggers the finite 2x
    budget rescue sequence.
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
            payload = build_chat_payload(model_name, item, mode, profile)
            base_num_predict = int(payload["options"]["num_predict"])
            policy = _empty_content_rescue_policy(profile)
            rescue = _budget_rescue_record(base_num_predict, policy)
            request_payload = payload
            current_num_predict = base_num_predict
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
            while True:
                attempt += 1
                try:
                    response = client.chat(request_payload)
                except Exception as exc:  # finite HTTP/network retries, same budget
                    last_error = str(exc)[:2000]
                    network_retries += 1
                    retry_history.append(
                        {
                            "kind": "http_or_network"
                            if isinstance(exc, OllamaError)
                            else "client_error",
                            "attempt": attempt,
                            "retry_index": network_retries,
                            "num_predict": current_num_predict,
                            "http_status": getattr(exc, "status", None),
                            "error": last_error,
                        }
                    )
                    if network_retries <= max_retries:
                        time.sleep(min(2.0, 0.25 * network_retries))
                        continue
                    break

                response_data = _response_record(response)
                is_rescue_request = rescue["used"]
                if is_rescue_request:
                    rescue["response_summaries"].append(
                        _response_summary(response_data, current_num_predict)
                    )
                content_present = bool(str(response_data["content"]).strip())
                subjective_truncated = (
                    content_present
                    and response_data.get("done_reason") == "length"
                    and item.benchmark != "objective-ja"
                )
                if content_present and not subjective_truncated:
                    record.update(
                        {
                            "status": "ok",
                            "attempts": attempt,
                            "network_retries": network_retries,
                            "retry_history": retry_history,
                            "initial_num_predict": base_num_predict,
                            "final_num_predict": current_num_predict,
                            "empty_content_rescue": rescue,
                            "content": response_data["content"],
                            "thinking": response_data["thinking"],
                            "thinking_length": response_data["thinking_length"],
                            "content_length": response_data["content_length"],
                            **response_data["metrics"],
                            "response": response_data,
                            "finished_at": utc_now(),
                        }
                    )
                    if item.benchmark == "objective-ja":
                        score, prediction = score_multiple_choice(response_data["content"], item)
                        record["objective_score"] = score
                        record["prediction"] = prediction
                    counts["ok"] += 1
                    break

                if not is_rescue_request:
                    rescue["response_summaries"].append(
                        _response_summary(response_data, current_num_predict)
                    )
                last_error = (
                    "Ollama response hit the output limit before completing subjective content"
                    if subjective_truncated
                    else "Ollama returned an empty content field"
                )
                next_num_predict = current_num_predict * policy["multiplier"]
                if (
                    not policy["enabled"]
                    or next_num_predict > rescue["max_num_predict"]
                ):
                    rescue["exhausted"] = True
                    break
                rescue["used"] = True
                rescue["rescue_request_count"] += 1
                current_num_predict = next_num_predict
                request_payload = _with_num_predict(payload, current_num_predict)

            if record.get("status") != "ok":
                record.update(
                    {
                        "status": "error",
                        "attempts": attempt,
                        "network_retries": network_retries,
                        "retry_history": retry_history,
                        "initial_num_predict": base_num_predict,
                        "final_num_predict": current_num_predict,
                        "empty_content_rescue": rescue,
                        "error": last_error or "unknown generation failure",
                        "finished_at": utc_now(),
                    }
                )
                counts["error"] += 1
            store.append("generations.jsonl", record)
            finished.add(key)
    return counts
