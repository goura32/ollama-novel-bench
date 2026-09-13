"""Cloud reference target generation through OpenRouter."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .datasets import BenchmarkItem, render_item_prompt, score_multiple_choice
from .openrouter import (
    OpenRouterClient,
    actual_model,
    actual_provider,
    extract_cost_usd,
    extract_reasoning_tokens,
    extract_usage,
    message_parts,
    safe_request_subset,
    safe_response_subset,
)
from .store import RunStore
from .util import atomic_write_text, json_dumps, safe_filename, sha256_bytes, utc_now

CLOUD_REFERENCE_BACKEND = "openrouter"
CLOUD_REFERENCE_PROVIDER = "openrouter"
CLOUD_REFERENCE_MODEL = "z-ai/glm-5.3-flash"
CLOUD_REFERENCE_MODES = (
    {"label": "min", "effort": "low"},
    {"label": "max", "effort": "max"},
)


def _total_int(values: list[int | None]) -> int | None:
    if not values or not all(isinstance(value, int) for value in values):
        return None
    return sum(value for value in values if value is not None)


def _total_float(values: list[float | None]) -> float | None:
    if not values or not all(isinstance(value, (int, float)) for value in values):
        return None
    return sum(float(value) for value in values if value is not None)


def cloud_generation_key(item: BenchmarkItem, mode: str) -> str:
    return f"cloud:{CLOUD_REFERENCE_MODEL}|{mode}|{item.item_id}"


def cloud_reference_config(
    profile: dict[str, Any],
    profile_name: str,
    *,
    force_enable: bool = False,
    disable: bool = False,
    force_only: bool = False,
) -> dict[str, Any]:
    """Normalize the frozen cloud-target plan for a new or resumed run."""
    configured = profile.get("cloud_reference")
    configured = dict(configured) if isinstance(configured, dict) else {}
    configured_model = configured.get("model")
    if configured_model and configured_model != CLOUD_REFERENCE_MODEL:
        raise ValueError(
            f"cloud reference model is fixed to {CLOUD_REFERENCE_MODEL}, not {configured_model}"
        )
    enabled = bool(configured.get("enabled", profile_name in {"quick", "full"}))
    if force_enable or force_only:
        enabled = True
    if disable:
        enabled = False
    modes: list[dict[str, str]] = []
    raw_modes = configured.get("modes") or list(CLOUD_REFERENCE_MODES)
    for raw_mode in raw_modes:
        if not isinstance(raw_mode, dict):
            raise ValueError("cloud_reference.modes must contain objects")
        label = str(raw_mode.get("label", ""))
        if label not in {"min", "max"}:
            raise ValueError("cloud reference mode labels must be min or max")
        expected_effort = "low" if label == "min" else "max"
        effort = str(raw_mode.get("effort", expected_effort))
        if effort != expected_effort:
            raise ValueError(f"cloud reference {label} must use reasoning effort {expected_effort}")
        modes.append({"label": label, "effort": effort})
    if [(mode["label"], mode["effort"]) for mode in modes] != [
        ("min", "low"),
        ("max", "max"),
    ]:
        raise ValueError("cloud reference must measure exactly min=low and max=max")
    base_max_tokens = int(configured.get("max_tokens", profile.get("num_predict", 768)))
    return {
        "enabled": enabled,
        "only": force_only,
        "backend": CLOUD_REFERENCE_BACKEND,
        "provider": CLOUD_REFERENCE_PROVIDER,
        "model": CLOUD_REFERENCE_MODEL,
        "reference_only": True,
        "ranking_eligible": False,
        "modes": modes,
        "max_tokens": base_max_tokens,
        "max_tokens_max": int(configured.get("max_tokens_max", max(8192, base_max_tokens))),
        "request_timeout_seconds": float(configured.get("request_timeout_seconds", 180)),
        "empty_content_rescue": dict(
            configured.get(
                "empty_content_rescue",
                {"enabled": True, "multiplier": 2, "max_multiplier": 8},
            )
        ),
        "max_retries": int(configured.get("max_retries", profile.get("max_retries", 2))),
        "self_judge": bool(configured.get("self_judge", True)),
        "independent_audit": bool(configured.get("independent_audit", True)),
    }


def build_cloud_target_payload(
    item: BenchmarkItem, *, effort: str, max_tokens: int
) -> dict[str, Any]:
    """Build the exact OpenRouter target request without Ollama-only controls."""
    return {
        "model": CLOUD_REFERENCE_MODEL,
        "messages": [{"role": "user", "content": render_item_prompt(item)}],
        "reasoning": {"effort": effort},
        "max_tokens": max_tokens,
        "provider": {
            "only": ["z-ai"],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
        "usage": {"include": True},
    }


def _write_target_log(
    store: RunStore,
    key: str,
    attempt: int,
    mode: str,
    request: dict[str, Any],
    response: dict[str, Any] | None,
    error: str | None,
) -> str:
    log_dir = store.logs_dir / "cloud_reference"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{safe_filename(key)}-attempt-{attempt}-{mode}.json"
    value: dict[str, Any] = {"request": safe_request_subset(request)}
    if response is not None:
        value["response"] = safe_response_subset(response)
    if error:
        value["error"] = error[:1000]
    atomic_write_text(path, json_dumps(value) + "\n")
    return str(path.relative_to(store.root))


def run_cloud_reference(
    client: Any | None,
    store: RunStore,
    items: list[BenchmarkItem],
    config: dict[str, Any],
    *,
    max_retries: int | None = None,
    key_path: Path | None = None,
) -> dict[str, int]:
    """Generate the GLM reference at low/max and append crash-safe records."""
    if not config.get("enabled"):
        return {"planned": 0, "skipped": 0, "ok": 0, "error": 0}
    runner = client or OpenRouterClient(
        timeout=float(config.get("request_timeout_seconds", 180)), key_path=key_path
    )
    retry_limit = int(config.get("max_retries", 2) if max_retries is None else max_retries)
    modes = list(config["modes"])
    rescue_settings = config.get("empty_content_rescue")
    rescue_settings = dict(rescue_settings) if isinstance(rescue_settings, dict) else {}
    rescue_enabled = bool(rescue_settings.get("enabled", True))
    rescue_multiplier = max(2, int(rescue_settings.get("multiplier", 2)))
    rescue_max_multiplier = max(
        rescue_multiplier, int(rescue_settings.get("max_multiplier", 4))
    )
    finished = store.completed_keys("generations.jsonl")
    counts = {"planned": len(modes) * len(items), "skipped": 0, "ok": 0, "error": 0}
    for mode in modes:
        label = mode["label"]
        effort = mode["effort"]
        for item in items:
            key = cloud_generation_key(item, label)
            if key in finished:
                counts["skipped"] += 1
                continue
            base_max_tokens = int(
                config["max_tokens_max"] if label == "max" else config["max_tokens"]
            )
            max_num_tokens = base_max_tokens * rescue_max_multiplier if rescue_enabled else base_max_tokens
            rescue = {
                "enabled": rescue_enabled,
                "trigger": "empty_content",
                "base_max_tokens": base_max_tokens,
                "multiplier": rescue_multiplier,
                "max_multiplier": rescue_max_multiplier,
                "max_num_tokens": max_num_tokens,
                "used": False,
                "exhausted": False,
                "rescue_request_count": 0,
                "response_summaries": [],
            }
            initial_payload = build_cloud_target_payload(
                item, effort=effort, max_tokens=base_max_tokens
            )
            prompt = initial_payload["messages"][0]["content"]
            record: dict[str, Any] = {
                "key": key,
                "model": CLOUD_REFERENCE_MODEL,
                "mode": label,
                "thinking_effort": effort,
                "thinking_kind": "levels",
                "benchmark": item.benchmark,
                "category": item.category,
                "item_id": item.item_id,
                "item": item.to_dict(),
                "backend": CLOUD_REFERENCE_BACKEND,
                "provider": CLOUD_REFERENCE_PROVIDER,
                "reference_only": True,
                "ranking_eligible": False,
                "target_role": "cloud_reference",
                "api_request": initial_payload,
                "request_safe": safe_request_subset(
                    initial_payload, prompt_sha256=sha256_bytes(prompt.encode("utf-8"))
                ),
                "request_history": [],
                "started_at": utc_now(),
            }
            attempts = 0
            errors: list[str] = []
            retry_history: list[dict[str, Any]] = []
            attempt_usages: list[dict[str, Any]] = []
            attempt_costs: list[float | None] = []
            attempt_reasoning: list[int | None] = []
            latency_total = 0.0
            current_max_tokens = base_max_tokens
            success = False
            stop = False
            while not stop:
                payload = build_cloud_target_payload(
                    item, effort=effort, max_tokens=current_max_tokens
                )
                record["request_history"].append(safe_request_subset(payload))
                budget_attempts = 0
                while budget_attempts <= retry_limit:
                    attempts += 1
                    budget_attempts += 1
                    started = time.perf_counter()
                    try:
                        response = runner.chat(payload)
                        elapsed = time.perf_counter() - started
                        latency_total += elapsed
                        content, thinking = message_parts(response)
                        usage = extract_usage(response)
                        cost = extract_cost_usd(usage)
                        reasoning_tokens = extract_reasoning_tokens(usage)
                        attempt_usages.append(usage)
                        attempt_costs.append(cost)
                        attempt_reasoning.append(reasoning_tokens)
                        response_safe = safe_response_subset(response)
                        log_path = _write_target_log(
                            store, key, attempts, label, payload, response, None
                        )
                        if not content.strip():
                            error = "OpenRouter returned an empty target content field"
                            errors.append(f"attempt {attempts}: {error}")
                            rescue["response_summaries"].append(
                                {
                                    "attempt": attempts,
                                    "max_tokens": current_max_tokens,
                                    "content_length": len(content),
                                    "thinking_length": len(thinking),
                                    "provider_actual": actual_provider(response),
                                    "model_actual": actual_model(response),
                                    "usage": usage,
                                    "cost_usd": cost,
                                    "reasoning_tokens": reasoning_tokens,
                                    "latency_s": elapsed,
                                    "response_path": log_path,
                                }
                            )
                            retry_history.append(
                                {
                                    "attempt": attempts,
                                    "kind": "empty_content",
                                    "max_tokens": current_max_tokens,
                                    "latency_s": elapsed,
                                }
                            )
                            # Empty content means the reasoning budget was consumed before
                            # final text. Retrying the same budget wastes time and money;
                            # increase the budget immediately. Transport failures below still
                            # use the finite same-budget retry policy.
                            next_max_tokens = current_max_tokens * rescue_multiplier
                            if not rescue_enabled or next_max_tokens > max_num_tokens:
                                rescue["exhausted"] = True
                                stop = True
                                break
                            rescue["used"] = True
                            rescue["rescue_request_count"] += 1
                            current_max_tokens = next_max_tokens
                            break
                        record.update(
                            {
                                "status": "ok",
                                "attempts": attempts,
                                "retry_history": retry_history,
                                "content": content,
                                "thinking": thinking,
                                "content_length": len(content),
                                "thinking_length": len(thinking),
                                "response": response_safe,
                                "response_path": log_path,
                                "provider_actual": actual_provider(response),
                                "model_actual": actual_model(response),
                                "usage": usage,
                                "attempt_usage": attempt_usages,
                                "reasoning_tokens": _total_int(attempt_reasoning),
                                "cost_usd": _total_float(attempt_costs),
                                "latency_s": latency_total,
                                "final_max_tokens": current_max_tokens,
                                "empty_content_rescue": rescue,
                                "finished_at": utc_now(),
                            }
                        )
                        if item.benchmark == "objective-ja":
                            score, prediction = score_multiple_choice(content, item)
                            record["objective_score"] = score
                            record["prediction"] = prediction
                        counts["ok"] += 1
                        success = True
                        stop = True
                        break
                    except Exception as exc:
                        elapsed = time.perf_counter() - started
                        latency_total += elapsed
                        error = str(exc)[:1000]
                        errors.append(f"attempt {attempts}: {error}")
                        retry_history.append(
                            {
                                "attempt": attempts,
                                "kind": "transport_or_response",
                                "max_tokens": current_max_tokens,
                                "error": error,
                                "latency_s": elapsed,
                            }
                        )
                        _write_target_log(store, key, attempts, label, payload, None, error)
                        if budget_attempts <= retry_limit:
                            time.sleep(min(2.0, 0.25 * budget_attempts))
                            continue
                        stop = True
                        break
            if not success:
                record.update(
                    {
                        "status": "error",
                        "attempts": attempts,
                        "retry_history": retry_history,
                        "attempt_usage": attempt_usages,
                        "reasoning_tokens": _total_int(attempt_reasoning),
                        "cost_usd": _total_float(attempt_costs),
                        "latency_s": latency_total or None,
                        "final_max_tokens": current_max_tokens,
                        "empty_content_rescue": rescue,
                        "error": "; ".join(errors)[-4000:] or "cloud target failed",
                        "finished_at": utc_now(),
                    }
                )
                counts["error"] += 1
            store.append("generations.jsonl", record)
            finished.add(key)
    return counts
