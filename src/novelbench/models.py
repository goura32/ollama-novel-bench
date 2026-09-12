"""Runtime model discovery and thinking-control probes."""

from __future__ import annotations

from typing import Any

from .ollama import OllamaClient

_GENERATION_CAPABILITIES = {"completion", "chat", "generate"}
_LEVEL_ORDER = {"low": 1, "medium": 2, "high": 3, "xhigh": 4, "max": 5}


def filter_generation_models(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude embedding-only entries without relying on model-name heuristics."""
    result = []
    for model in models:
        capabilities = {str(value).lower() for value in model.get("capabilities", [])}
        if capabilities and capabilities <= {"embedding"}:
            continue
        if capabilities and not (capabilities & _GENERATION_CAPABILITIES):
            # Some older servers omit completion but expose a useful model entry.
            # Thinking is still a generation capability; a bare embedding entry is not.
            if "thinking" not in capabilities:
                continue
        result.append(model)
    return result


def _message_parts(response: dict[str, Any]) -> tuple[str, str]:
    message = response.get("message")
    if not isinstance(message, dict):
        message = {}
    content = message.get("content", response.get("response", ""))
    thinking = message.get("thinking", response.get("thinking", ""))
    return str(content or ""), str(thinking or "")


def _probe_payload(model_name: str, field: str | None, value: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "短い確認です。日本語で一語だけ返してください。"}],
        "stream": False,
        "options": {"num_predict": 32},
    }
    if field is not None:
        payload[field] = value
    return payload


def _run_probe(
    client: Any, model_name: str, field: str | None, value: Any, label: str
) -> dict[str, Any]:
    payload = _probe_payload(model_name, field, value)
    result: dict[str, Any] = {
        "label": label,
        "field": field,
        "value": value,
        "request": payload,
        "accepted": False,
        "thinking_present": False,
        "thinking_length": 0,
        "content_length": 0,
    }
    try:
        response = client.chat(payload)
        content, thinking = _message_parts(response)
        result.update(
            {
                "accepted": True,
                "thinking_present": bool(thinking),
                "thinking_length": len(thinking),
                "content_length": len(content),
                "done": response.get("done"),
            }
        )
    except Exception as exc:  # probes must not prevent the next control candidate
        result["error"] = str(exc)[:500]
    return result


def detect_thinking_modes(client: Any, model: dict[str, Any]) -> dict[str, Any]:
    capabilities = {str(value).lower() for value in model.get("capabilities", [])}
    if "thinking" not in capabilities:
        return {
            "kind": "none",
            "modes": [{"label": "none", "field": None, "value": None}],
            "probe_results": [],
        }

    model_name = str(model["name"])
    probes: list[dict[str, Any]] = []
    probes.append(_run_probe(client, model_name, "think", False, "off"))
    probes.append(_run_probe(client, model_name, "think", True, "on"))

    # Try the documented level values in descending order.  We retain every
    # response in the manifest rather than assuming a family from its name.
    for level in ("max", "xhigh", "high", "medium", "low", "none", "off"):
        probes.append(_run_probe(client, model_name, "think", level, level))

    level_results = [
        probe
        for probe in probes
        if probe["field"] == "think"
        and probe["value"] in _LEVEL_ORDER
        and probe["accepted"]
        and probe["thinking_present"]
    ]

    # A few Ollama-compatible servers expose the OpenAI-style name instead.
    if not level_results:
        for level in ("max", "xhigh", "high", "medium", "low", "none", "off"):
            probes.append(
                _run_probe(
                    client, model_name, "reasoning_effort", level, f"reasoning_effort:{level}"
                )
            )
        level_results = [
            probe
            for probe in probes
            if probe["field"] == "reasoning_effort"
            and probe["value"] in _LEVEL_ORDER
            and probe["accepted"]
            and probe["thinking_present"]
        ]

    off_candidates = [
        probe
        for probe in probes
        if probe["accepted"]
        and probe["field"] in {"think", "reasoning_effort"}
        and probe["value"] in (False, "none", "off")
        and not probe["thinking_present"]
    ]
    on_candidates = [
        probe
        for probe in probes
        if probe["accepted"] and probe["thinking_present"] and probe["value"] is True
    ]

    if level_results:
        highest = max(level_results, key=lambda probe: _LEVEL_ORDER[str(probe["value"])])
        minimum = off_candidates[0] if off_candidates else probes[0]
        modes = [
            {
                "label": "min",
                "field": minimum["field"],
                "value": minimum["value"],
                "probe_label": minimum["label"],
            },
            {
                "label": "max",
                "field": highest["field"],
                "value": highest["value"],
                "probe_label": highest["label"],
            },
        ]
        kind = "levels"
    elif on_candidates or any(
        probe["accepted"] and probe["field"] == "think" and probe["value"] is True
        for probe in probes
    ):
        modes = [
            {"label": "min", "field": "think", "value": False, "probe_label": "off"},
            {"label": "max", "field": "think", "value": True, "probe_label": "on"},
        ]
        kind = "binary" if off_candidates else "always_on"
    else:
        modes = [
            {"label": "min", "field": "think", "value": False, "probe_label": "off"},
            {"label": "max", "field": "think", "value": True, "probe_label": "on"},
        ]
        kind = "unknown"

    return {
        "kind": kind,
        "modes": modes,
        "min": modes[0],
        "max": modes[-1],
        "accepted_levels": sorted(
            {str(probe["value"]) for probe in level_results},
            key=lambda value: _LEVEL_ORDER.get(value, 0),
        ),
        "probe_results": probes,
    }


def _model_record(tag: dict[str, Any], shown: dict[str, Any]) -> dict[str, Any]:
    details = dict(tag.get("details") or {})
    details.update(shown.get("details") or {})
    tag_capabilities = list(tag.get("capabilities") or [])
    shown_capabilities = list(shown.get("capabilities") or [])
    capabilities = list(dict.fromkeys(tag_capabilities + shown_capabilities))
    record: dict[str, Any] = {
        "name": tag.get("name") or tag.get("model"),
        "digest": tag.get("digest"),
        "size": tag.get("size"),
        "modified_at": tag.get("modified_at"),
        "capabilities": capabilities,
        "details": details,
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "context_length": details.get("context_length"),
    }
    # Keep useful /api/show metadata but never store the potentially huge prompt
    # template, system prompt, license text, or modelfile in a run manifest.
    if shown.get("model_info"):
        record["model_info"] = shown["model_info"]
    return record


def _capability_only_thinking(record: dict[str, Any]) -> dict[str, Any]:
    if "thinking" not in {str(value).lower() for value in record.get("capabilities", [])}:
        return {
            "kind": "none",
            "modes": [{"label": "none", "field": None, "value": None}],
            "probe_results": [],
        }
    return {
        "kind": "capability_only",
        "modes": [
            {"label": "min", "field": "think", "value": False},
            {"label": "max", "field": "think", "value": True},
        ],
        "probe_results": [],
    }


def probe_model_profiles(client: Any, profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Probe only already-selected models and preserve the raw probe audit."""
    result: list[dict[str, Any]] = []
    for profile in profiles:
        record = dict(profile)
        capabilities = {str(value).lower() for value in record.get("capabilities", [])}
        record["thinking"] = (
            detect_thinking_modes(client, record)
            if "thinking" in capabilities
            else _capability_only_thinking(record)
        )
        result.append(record)
    return result


def discover_models(client: OllamaClient, *, probe: bool = True) -> list[dict[str, Any]]:
    tags = client.tags()
    entries = filter_generation_models(list(tags.get("models") or []))
    profiles: list[dict[str, Any]] = []
    for tag in sorted(entries, key=lambda value: str(value.get("name") or value.get("model"))):
        name = str(tag.get("name") or tag.get("model"))
        shown = client.show(name)
        record = _model_record(tag, shown)
        record["thinking"] = _capability_only_thinking(record)
        profiles.append(record)
    return probe_model_profiles(client, profiles) if probe else profiles


def select_models(profiles: list[dict[str, Any]], requested: str | None) -> list[dict[str, Any]]:
    if not requested:
        return profiles
    names = [name.strip() for name in requested.split(",") if name.strip()]
    by_name = {str(profile["name"]): profile for profile in profiles}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(
            f"requested model(s) are not generation-capable or not installed: {', '.join(unknown)}"
        )
    return [by_name[name] for name in names]
