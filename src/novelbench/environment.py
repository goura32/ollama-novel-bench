"""Non-secret host and runtime metadata for manifests."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Any

from .util import run_command

_SAFE_ENV_KEYS = (
    "OLLAMA_HOST",
    "OLLAMA_MAX_LOADED_MODELS",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_FLASH_ATTENTION",
    "OLLAMA_KV_CACHE_TYPE",
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "CONTAINER_IMAGE",
    "HOSTNAME",
)


def _gpu_info() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "reason": "nvidia-smi not found"}
    code, stdout, stderr = run_command(
        [command, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        timeout=10,
    )
    return {
        "available": code == 0,
        "command": "nvidia-smi",
        "rows": [line.strip() for line in stdout.splitlines() if line.strip()],
        "error": stderr[-500:] if code else None,
    }


def collect_environment(repo_root: Path, *, ollama_version: str | None = None) -> dict[str, Any]:
    code, git_commit, _ = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root, timeout=10)
    return {
        "host": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "hostname": platform.node(),
        },
        "gpu": _gpu_info(),
        "ollama": {
            "base_url": os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            "version": ollama_version,
        },
        "container_env": {key: os.environ[key] for key in _SAFE_ENV_KEYS if os.environ.get(key)},
        "git_commit": git_commit if code == 0 else None,
    }
