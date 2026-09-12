"""Crash-tolerant files for a benchmark run."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .util import append_jsonl, atomic_write_text, json_dumps


class RunStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.data_dir = self.root / "data"
        self.logs_dir = self.root / "logs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def append(self, filename: str, record: dict[str, Any]) -> None:
        append_jsonl(self.data_dir / filename, record)

    def read_jsonl(self, filename: str) -> Iterator[dict[str, Any]]:
        path = self.data_dir / filename
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    # A process can die between write(2) and fsync(2).  Complete
                    # records are never rewritten; an incomplete tail is safe to skip.
                    continue
                if isinstance(value, dict):
                    yield value

    def completed_keys(self, filename: str) -> set[str]:
        return {str(record["key"]) for record in self.read_jsonl(filename) if record.get("key")}

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_text(self.manifest_path, json_dumps(manifest) + "\n")

    def load_manifest(self) -> dict[str, Any]:
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"manifest is not an object: {self.manifest_path}")
        return value

    def write_items(self, items: list[dict[str, Any]]) -> None:
        path = self.data_dir / "items.jsonl"
        if path.exists() and any(True for _ in self.read_jsonl("items.jsonl")):
            return
        for item in items:
            self.append("items.jsonl", item)

    def load_items(self) -> list[dict[str, Any]]:
        return list(self.read_jsonl("items.jsonl"))
