import json
from pathlib import Path

from novelbench.store import RunStore


def test_jsonl_store_skips_completed_keys_and_ignores_partial_tail(tmp_path: Path):
    store = RunStore(tmp_path / "run")
    store.append("generations.jsonl", {"key": "done", "status": "ok"})
    path = tmp_path / "run" / "data" / "generations.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"key":"broken"')

    records = list(store.read_jsonl("generations.jsonl"))
    assert records == [{"key": "done", "status": "ok"}]
    assert store.completed_keys("generations.jsonl") == {"done"}


def test_manifest_updates_atomically(tmp_path: Path):
    store = RunStore(tmp_path / "run")
    manifest = {"run_id": "r1", "status": "running"}
    store.write_manifest(manifest)
    manifest["status"] = "complete"
    store.write_manifest(manifest)

    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["status"] == "complete"
