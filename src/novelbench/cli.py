"""Command line interface for novelbench."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .cloud import cloud_reference_config, run_cloud_reference
from .datasets import BenchmarkItem, load_profile_items
from .environment import collect_environment
from .generation import run_generations
from .judge import (
    JUDGE_BACKEND,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    JUDGE_REASONING,
    LUNA_AUDIT_BACKEND,
    LUNA_AUDIT_MODEL,
    LUNA_AUDIT_PROVIDER,
    LUNA_AUDIT_REASONING,
    judge_run,
    luna_audit_run,
)
from .models import discover_models, probe_model_profiles, select_models
from .ollama import OllamaClient
from .reporting import build_report
from .store import RunStore
from .util import json_dumps, run_command, safe_run_id, utc_now

PROFILE_PATH = Path("configs/profiles.json")
DEFAULT_CACHE = Path.home() / ".cache" / "ollama-novel-bench" / "datasets"


def repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [Path.cwd(), *current.parents]:
        if (candidate / "pyproject.toml").exists() and (
            candidate / "configs" / "profiles.json"
        ).exists():
            return candidate
    raise RuntimeError("could not locate repository root")


def load_profiles(root: Path) -> dict[str, dict[str, Any]]:
    value = json.loads((root / PROFILE_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configs/profiles.json must contain an object")
    return {str(name): dict(config) for name, config in value.items()}


def base_url(value: str | None) -> str:
    selected = value or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434"
    if "://" not in selected:
        selected = f"http://{selected}"
    return selected.rstrip("/")


def run_dir(root: Path, run_id: str) -> Path:
    return root / "results" / safe_run_id(run_id)


def select_run_models(
    profiles: list[dict[str, Any]], requested: str | None, profile_name: str
) -> list[dict[str, Any]]:
    """Apply profile policy before expensive thinking probes are run."""
    selected = select_models(profiles, requested)
    if profile_name == "smoke":
        if requested and len(selected) != 1:
            raise ValueError("smoke profile requires exactly one explicit model")
        selected = selected[:1]
    return selected


def _manifest_base(
    root: Path,
    client: OllamaClient,
    run_id: str,
    profile_name: str,
    profile: dict[str, Any],
    models: list[dict[str, Any]],
    items: list[BenchmarkItem],
    sources: list[dict[str, Any]],
    cloud_reference: dict[str, Any],
    ollama_version: str | None,
) -> dict[str, Any]:
    version = ollama_version
    environment = collect_environment(root, ollama_version=str(version) if version else None)
    return {
        "schema_version": "0.1",
        "run_id": run_id,
        "created_at": utc_now(),
        "profile": profile_name,
        "status": "running",
        "smoke_warning": "予備試験・ランキング用途不可" if profile_name == "smoke" else None,
        "seed": profile.get("seed"),
        "config": profile,
        "execution": {
            "target_order": "model -> thinking mode -> item",
            "strictly_serial": True,
            "resume_policy": "any terminal generation record is not rerun",
            "ollama_max_loaded_models_assumption": 1,
        },
        "generation_policy": {
            "context_policy": "model_native_max",
            "num_predict": profile.get("num_predict"),
            "num_predict_policy": (
                "profile_fixed"
                if profile.get("num_predict") is not None
                else "unset_until_native_context_exhaustion"
            ),
            "transport_retry_count": int(profile.get("max_retries", 2)),
        },
        "ollama": {"base_url": client.base_url, "version": version},
        "environment": environment,
        "models": models,
        "dataset_sources": sources,
        "item_count": len(items),
        "item_ids": [item.item_id for item in items],
        "items_file": "data/items.jsonl",
        "judge": {
            "backend": JUDGE_BACKEND,
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "reasoning": JUDGE_REASONING,
            "endpoint": "https://openrouter.ai/api/v1/chat/completions",
            "routing": {"require_parameters": True, "sort": "price"},
        },
        "independent_audit": {
            "backend": LUNA_AUDIT_BACKEND,
            "provider": LUNA_AUDIT_PROVIDER,
            "model": LUNA_AUDIT_MODEL,
            "reasoning": LUNA_AUDIT_REASONING,
            "role": "cloud_reference_creative_only",
        },
        "cloud_reference": cloud_reference,
        "package_version": __version__,
    }


def _update_progress(
    store: RunStore,
    manifest: dict[str, Any],
    generation_counts: dict[str, Any],
    judge_counts: dict[str, Any] | None = None,
) -> None:
    manifest["progress"] = {"generation": generation_counts, "judge": judge_counts}
    store.write_manifest(manifest)


def _execute_run(
    root: Path,
    *,
    run_id: str,
    profile_name: str,
    requested_models: str | None,
    base: str,
    cache_dir: Path,
    resume: bool = False,
    force_cloud_reference: bool = False,
    disable_cloud_reference: bool = False,
    cloud_reference_only: bool = False,
) -> dict[str, Any]:
    profiles = load_profiles(root)
    if profile_name not in profiles:
        raise ValueError(f"unknown profile: {profile_name}")
    profile = profiles[profile_name]
    target_dir = run_dir(root, run_id)
    if resume:
        store = RunStore(target_dir)
        manifest = store.load_manifest()
        profile = dict(manifest.get("config") or profile)
        profile_name = str(manifest.get("profile", profile_name))
        models = list(manifest.get("models") or [])
        item_values = store.load_items()
        cloud_reference = dict(manifest.get("cloud_reference") or {})
        if not cloud_reference:
            cloud_reference = cloud_reference_config(profile, profile_name)
        if cloud_reference_only:
            raise ValueError("--cloud-reference-only is available only for a new run")
        if force_cloud_reference:
            cloud_reference["enabled"] = True
        if disable_cloud_reference:
            cloud_reference["enabled"] = False
        if (not models and not cloud_reference.get("enabled")) or not item_values:
            raise ValueError("run cannot resume: no local/cloud target or data/items.jsonl is missing")
        items = [BenchmarkItem.from_dict(value) for value in item_values]
        manifest["cloud_reference"] = cloud_reference
        manifest["status"] = "running"
        store.write_manifest(manifest)
        version = (manifest.get("ollama") or {}).get("version")
    else:
        if (target_dir / "manifest.json").exists():
            raise ValueError(f"run already exists: {run_id}; use novelbench resume {run_id}")
        target_dir.mkdir(parents=True, exist_ok=True)
        cloud_reference = cloud_reference_config(
            profile,
            profile_name,
            force_enable=force_cloud_reference,
            disable=disable_cloud_reference,
            force_only=cloud_reference_only,
        )
        if cloud_reference_only and requested_models:
            raise ValueError("--cloud-reference-only cannot be combined with --models")
        if cloud_reference_only and disable_cloud_reference:
            raise ValueError("--cloud-reference-only cannot be combined with --no-cloud-reference")
        client = OllamaClient(base, timeout=600)
        if cloud_reference_only:
            models = []
            version = None
        else:
            all_models = discover_models(client, probe=False)
            models = select_run_models(all_models, requested_models, profile_name)
            if not models:
                raise ValueError("Ollama returned no generation-capable models")
            models = probe_model_profiles(client, models)
            version_value = client.version().get("version")
            version = str(version_value) if version_value else None
        items, sources = load_profile_items(root, cache_dir, profile_name, profile)
        store = RunStore(target_dir)
        manifest = _manifest_base(
            root,
            client,
            run_id,
            profile_name,
            profile,
            models,
            items,
            sources,
            cloud_reference,
            version,
        )
        store.write_manifest(manifest)
        store.write_items([item.to_dict() for item in items])

    client = OllamaClient(base, timeout=600)
    local_counts = run_generations(client, store, models, items, profile)
    cloud_counts = run_cloud_reference(None, store, items, cloud_reference)
    generation_counts = {
        field: local_counts[field] + cloud_counts[field]
        for field in ("planned", "skipped", "ok", "error")
    }
    generation_counts["local"] = local_counts
    generation_counts["cloud_reference"] = cloud_counts
    manifest["cloud_reference"] = cloud_reference
    manifest["status"] = "generation_complete"
    _update_progress(store, manifest, generation_counts)
    judge_counts = None
    if profile.get("judge", True):
        judge_counts = judge_run(
            root,
            store,
            items,
            only_missing=True,
            timeout=float(profile.get("judge_timeout", 180)),
            max_retries=int(profile.get("max_retries", 2)),
        )
        audit_counts = None
        if cloud_reference.get("enabled") and cloud_reference.get("independent_audit", True):
            audit_counts = luna_audit_run(
                root,
                store,
                items,
                only_missing=True,
                timeout=float(profile.get("judge_timeout", 300)),
                max_retries=int(profile.get("max_retries", 2)),
            )
            judge_counts["independent_audit"] = audit_counts
    audit_error = bool(judge_counts and judge_counts.get("independent_audit", {}).get("error"))
    manifest["status"] = (
        "complete_with_errors"
        if generation_counts["error"] or (judge_counts and judge_counts["error"]) or audit_error
        else "complete"
    )
    _update_progress(store, manifest, generation_counts, judge_counts)
    report = build_report(target_dir)
    return {
        "run_id": run_id,
        "profile": profile_name,
        "generation": generation_counts,
        "judge": judge_counts,
        "report": report,
    }


def _git_publish(root: Path, run_id: str) -> dict[str, Any]:
    target_dir = run_dir(root, run_id)
    report = build_report(target_dir)
    prefix = f"results/{run_id}/"
    code, status_text, status_err = run_command(
        ["git", "status", "--porcelain"], cwd=root, timeout=20
    )
    if code != 0:
        raise RuntimeError(f"git status failed: {status_err or status_text}")
    dirty_paths = [line[3:] for line in status_text.splitlines() if len(line) >= 4]
    outside = [
        path for path in dirty_paths if path != f"results/{run_id}" and not path.startswith(prefix)
    ]
    status_path = target_dir / "publish.json"
    if outside:
        payload = {
            "state": "blocked_dirty_tree",
            "run_id": run_id,
            "outside_run_changes": outside,
            "recorded_at": utc_now(),
        }
        status_path.write_text(json_dumps(payload) + "\n", encoding="utf-8")
        raise RuntimeError(f"publish blocked by dirty files outside {prefix}: {', '.join(outside)}")

    status_path.write_text(
        json_dumps(
            {"state": "staging", "run_id": run_id, "report": report, "recorded_at": utc_now()}
        )
        + "\n",
        encoding="utf-8",
    )
    code, _, stderr = run_command(["git", "add", "--", prefix], cwd=root, timeout=20)
    if code != 0:
        raise RuntimeError(f"git add failed: {stderr}")
    code, stdout, stderr = run_command(
        ["git", "commit", "-m", f"results: add {run_id}"], cwd=root, timeout=60
    )
    if code != 0:
        raise RuntimeError(f"git commit failed: {stderr or stdout}")
    code, stdout, stderr = run_command(["git", "push", "origin", "main"], cwd=root, timeout=180)
    if code != 0:
        status_path.write_text(
            json_dumps(
                {
                    "state": "push_failed",
                    "run_id": run_id,
                    "error": stderr or stdout,
                    "recorded_at": utc_now(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"git push failed: {stderr or stdout}")
    code, local_sha, _ = run_command(["git", "rev-parse", "HEAD"], cwd=root, timeout=20)
    code_remote, remote_text, remote_err = run_command(
        ["git", "ls-remote", "origin", "refs/heads/main"], cwd=root, timeout=30
    )
    remote_sha = remote_text.split()[0] if remote_text else ""
    if code != 0 or code_remote != 0 or local_sha != remote_sha:
        status_path.write_text(
            json_dumps(
                {
                    "state": "push_unverified",
                    "run_id": run_id,
                    "local_sha": local_sha,
                    "remote_sha": remote_sha,
                    "error": remote_err,
                    "recorded_at": utc_now(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"push verification failed: local={local_sha} remote={remote_sha}")
    status_path.write_text(
        json_dumps(
            {
                "state": "pushed_verified",
                "run_id": run_id,
                "local_sha": local_sha,
                "remote_sha": remote_sha,
                "recorded_at": utc_now(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    code, _, stderr = run_command(
        ["git", "add", "--", f"results/{run_id}/publish.json"], cwd=root, timeout=20
    )
    if code != 0:
        raise RuntimeError(f"git add publish status failed: {stderr}")
    code, stdout, stderr = run_command(
        ["git", "commit", "-m", f"chore: record verified publication {run_id}"],
        cwd=root,
        timeout=60,
    )
    if code != 0:
        raise RuntimeError(f"git commit publish status failed: {stderr or stdout}")
    code, stdout, stderr = run_command(["git", "push", "origin", "main"], cwd=root, timeout=180)
    if code != 0:
        raise RuntimeError(f"git push publish status failed: {stderr or stdout}")
    code_remote, remote_text, remote_err = run_command(
        ["git", "ls-remote", "origin", "refs/heads/main"], cwd=root, timeout=30
    )
    final_remote = remote_text.split()[0] if remote_text else ""
    final_local = run_command(["git", "rev-parse", "HEAD"], cwd=root, timeout=20)[1]
    if code_remote != 0 or final_local != final_remote:
        raise RuntimeError(
            f"final push verification failed: local={final_local} remote={final_remote} ({remote_err})"
        )
    return {
        "run_id": run_id,
        "state": "pushed_verified",
        "commit": final_local,
        "remote": final_remote,
    }


def _cmd_models(args: argparse.Namespace, root: Path) -> int:
    client = OllamaClient(base_url(args.base_url), timeout=60)
    profiles = discover_models(client, probe=bool(args.probe_thinking))
    print(json.dumps({"ollama": client.base_url, "models": profiles}, ensure_ascii=False, indent=2))
    return 0


def _cmd_run(args: argparse.Namespace, root: Path, resume: bool = False) -> int:
    result = _execute_run(
        root,
        run_id=args.run_id,
        profile_name=(args.profile if not resume else "smoke"),
        requested_models=getattr(args, "models", None),
        base=base_url(args.base_url),
        cache_dir=Path(args.cache_dir).expanduser(),
        resume=resume,
        force_cloud_reference=bool(getattr(args, "cloud_reference", False)),
        disable_cloud_reference=bool(getattr(args, "no_cloud_reference", False)),
        cloud_reference_only=bool(getattr(args, "cloud_reference_only", False)),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.publish:
        print(json.dumps(_git_publish(root, args.run_id), ensure_ascii=False, indent=2))
    return 0


def _cmd_resume(args: argparse.Namespace, root: Path) -> int:
    return _cmd_run(args, root, resume=True)


def _cmd_judge(args: argparse.Namespace, root: Path) -> int:
    target = run_dir(root, args.run_id)
    store = RunStore(target)
    manifest = store.load_manifest()
    items = [BenchmarkItem.from_dict(value) for value in store.load_items()]
    counts = judge_run(
        root,
        store,
        items,
        only_missing=False,
        timeout=float(manifest.get("config", {}).get("judge_timeout", 180)),
        max_retries=int(manifest.get("config", {}).get("max_retries", 2)),
    )
    manifest["last_judge"] = {
        "at": utc_now(),
        "counts": counts,
        "fixed_configuration": {
            "backend": JUDGE_BACKEND,
            "provider": JUDGE_PROVIDER,
            "model": JUDGE_MODEL,
            "reasoning": JUDGE_REASONING,
        },
    }
    store.write_manifest(manifest)
    report = build_report(target)
    print(
        json.dumps(
            {"run_id": args.run_id, "judge": counts, "report": report}, ensure_ascii=False, indent=2
        )
    )
    return 0


def _cmd_audit_luna(args: argparse.Namespace, root: Path) -> int:
    target = run_dir(root, args.run_id)
    store = RunStore(target)
    manifest = store.load_manifest()
    items = [BenchmarkItem.from_dict(value) for value in store.load_items()]
    counts = luna_audit_run(
        root,
        store,
        items,
        only_missing=not bool(getattr(args, "force", False)),
        timeout=float(manifest.get("config", {}).get("judge_timeout", 300)),
        max_retries=int(manifest.get("config", {}).get("max_retries", 2)),
    )
    manifest["last_independent_audit"] = {
        "at": utc_now(),
        "counts": counts,
        "fixed_configuration": {
            "backend": LUNA_AUDIT_BACKEND,
            "provider": LUNA_AUDIT_PROVIDER,
            "model": LUNA_AUDIT_MODEL,
            "reasoning": LUNA_AUDIT_REASONING,
        },
    }
    store.write_manifest(manifest)
    report = build_report(target)
    print(
        json.dumps(
            {"run_id": args.run_id, "independent_audit": counts, "report": report},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _cmd_report(args: argparse.Namespace, root: Path) -> int:
    result = build_report(run_dir(root, args.run_id))
    print(json.dumps({"run_id": args.run_id, **result}, ensure_ascii=False, indent=2))
    return 0


def _cmd_env(args: argparse.Namespace, root: Path) -> int:
    client = OllamaClient(base_url(args.base_url), timeout=20)
    version = client.version().get("version")
    value = collect_environment(root, ollama_version=str(version) if version else None)
    value["ollama_tags_count"] = len(client.tags().get("models", []))
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def _cmd_shortlist(args: argparse.Namespace, root: Path) -> int:
    target = run_dir(root, args.run_id)
    build_report(target)
    ranking_path = target / "data" / "ranking.csv"
    with ranking_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows[: args.top]:
        selected.append(
            {
                "rank": int(row["rank"]),
                "model": row["model"],
                "mode": row["mode"],
                "novel_primary_score": row["novel_primary_score"] or None,
            }
        )
    payload = {
        "run_id": args.run_id,
        "top": args.top,
        "models": selected,
        "pairwise_ready": True,
        "note": "full上位候補のA/B・B/A pairwise対象。",
    }
    (target / "data" / "shortlist.json").write_text(json_dumps(payload) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novelbench", description="Reproducible Ollama Japanese novel-writing benchmark"
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    env_parser = sub.add_parser("env", help="show Ollama, host and safe environment metadata")
    env_parser.add_argument("--base-url")

    models_parser = sub.add_parser("models", help="list generation-capable Ollama models")
    models_parser.add_argument("--probe-thinking", action="store_true")
    models_parser.add_argument("--base-url")

    run_parser = sub.add_parser("run", help="generate, judge and report a new run")
    run_parser.add_argument("--profile", choices=("smoke", "quick", "full", "eqcw"), required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--models", help="comma-separated exact local Ollama names")
    run_parser.add_argument("--base-url")
    run_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    cloud_group = run_parser.add_mutually_exclusive_group()
    cloud_group.add_argument(
        "--cloud-reference",
        action="store_true",
        help="include the GLM cloud reference target (also enables it for smoke)",
    )
    cloud_group.add_argument(
        "--no-cloud-reference",
        action="store_true",
        help="disable the cloud reference target",
    )
    run_parser.add_argument(
        "--cloud-reference-only",
        action="store_true",
        help="run only the cloud GLM reference; skips Ollama discovery",
    )
    run_parser.add_argument("--publish", action="store_true")

    resume_parser = sub.add_parser("resume", help="resume pending items in a run")
    resume_parser.add_argument("run_id")
    resume_parser.add_argument("--base-url")
    resume_parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    resume_cloud_group = resume_parser.add_mutually_exclusive_group()
    resume_cloud_group.add_argument("--cloud-reference", action="store_true")
    resume_cloud_group.add_argument("--no-cloud-reference", action="store_true")
    resume_parser.add_argument(
        "--cloud-reference-only",
        action="store_true",
        help="not valid for resume; kept for command symmetry",
    )
    resume_parser.add_argument("--publish", action="store_true")

    judge_parser = sub.add_parser(
        "judge", help="rejudge creative generation data with the fixed OpenRouter GLM Judge"
    )
    judge_parser.add_argument("run_id")

    audit_parser = sub.add_parser(
        "audit-luna", help="run the independent Luna(Max) audit on cloud-reference creative data"
    )
    audit_parser.add_argument("run_id")
    audit_parser.add_argument(
        "--force", action="store_true", help="append a fresh Luna audit even when one already exists"
    )

    report_parser = sub.add_parser("report", help="rebuild CSV/JSON and charts")
    report_parser.add_argument("run_id")

    publish_parser = sub.add_parser("publish", help="report, commit and push one run to main")
    publish_parser.add_argument("run_id")

    shortlist_parser = sub.add_parser(
        "shortlist", help="select top candidates for full/pairwise work"
    )
    shortlist_parser.add_argument("run_id")
    shortlist_parser.add_argument("--top", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = repo_root()
    try:
        if args.command == "env":
            return _cmd_env(args, root)
        if args.command == "models":
            return _cmd_models(args, root)
        if args.command == "run":
            return _cmd_run(args, root)
        if args.command == "resume":
            args.run_id = args.run_id
            return _cmd_resume(args, root)
        if args.command == "judge":
            return _cmd_judge(args, root)
        if args.command == "audit-luna":
            return _cmd_audit_luna(args, root)
        if args.command == "report":
            return _cmd_report(args, root)
        if args.command == "publish":
            print(json.dumps(_git_publish(root, args.run_id), ensure_ascii=False, indent=2))
            return 0
        if args.command == "shortlist":
            return _cmd_shortlist(args, root)
    except KeyboardInterrupt:
        print("中断しました。novelbench resume <run-id> で続行できます。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"novelbench error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.command}")
    return 2
