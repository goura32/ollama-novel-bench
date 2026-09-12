"""Dataset adapters and deterministic sampling.

The large external datasets are fetched into a user cache, not redistributed in
this repository.  The selected records are copied into a run's items.jsonl so a
run remains auditable and can be resumed without re-fetching.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .util import atomic_write_text, json_dumps, sha256_bytes, utc_now

JCQ_URL = "https://huggingface.co/datasets/nlp-waseda/JCQ/resolve/7961ea89f5f1271663a7a1b62f1a74c274c86116/test.jsonl?download=true"
JMMLU_URL = "https://huggingface.co/datasets/nlp-waseda/JMMLU/resolve/main/JMMLU.zip?download=true"
JCOMMONSENSEQA_URL = "https://raw.githubusercontent.com/yahoojapan/JGLUE/refs/tags/v1.2.0/datasets/jcommonsenseqa-v1.2/valid-v1.2.json"
JAMC_ROWS_URL = "https://datasets-server.huggingface.co/rows"
EQCW_PROMPTS_URL = "https://raw.githubusercontent.com/EQ-bench/creative-writing-bench/main/data/creative_writing_prompts_v3.json"


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    benchmark: str
    category: str
    prompt: str
    choices: tuple[str, ...] = ()
    answer: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    rubric: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["choices"] = list(self.choices)
        value["rubric"] = list(self.rubric)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BenchmarkItem:
        return cls(
            item_id=str(value["item_id"]),
            benchmark=str(value["benchmark"]),
            category=str(value.get("category", "")),
            prompt=str(value.get("prompt", "")),
            choices=tuple(str(choice) for choice in value.get("choices", [])),
            answer=value.get("answer"),
            metadata=dict(value.get("metadata") or {}),
            rubric=tuple(str(axis) for axis in value.get("rubric", [])),
        )


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


def _download(
    url: str, cache_dir: Path, *, name: str | None = None, timeout: float = 120
) -> tuple[bytes, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = name or _cache_key(url)
    data_path = cache_dir / f"{stem}.bin"
    metadata_path = cache_dir / f"{stem}.json"
    if data_path.exists() and metadata_path.exists():
        data = data_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") == sha256_bytes(data):
            return data, metadata
    request = urllib.request.Request(url, headers={"User-Agent": "ollama-novel-bench/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
            resolved_url = response.geturl()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"dataset download failed: {url}: {exc}") from exc
    metadata = {
        "url": url,
        "resolved_url": resolved_url,
        "fetched_at": utc_now(),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }
    temporary = data_path.with_suffix(".tmp")
    temporary.write_bytes(data)
    temporary.replace(data_path)
    atomic_write_text(metadata_path, json_dumps(metadata) + "\n")
    return data, metadata


def _safe_resolved_url(value: Any) -> str | None:
    if not value:
        return None
    parsed = urllib.parse.urlsplit(str(value))
    if parsed.scheme and parsed.netloc:
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    return str(value).split("?", 1)[0]


def _source(
    name: str,
    *,
    url: str,
    metadata: dict[str, Any],
    license_name: str,
    version: str,
    count: int | None = None,
) -> dict[str, Any]:
    result = {
        "name": name,
        "url": url,
        "version": version,
        "license": license_name,
        "sha256": metadata.get("sha256"),
        "resolved_url": _safe_resolved_url(metadata.get("resolved_url")),
    }
    if count is not None:
        result["records_available"] = count
    return result


def _parse_json_lines(data: bytes) -> list[dict[str, Any]]:
    records = []
    for line in data.decode("utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
    return records


def load_jcq(cache_dir: Path) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    data, metadata = _download(JCQ_URL, cache_dir, name="jcq-test")
    raw = _parse_json_lines(data)
    items = [
        BenchmarkItem(
            item_id=f"jcq-{record['id']}",
            benchmark="jcq",
            category=str(record["task"]),
            prompt=str(record["question"]),
            metadata={"source_id": record["id"]},
            rubric=("fluency", "flexibility", "originality", "elaboration"),
        )
        for record in raw
    ]
    return items, _source(
        "JCQ",
        url=JCQ_URL,
        metadata=metadata,
        license_name="CC BY 4.0",
        version="commit 7961ea8",
        count=len(items),
    )


def load_jcommonsenseqa(cache_dir: Path) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    data, metadata = _download(JCOMMONSENSEQA_URL, cache_dir, name="jcommonsenseqa-valid-v1.2")
    items = []
    for record in _parse_json_lines(data):
        choices = tuple(str(record[f"choice{i}"]) for i in range(5))
        items.append(
            BenchmarkItem(
                item_id=f"jcommonsenseqa-{record['q_id']}",
                benchmark="objective-ja",
                category="JCommonsenseQA",
                prompt=str(record["question"]),
                choices=choices,
                answer=int(record["label"]),
                metadata={"source_id": record["q_id"], "source_dataset": "JGLUE v1.2.0"},
            )
        )
    return items, _source(
        "JCommonsenseQA",
        url=JCOMMONSENSEQA_URL,
        metadata=metadata,
        license_name="CC BY-SA 4.0",
        version="JGLUE v1.2.0",
        count=len(items),
    )


def _answer_index(value: str) -> int:
    value = value.strip().upper()
    if value in "ABCD":
        return ord(value) - ord("A")
    if value.isdigit() and 0 <= int(value) < 4:
        return int(value)
    raise ValueError(f"unsupported JMMLU answer: {value!r}")


def load_jmmlu(cache_dir: Path) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    data, metadata = _download(JMMLU_URL, cache_dir, name="jmmlu")
    items: list[BenchmarkItem] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for filename in sorted(archive.namelist()):
            if not filename.startswith("JMMLU/test/") or not filename.endswith(".csv"):
                continue
            category = Path(filename).stem
            text = archive.read(filename).decode("utf-8-sig")
            for row_number, row in enumerate(csv.reader(io.StringIO(text))):
                if not row or row[0].strip().lower() in {"question", "問題"}:
                    continue
                if len(row) < 6:
                    continue
                choices = tuple(str(choice) for choice in row[1:5])
                try:
                    answer = _answer_index(row[5])
                except ValueError:
                    continue
                items.append(
                    BenchmarkItem(
                        item_id=f"jmmlu-{category}-{row_number:04d}",
                        benchmark="objective-ja",
                        category=f"JMMLU/{category}",
                        prompt=str(row[0]),
                        choices=choices,
                        answer=answer,
                        metadata={
                            "source_id": f"{category}:{row_number}",
                            "source_dataset": "JMMLU",
                        },
                    )
                )
    return items, _source(
        "JMMLU",
        url=JMMLU_URL,
        metadata=metadata,
        license_name="CC BY-NC-ND 4.0",
        version="Hugging Face main revision (sha256 recorded)",
        count=len(items),
    )


def _load_rows_page(
    dataset: str, config: str, split: str, offset: int, length: int = 100
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    # Import here to keep the module's import surface obvious in stack traces.
    url = f"{JAMC_ROWS_URL}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "ollama-novel-bench/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Hugging Face dataset-server request failed: {url}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"unexpected dataset-server response: {url}")
    return value


def load_jamcqa(
    cache_dir: Path, *, split: str = "test"
) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"jamcqa-v1.0-{split}.json"
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        rows = payload["rows"]
    else:
        rows = []
        offset = 0
        total = None
        while total is None or offset < total:
            page = _load_rows_page("sbintuitions/JamC-QA", "v1.0", split, offset)
            rows.extend(row["row"] for row in page.get("rows", []))
            total = int(page.get("num_rows_total", len(rows)))
            if not page.get("rows"):
                break
            offset += len(page["rows"])
        raw = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
        metadata = {
            "url": f"{JAMC_ROWS_URL}?dataset=sbintuitions/JamC-QA&config=v1.0&split={split}",
            "resolved_url": None,
            "fetched_at": utc_now(),
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
        }
        atomic_write_text(cache_path, json_dumps({"metadata": metadata, "rows": rows}) + "\n")
    items = []
    for record in rows:
        choices = tuple(str(record[f"choice{i}"]) for i in range(4))
        items.append(
            BenchmarkItem(
                item_id=f"jamcqa-{record['qid']}",
                benchmark="objective-ja",
                category=f"JamC-QA/{record['category']}",
                prompt=str(record["question"]),
                choices=choices,
                answer=int(record["answer_index"]),
                metadata={"source_id": record["qid"], "difficulty": record.get("difficulty")},
            )
        )
    return items, _source(
        "JamC-QA",
        url=metadata["url"],
        metadata=metadata,
        license_name="CC BY-SA 4.0",
        version="v1.0 / dataset-server snapshot",
        count=len(items),
    )


def load_eqcw(cache_dir: Path) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    data, metadata = _download(EQCW_PROMPTS_URL, cache_dir, name="eqcw-prompts-v3")
    raw = json.loads(data.decode("utf-8"))
    items = []
    for key, record in sorted(raw.items(), key=lambda pair: int(pair[0])):
        items.append(
            BenchmarkItem(
                item_id=f"eqcw-{key}",
                benchmark="eqcw",
                category=str(record.get("category", "creative writing")),
                prompt=str(record["writing_prompt"]),
                metadata={
                    "title": record.get("title"),
                    "seed_modifiers": record.get("seed_modifiers", []),
                },
                rubric=(
                    "craft",
                    "emotional_impact",
                    "characterization",
                    "coherence",
                    "originality",
                ),
            )
        )
    return items, _source(
        "EQ-Bench Creative Writing v3",
        url=EQCW_PROMPTS_URL,
        metadata=metadata,
        license_name="See upstream repository",
        version="main (sha256 recorded)",
        count=len(items),
    )


def load_novel_ja(root: Path) -> tuple[list[BenchmarkItem], dict[str, Any]]:
    path = root / "data" / "novel_ja.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for record in raw:
        items.append(
            BenchmarkItem(
                item_id=str(record["id"]),
                benchmark="novel-ja",
                category=str(record["category"]),
                prompt=str(record["prompt"]),
                metadata={"focus": record.get("focus"), "license": "MIT"},
                rubric=tuple(record.get("rubric", [])),
            )
        )
    source = {
        "name": "novel-ja",
        "url": "https://github.com/goura32/ollama-novel-bench/blob/main/data/novel_ja.json",
        "version": "repository dataset",
        "license": "MIT",
        "records_available": len(items),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return items, source


def deterministic_sample(
    items: Sequence[BenchmarkItem],
    *,
    count_per_category: int | None = None,
    count: int | None = None,
    seed: int = 20260912,
    categories: Iterable[str] | None = None,
) -> list[BenchmarkItem]:
    """Sample sorted records with a local PRNG; never depends on hash randomization."""
    allowed = set(categories) if categories is not None else None
    eligible = [
        item
        for item in sorted(items, key=lambda value: value.item_id)
        if allowed is None or item.category in allowed
    ]
    rng = random.Random(seed)
    if count_per_category is not None:
        selected: list[BenchmarkItem] = []
        grouped: dict[str, list[BenchmarkItem]] = {}
        for item in eligible:
            grouped.setdefault(item.category, []).append(item)
        for category in sorted(grouped):
            group = grouped[category]
            take = min(count_per_category, len(group))
            selected.extend(rng.sample(group, take))
        return sorted(selected, key=lambda value: value.item_id)
    if count is None or count >= len(eligible):
        return eligible
    return sorted(rng.sample(eligible, count), key=lambda value: value.item_id)


def render_item_prompt(item: BenchmarkItem) -> str:
    if item.choices:
        labels = "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(item.choices)
        )
        return (
            "次の日本語の四択問題に答えてください。選択肢の記号（A、B、C、D、E）のみを返してください。\n\n"
            f"問題:\n{item.prompt}\n\n選択肢:\n{labels}"
        )
    return (
        "以下の依頼に日本語で答えてください。依頼への回答本文だけを返し、前置きや採点に関する説明は不要です。\n\n"
        f"依頼:\n{item.prompt}"
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def score_multiple_choice(output: str, item: BenchmarkItem) -> tuple[float, int | None]:
    if not item.choices or not isinstance(item.answer, int):
        return 0.0, None
    text = output.strip()
    match = re.search(
        r"(?:^|[\s\[\(（「【])([A-EＡ-Ｅ])(?:$|[\s\]\)）】。、:：])", text, re.IGNORECASE
    )
    if match:
        token = match.group(1).translate(str.maketrans("ＡＢＣＤＥ", "ABCDE")).upper()
        prediction = ord(token) - ord("A")
        if 0 <= prediction < len(item.choices):
            return float(prediction == item.answer), prediction
    normalized = _normalize_text(text)
    for index, choice in enumerate(item.choices):
        if normalized == _normalize_text(choice):
            return float(index == item.answer), index
    return 0.0, None


def load_profile_items(
    root: Path,
    cache_dir: Path,
    profile_name: str,
    profile: dict[str, Any],
) -> tuple[list[BenchmarkItem], list[dict[str, Any]]]:
    """Load only the records requested by a profile and return source snapshots."""
    seed = int(profile.get("seed", 20260912))
    items: list[BenchmarkItem] = []
    sources: list[dict[str, Any]] = []
    objective = dict(profile.get("objective") or {})

    if objective.get("jmmlu", 0):
        records, source = load_jmmlu(cache_dir)
        items.extend(deterministic_sample(records, count=int(objective["jmmlu"]), seed=seed + 1))
        sources.append(source)
    if objective.get("jcommonsenseqa", 0):
        records, source = load_jcommonsenseqa(cache_dir)
        items.extend(
            deterministic_sample(records, count=int(objective["jcommonsenseqa"]), seed=seed + 2)
        )
        sources.append(source)
    if objective.get("jamcqa", 0):
        records, source = load_jamcqa(cache_dir, split="test" if profile_name == "full" else "dev")
        items.extend(deterministic_sample(records, count=int(objective["jamcqa"]), seed=seed + 3))
        sources.append(source)

    jcq_per_task = int(profile.get("jcq_per_task", 0))
    if jcq_per_task:
        records, source = load_jcq(cache_dir)
        items.extend(
            deterministic_sample(
                records,
                count_per_category=jcq_per_task,
                seed=seed + 4,
                categories=profile.get("jcq_categories"),
            )
        )
        sources.append(source)

    novel_count = int(profile.get("novel_count", 0))
    if novel_count:
        records, source = load_novel_ja(root)
        items.extend(deterministic_sample(records, count=novel_count, seed=seed + 5))
        sources.append(source)

    eqcw_count = int(profile.get("eqcw_count", 0))
    if eqcw_count:
        records, source = load_eqcw(cache_dir)
        selected = deterministic_sample(records, count=eqcw_count, seed=seed + 6)
        iterations = int(profile.get("eqcw_iterations", 1))
        for iteration in range(1, iterations + 1):
            for item in selected:
                metadata = dict(item.metadata)
                metadata["iteration"] = iteration
                items.append(
                    BenchmarkItem(
                        item_id=f"{item.item_id}-iter-{iteration}",
                        benchmark=item.benchmark,
                        category=item.category,
                        prompt=item.prompt,
                        choices=item.choices,
                        answer=item.answer,
                        metadata=metadata,
                        rubric=item.rubric,
                    )
                )
        sources.append(source)

    if not items:
        raise ValueError(f"profile {profile_name!r} selected no benchmark items")
    return sorted(items, key=lambda item: (item.benchmark, item.category, item.item_id)), sources
