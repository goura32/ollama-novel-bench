from novelbench.datasets import (
    BenchmarkItem,
    _source,
    deterministic_sample,
    score_multiple_choice,
)


def _items():
    return [
        BenchmarkItem(
            item_id=f"jcq-{i}",
            benchmark="jcq",
            category=f"task-{i % 2}",
            prompt=f"質問 {i}",
            choices=(),
            answer=None,
            metadata={},
        )
        for i in range(10)
    ]


def test_deterministic_sample_is_stable_and_balanced_by_category():
    first = deterministic_sample(_items(), count_per_category=2, seed=42)
    second = deterministic_sample(_items(), count_per_category=2, seed=42)

    assert [item.item_id for item in first] == [item.item_id for item in second]
    assert len(first) == 4
    assert {item.category for item in first} == {"task-0", "task-1"}


def test_multiple_choice_accepts_letter_or_exact_option():
    item = BenchmarkItem(
        item_id="q1",
        benchmark="objective-ja",
        category="JCommonsenseQA",
        prompt="何?",
        choices=("赤", "青", "緑", "黄"),
        answer=1,
        metadata={},
    )

    assert score_multiple_choice("B", item) == (1.0, 1)
    assert score_multiple_choice("青", item) == (1.0, 1)
    assert score_multiple_choice("A", item) == (0.0, 0)
    assert score_multiple_choice("わかりません", item) == (0.0, None)


def test_source_removes_query_from_resolved_download_url():
    source = _source(
        "dataset",
        url="https://example.com/data.zip",
        metadata={"sha256": "abc", "resolved_url": "https://cdn.example/data.zip?signed=redacted"},
        license_name="MIT",
        version="test",
    )

    assert source["resolved_url"] == "https://cdn.example/data.zip"
