import pytest

from novelbench.cli import select_run_models


@pytest.fixture
def profiles():
    return [
        {
            "name": "first",
            "capabilities": ["completion", "thinking"],
            "thinking": {"modes": []},
        },
        {"name": "second", "capabilities": ["completion"], "thinking": {"modes": []}},
    ]


def test_smoke_defaults_to_one_deterministic_model(profiles):
    selected = select_run_models(profiles, None, "smoke")
    assert [model["name"] for model in selected] == ["first"]


def test_smoke_rejects_multiple_explicit_models(profiles):
    with pytest.raises(ValueError, match="exactly one"):
        select_run_models(profiles, "first,second", "smoke")


def test_non_smoke_keeps_all_models_without_filter(profiles):
    selected = select_run_models(profiles, None, "quick")
    assert [model["name"] for model in selected] == ["first", "second"]
